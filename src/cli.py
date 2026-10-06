"""Command line entry point.

    bda info    --video data/raw/match.mp4
    bda track   --video data/raw/match.mp4 --match-id msia_open_f --seconds 60
    bda track   --video data/raw/match.mp4 --match-id msia_open_f --play-only   # after `segment`
    bda overlay --video data/raw/match.mp4 --match-id msia_open_f
    bda segment --video data/raw/match.mp4 --match-id msia_open_f --sheet
    bda segment-eval --match-id msia_open_f --truth data/labels/msia_open_f.play.csv
    bda court   --video data/raw/match.mp4 --match-id msia_open_f
    bda court-click --video data/raw/match.mp4 --match-id msia_open_f
    bda players --video data/raw/match.mp4 --match-id msia_open_f   # after `court`
    bda players-overlay --video data/raw/match.mp4 --match-id msia_open_f --start 30000 --seconds 30
    bda shots   --match-id msia_open_f                               # after `players`
    bda shots-review --video data/raw/match.mp4 --match-id msia_open_f --n 20
    bda label                                                        # then open http://127.0.0.1:8765
    bda outcomes                                                     # rally winners, all matches
    bda quality report                                               # phase 7: CV, acceptance, learning curve
    bda quality train  --source hand
    bda quality predict --source hand
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import cache_dir, load_config


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", default=None, help="config file (default: configs/default.yaml)")
    parser.add_argument(
        "-o",
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. -o device=cpu -o shuttle.batch_size=4",
    )


def _add_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=int, default=0, help="first frame (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="last frame (exclusive)")
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="length in seconds from --start; overrides --end",
    )


def _resolve_range(args, video_path: Path) -> tuple[int, int | None]:
    from video import probe_video

    if args.seconds is None:
        return args.start, args.end
    info = probe_video(video_path)
    return args.start, args.start + int(round(args.seconds * info.fps))


def cmd_info(args) -> int:
    from video import probe_video

    info = probe_video(args.video)
    print(
        f"{info.path.name}: {info.width}x{info.height} @ {info.fps:.3f} fps, "
        f"{info.frame_count} frames ({info.duration_s:.1f} s)"
    )
    return 0


def cmd_track(args) -> int:
    from shuttle import track_play_spans, track_shuttle_cached

    cfg = load_config(args.config, args.overrides)
    if args.play_only:
        if args.start or args.end is not None or args.seconds is not None or args.overlay:
            raise SystemExit("--play-only tracks the play spans from `bda segment`; it takes no range or "
                             "--overlay (render a stretch with `bda overlay --start ... --seconds ...`)")
        track_play_spans(cfg, args.match_id, args.video, force=args.force)
        return 0
    start, end = _resolve_range(args, Path(args.video))
    df = track_shuttle_cached(
        cfg,
        match_id=args.match_id,
        video_path=args.video,
        start_frame=start,
        end_frame=end,
        force=args.force,
    )
    detected = int(df["visible"].sum())
    print(f"{detected}/{len(df)} frames with a detection ({detected / len(df):.1%})")
    if args.overlay:
        _write_overlay(cfg, args, df, start, end)
    return 0


def cmd_overlay(args) -> int:
    cfg = load_config(args.config, args.overrides)
    traj_file = cache_dir(cfg, args.match_id) / "shuttle.csv"
    if not traj_file.exists():
        raise SystemExit(f"no trajectory at {traj_file}; run `bda track` first")
    df = pd.read_csv(traj_file)
    start, end = _resolve_range(args, Path(args.video))
    _write_overlay(cfg, args, df, start, end)
    return 0


def _write_overlay(cfg, args, df: pd.DataFrame, start: int, end: int | None) -> None:
    from video import write_overlay_video

    out = Path(args.out) if getattr(args, "out", None) else cache_dir(cfg, args.match_id) / "overlay.mp4"
    write_overlay_video(
        args.video,
        df,
        out,
        start_frame=start,
        end_frame=end,
        traj_len=int(cfg.get("overlay.traj_len", 8)),
        radius=int(cfg.get("overlay.radius", 3)),
        color=cfg.get("overlay.color", [0, 0, 255]),
    )
    print(f"overlay: wrote {out}")


def cmd_segment(args) -> int:
    from segment import segment_video, write_contact_sheet
    from video import probe_video

    cfg = load_config(args.config, args.overrides)
    start, end = _resolve_range(args, Path(args.video))
    segment_video(
        cfg,
        match_id=args.match_id,
        video_path=args.video,
        start_frame=start,
        end_frame=end,
        force=args.force,
    )
    if args.sheet:
        out_dir = cache_dir(cfg, args.match_id)
        views = pd.read_csv(out_dir / "view_segments.csv")
        out = write_contact_sheet(args.video, views, out_dir / "segments.png", probe_video(args.video).fps)
        print(f"sheet: wrote {out}")
    return 0


def cmd_segment_eval(args) -> int:
    """Precision/recall of play detection against a hand-marked truth file.

    The truth CSV has columns `start_frame, end_frame` (end exclusive), one
    row per span of play view.
    """
    from segment import play_precision_recall

    cfg = load_config(args.config, args.overrides)
    seg_file = cache_dir(cfg, args.match_id) / "segments.csv"
    if not seg_file.exists():
        raise SystemExit(f"no segments at {seg_file}; run `bda segment` first")
    truth = pd.read_csv(args.truth)
    spans = list(zip(truth["start_frame"].astype(int), truth["end_frame"].astype(int)))
    precision, recall = play_precision_recall(pd.read_csv(seg_file), spans)
    print(f"play-view detection: precision {precision:.1%}, recall {recall:.1%} (frame-level)")
    return 0


def cmd_court(args) -> int:
    from court import solve_homographies

    cfg = load_config(args.config, args.overrides)
    solve_homographies(cfg, args.match_id, args.video, force=args.force)
    return 0


def cmd_court_click(args) -> int:
    """Manual fallback: click the four doubles corners on the empty-court background."""
    import cv2

    from court import apply_manual_corners, click_corners

    cfg = load_config(args.config, args.overrides)
    bg_file = cache_dir(cfg, args.match_id) / "court_background.png"
    if not bg_file.exists():
        raise SystemExit(f"no {bg_file}; run `bda court` first (it writes the background even when the fit fails)")
    corners = click_corners(cv2.imread(str(bg_file)))
    if corners is None:
        print("court-click: aborted, nothing written")
        return 1
    apply_manual_corners(cfg, args.match_id, args.video, corners, args.segment or None)
    return 0


def cmd_players(args) -> int:
    from players import detect_poses, select_players

    cfg = load_config(args.config, args.overrides)
    detect_poses(cfg, args.match_id, args.video, force=args.force)
    if not args.detect_only:
        select_players(cfg, args.match_id, force=True)  # seconds; always redone
    return 0


def cmd_players_overlay(args) -> int:
    from players import write_players_overlay

    cfg = load_config(args.config, args.overrides)
    start, end = _resolve_range(args, Path(args.video))
    if end is None:
        raise SystemExit("give --seconds or --end: a whole match is hours of video")
    out = Path(args.out) if args.out else cache_dir(cfg, args.match_id) / "players_overlay.mp4"
    write_players_overlay(cfg, args.match_id, args.video, out, start, end)
    print(f"players-overlay: wrote {out}")
    return 0


def cmd_shots(args) -> int:
    """Phase 5: hits, the camera, each shot's flight, `shots.csv`."""
    from camera import solve_camera
    from contacts import detect_contacts
    from features import shot_features

    cfg = load_config(args.config, args.overrides)
    detect_contacts(cfg, args.match_id, force=args.force)
    solve_camera(cfg, args.match_id, force=args.force)
    shot_features(cfg, args.match_id)
    return 0


def cmd_shots_review(args) -> int:
    from review import write_shot_review

    cfg = load_config(args.config, args.overrides)
    write_shot_review(cfg, args.match_id, args.video, n=args.n, seed=args.seed)
    return 0


def cmd_label(args) -> int:
    """Phase 6: the keyboard-only labelling page on localhost."""
    from labeler import serve

    cfg = load_config(args.config, args.overrides)
    server = serve(cfg, args.match or None, args.order or cfg.labeler.order, args.seed, args.port, args.labeler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("labeler: stopped")
    finally:
        server.server_close()
    return 0


def _all_matches(cfg) -> list[str]:
    return sorted(p.parent.name for p in Path(cfg.paths.cache_dir).glob("*/shots.csv"))


def cmd_outcomes(args) -> int:
    """Rally winners and the score, read off the broadcast's score graphic (for phase 7)."""
    from outcomes import OutcomeError, check_scores, find_outcomes, remap

    cfg = load_config(args.config, args.overrides)
    if args.check:
        res = check_scores(cfg, Path(__file__).resolve().parents[1] / "docs" / "match_scores.csv")
        if res.empty:
            print("outcomes: no match with a known score has a rallies.csv yet (run `bda outcomes`)")
            return 0
        print(res.to_string(index=False))
        print(f"outcomes: {res['decided'].sum()}/{res['rallies'].sum()} rallies decided, "
              f"{res['points_off'].sum()} points off over {len(res)} matches with a known score")
        return 0
    failed = []
    for m in args.match or _all_matches(cfg):
        try:
            if args.remap:
                remap(cfg, m)
            else:
                find_outcomes(cfg, m, force=args.force)
        except (OutcomeError, FileNotFoundError) as e:
            print(f"outcomes: {e}")
            failed.append(m)
    if failed:
        print(f"outcomes: no score graphic read in {len(failed)} match(es): {', '.join(failed)}")
    return 1 if failed else 0


def cmd_quality(args) -> int:
    """Phase 7: the shot-quality model — report (CV, acceptance, learning curve), train, predict."""
    import quality

    cfg = load_config(args.config, args.overrides)
    matches = args.match or None
    if args.action == "report":
        quality.report(cfg, matches=matches)
    elif args.action == "train":
        try:
            meta = quality.train_final(cfg, source=args.source, matches=matches)
        except quality.TooFewLabels as e:
            print(f"quality: {e}")
            return 1
        print(f"quality: trained on {meta['n_train']} {args.source} labels")
    else:
        out = quality.predict(cfg, source=args.source, matches=matches)
        print(f"quality: wrote quality.csv for {len(out)} match(es)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bda", description="badminton match analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="print video metadata")
    p_info.add_argument("--video", required=True)
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    p_track = sub.add_parser("track", help="run TrackNet and cache the shuttle trajectory")
    p_track.add_argument("--video", required=True)
    p_track.add_argument("--match-id", required=True)
    p_track.add_argument("--force", action="store_true", help="recompute even if cached")
    p_track.add_argument("--overlay", action="store_true", help="also render the overlay video")
    p_track.add_argument("--play-only", action="store_true",
                         help="track only the play spans found by `bda segment` (whole match, ~30%% of frames)")
    p_track.add_argument("--out", default=None, help="overlay output path")
    _add_range(p_track)
    _add_common(p_track)
    p_track.set_defaults(func=cmd_track)

    p_overlay = sub.add_parser("overlay", help="render a cached trajectory onto the video")
    p_overlay.add_argument("--video", required=True)
    p_overlay.add_argument("--match-id", required=True)
    p_overlay.add_argument("--out", default=None)
    _add_range(p_overlay)
    _add_common(p_overlay)
    p_overlay.set_defaults(func=cmd_overlay)

    p_seg = sub.add_parser("segment", help="play-view and rally segmentation")
    p_seg.add_argument("--video", required=True)
    p_seg.add_argument("--match-id", required=True)
    p_seg.add_argument("--force", action="store_true", help="recompute even if cached")
    p_seg.add_argument("--sheet", action="store_true", help="also write a per-span contact sheet")
    _add_range(p_seg)
    _add_common(p_seg)
    p_seg.set_defaults(func=cmd_segment)

    p_eval = sub.add_parser("segment-eval", help="precision/recall of play detection vs a truth CSV")
    p_eval.add_argument("--match-id", required=True)
    p_eval.add_argument("--truth", required=True, help="CSV with start_frame,end_frame rows of play view")
    _add_common(p_eval)
    p_eval.set_defaults(func=cmd_segment_eval)

    p_court = sub.add_parser("court", help="fit the court homography for each play span (needs `segment`)")
    p_court.add_argument("--video", required=True)
    p_court.add_argument("--match-id", required=True)
    p_court.add_argument("--force", action="store_true", help="recompute even if cached; drops manual entries")
    _add_common(p_court)
    p_court.set_defaults(func=cmd_court)

    p_click = sub.add_parser("court-click", help="manual fallback: click the four doubles corners")
    p_click.add_argument("--video", required=True)
    p_click.add_argument("--match-id", required=True)
    p_click.add_argument("--segment", type=int, action="append", default=[],
                         help="play segment id to apply to (repeatable); default: every span that needs it")
    _add_common(p_click)
    p_click.set_defaults(func=cmd_court_click)

    p_players = sub.add_parser("players", help="pose + tracks of both players over the play spans (needs `court`)")
    p_players.add_argument("--video", required=True)
    p_players.add_argument("--match-id", required=True)
    p_players.add_argument("--force", action="store_true", help="redetect even if cached (~20 min a match)")
    p_players.add_argument("--detect-only", action="store_true", help="stop after poses.parquet")
    _add_common(p_players)
    p_players.set_defaults(func=cmd_players)

    p_pover = sub.add_parser("players-overlay", help="render players.parquet onto a stretch of the video")
    p_pover.add_argument("--video", required=True)
    p_pover.add_argument("--match-id", required=True)
    p_pover.add_argument("--out", default=None, help="default: data/cache/<match_id>/players_overlay.mp4")
    _add_range(p_pover)
    _add_common(p_pover)
    p_pover.set_defaults(func=cmd_players_overlay)

    p_shots = sub.add_parser("shots", help="hits, camera, 3D flights and shots.csv (needs `players`)")
    p_shots.add_argument("--match-id", required=True)
    p_shots.add_argument("--force", action="store_true", help="redo contacts and the camera even if cached")
    _add_common(p_shots)
    p_shots.set_defaults(func=cmd_shots)

    p_review = sub.add_parser("shots-review", help="sheets of random shots to check against the video")
    p_review.add_argument("--video", required=True)
    p_review.add_argument("--match-id", required=True)
    p_review.add_argument("--n", type=int, default=20)
    p_review.add_argument("--seed", type=int, default=0)
    _add_common(p_review)
    p_review.set_defaults(func=cmd_shots_review)

    p_label = sub.add_parser("label", help="label shot outcomes in the browser, keyboard only (needs `shots`)")
    p_label.add_argument("--match", action="append", default=[], help="only this match (repeatable); default: all")
    p_label.add_argument("--order", choices=["random", "rally"], default=None,
                         help="rally: match by match, rally by rally, in play order; random: one shuffle over "
                              "all shots (default: labeler.order in the config)")
    p_label.add_argument("--seed", type=int, default=0, help="the shuffle; keep it to resume the same order")
    p_label.add_argument("--port", type=int, default=8765)
    p_label.add_argument("--labeler", default="bank", help="who is labelling (stored with each label)")
    _add_common(p_label)
    p_label.set_defaults(func=cmd_label)

    p_out = sub.add_parser("outcomes", help="rally winners and the score from the score graphic (needs `shots`)")
    p_out.add_argument("--match", action="append", default=[], help="only this match (repeatable); default: all")
    p_out.add_argument("--force", action="store_true", help="recompute even if cached")
    p_out.add_argument("--remap", action="store_true", help="only redo the rows -> players match (no video)")
    p_out.add_argument("--check", action="store_true", help="compare the games read with docs/match_scores.csv")
    _add_common(p_out)
    p_out.set_defaults(func=cmd_outcomes)

    p_q = sub.add_parser("quality", help="phase 7: the shot-quality model (needs `outcomes`, labels)")
    p_q.add_argument("action", choices=["report", "train", "predict"],
                     help="report: CV by match, the acceptance check, the learning curve; "
                          "train: fit and save the model; predict: write quality.csv per match")
    p_q.add_argument("--source", choices=["hand", "baseline"], default="hand",
                     help="labels: yours (default) or the free rally-outcome baseline")
    p_q.add_argument("--match", action="append", default=[], help="only this match (repeatable); default: all")
    _add_common(p_q)
    p_q.set_defaults(func=cmd_quality)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
