"""Command line entry point.

    bda info    --video data/raw/match.mp4
    bda track   --video data/raw/match.mp4 --match-id msia_open_f --seconds 60
    bda overlay --video data/raw/match.mp4 --match-id msia_open_f
    bda segment --video data/raw/match.mp4 --match-id msia_open_f --sheet
    bda segment-eval --match-id msia_open_f --truth data/labels/msia_open_f.play.csv
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
    from shuttle import track_shuttle_cached

    cfg = load_config(args.config, args.overrides)
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
        camera = pd.read_csv(out_dir / "camera_segments.csv")
        out = write_contact_sheet(args.video, camera, out_dir / "segments.png", probe_video(args.video).fps)
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

    p_seg = sub.add_parser("segment", help="camera cuts, play-view and rally segmentation")
    p_seg.add_argument("--video", required=True)
    p_seg.add_argument("--match-id", required=True)
    p_seg.add_argument("--force", action="store_true", help="recompute even if cached")
    p_seg.add_argument("--sheet", action="store_true", help="also write a per-segment contact sheet")
    _add_range(p_seg)
    _add_common(p_seg)
    p_seg.set_defaults(func=cmd_segment)

    p_eval = sub.add_parser("segment-eval", help="precision/recall of play detection vs a truth CSV")
    p_eval.add_argument("--match-id", required=True)
    p_eval.add_argument("--truth", required=True, help="CSV with start_frame,end_frame rows of play view")
    _add_common(p_eval)
    p_eval.set_defaults(func=cmd_segment_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
