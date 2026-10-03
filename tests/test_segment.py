import numpy as np
import pandas as pd
import pytest

import segment
from config import Config, load_config
from segment import (
    SEGMENT_COLUMNS,
    VIEW_COLUMNS,
    _runs,
    _unpack,
    frame_line_masks,
    play_precision_recall,
    play_spans,
    play_threshold,
    play_view_scores,
    rally_spans,
    segment_video,
    split_rallies,
    write_contact_sheet,
)

CFG_SEG = load_config().segment


@pytest.fixture
def masks(broadcast_video):
    return frame_line_masks(broadcast_video["path"], load_config(), 0, broadcast_video["n_frames"])


@pytest.fixture
def scored(masks):
    frames, packed, width = masks
    return play_view_scores(frames, packed, width, CFG_SEG)


def _scene_scores(scores, broadcast_video, kind):
    by = scores.set_index("frame")["score"]
    flash = range(*broadcast_video["flash"])
    return pd.concat(
        [by.loc[a : b - 1].drop(index=[f for f in flash if a <= f < b]) for k, a, b in broadcast_video["scenes"] if k == kind]
    )


def test_masks_cover_every_frame(masks, broadcast_video):
    frames, packed, width = masks
    assert frames.tolist() == list(range(broadcast_video["n_frames"]))
    assert width == CFG_SEG.mask_width
    assert _unpack(packed[:1], width).shape == (1, 180, 320)


def test_play_view_scores_separate_the_main_camera(scored, broadcast_video):
    scores, _, _ = scored
    play = _scene_scores(scores, broadcast_video, "play")
    assert play.min() > 0.85
    # The other camera has a green mat and white lines too — it is what the old
    # court-colour heuristic called play — but its layout does not match.
    assert _scene_scores(scores, broadcast_video, "other").max() < 0.4
    assert _scene_scores(scores, broadcast_video, "crowd").max() < 0.4
    # The flash is white everywhere: full recall, almost no precision.
    flash = scores.set_index("frame").loc[range(*broadcast_video["flash"]), "score"]
    assert flash.max() < 0.4


def test_template_excludes_the_score_graphic(scored, broadcast_video):
    _, template, overlay = scored
    x0, y0, x1, y1 = (v // 2 for v in broadcast_video["score_box"])  # 640 -> 320 px
    box = np.zeros_like(template)
    box[y0 + 1 : y1, x0 + 1 : x1] = True
    assert not (template & box).any()
    assert overlay[box].mean() > 0.9
    # Only the box: the other camera's lines must not be hidden as overlay,
    # or its low score would prove nothing about the layout match.
    assert overlay.sum() < 2 * box.sum()
    assert template.sum() > 300  # the court lines are there


def test_play_view_scores_survive_a_dominant_score_graphic():
    """All England 2019: the score graphic is white in more frames than the
    play view, whose lines are faint (each line pixel white in ~70% of play
    frames), and there are many dark close-ups showing only the graphic."""
    rng = np.random.default_rng(0)
    h, w = 180, 320
    lines = np.zeros((h, w), bool)
    lines[60, 60:260] = lines[170, 30:290] = True       # baselines
    lines[60:171, 160] = True                            # centre line
    for y in range(60, 171):                             # sidelines, in perspective
        lines[y, int(60 - (y - 60) * 30 / 110)] = lines[y, int(259 + (y - 60) * 30 / 110)] = True
    bug = np.zeros((h, w), bool)
    bug[10:20, 20:60] = True
    kinds = ["play"] * 250 + ["dark"] * 400 + ["crowd"] * 350
    masks = np.zeros((len(kinds), h, w), bool)
    for i, kind in enumerate(kinds):
        if kind == "play":
            masks[i] = lines & (rng.random((h, w)) < 0.7)
        elif kind == "crowd":
            masks[i] = rng.random((h, w)) < 0.05
        if kind != "play" or i % 2:                      # the graphic: always off play, half the time on it
            masks[i] |= bug
    scores, template, _ = play_view_scores(np.arange(len(kinds)), np.packbits(masks, axis=-1), w, CFG_SEG)
    play = np.array([k == "play" for k in kinds])
    s = scores["score"].to_numpy()
    assert s[play].min() > play_threshold(s, CFG_SEG) > s[~play].max()
    assert not (template & bug).any()


@pytest.mark.parametrize(
    "play_mode, other_tail, lo, hi",
    [
        ((0.70, 0.85), (0.00, 0.40), 0.40, 0.70),  # 2018-19: score graphic on screen, play scores lower
        ((0.75, 0.95), (0.00, 0.55), 0.55, 0.75),  # All England 2026: non-play tail reaches 0.55
        ((0.85, 1.00), (0.00, 0.35), 0.35, 0.85),  # 2026 Worlds / China Masters
    ],
)
def test_play_threshold_finds_the_valley(play_mode, other_tail, lo, hi):
    rng = np.random.default_rng(1)
    score = np.r_[rng.uniform(*other_tail, 7000), rng.uniform(*play_mode, 3000), rng.uniform(lo, hi, 15)]
    t = play_threshold(score, CFG_SEG)
    assert lo < t < hi
    assert play_threshold(score, Config({"play_score": 0.42})) == 0.42  # a number overrides auto


def test_play_spans_keep_only_the_main_match():
    """A stream opening on the end of the previous match, same court and camera."""
    fps = 30.0
    minute = int(60 * fps)
    score = np.full(30 * minute, 0.05)
    score[: minute] = 0.95                                 # previous match, 1 min of play
    for start in range(9 * minute, 29 * minute, 2 * minute):
        score[start : start + minute] = 0.95               # the match: play with 1-min gaps
    score[20 * minute : 24 * minute] = 0.05                # a 4-min break inside it stays
    scores = pd.DataFrame({"frame": np.arange(len(score)), "score": score})
    views = play_spans(scores, fps, CFG_SEG)
    play = views[views.is_play == 1]
    assert play.start_frame.min() == 9 * minute
    assert views.attrs["dropped"] == [(0, minute)]
    assert (play.start_frame >= 24 * minute).any()        # play after the 4-min break kept


def test_play_spans_drop_glimpses_shorter_than_play_min_s():
    score = np.r_[np.full(300, 0.05), np.full(30, 0.95), np.full(300, 0.05), np.full(120, 0.95)]
    scores = pd.DataFrame({"frame": np.arange(len(score)), "score": score})
    views = play_spans(scores, 30.0, CFG_SEG)  # 1 s glimpse, then a 4 s span
    assert list(zip(views.start_frame, views.end_frame, views.is_play)) == [(0, 630, 0), (630, 750, 1)]


def test_play_view_scores_fail_without_a_fixed_camera():
    rng = np.random.default_rng(0)
    noise = rng.random((60, 180, 320)) > 0.97
    packed = np.packbits(noise, axis=-1)
    with pytest.raises(RuntimeError, match="shares a fixed layout|dominant line layout|no fixed play camera"):
        play_view_scores(np.arange(60), packed, 320, CFG_SEG)


def test_play_spans_tile_the_range_and_absorb_a_flash():
    score = np.r_[np.full(90, 0.95), np.full(60, 0.05), np.full(50, 0.95), np.full(3, 0.1), np.full(57, 0.95)]
    scores = pd.DataFrame({"frame": np.arange(1000, 1260), "score": score})
    views = play_spans(scores, 30.0, CFG_SEG)
    assert list(views.columns) == VIEW_COLUMNS
    assert list(zip(views.start_frame, views.end_frame, views.is_play)) == [
        (1000, 1090, 1), (1090, 1150, 0), (1150, 1260, 1)
    ]
    assert views["segment_id"].tolist() == [0, 1, 2]
    assert views["score"].iloc[1] == pytest.approx(0.05)


def test_runs():
    assert _runs(np.array([0, 1, 1, 0, 1], dtype=bool)) == [(1, 3), (4, 5)]
    assert _runs(np.zeros(3, dtype=bool)) == []
    assert _runs(np.array([], dtype=bool)) == []


def test_rally_spans_bridge_gaps_and_drop_short_runs():
    fps = 30.0
    cfg = Config({"rally_smooth_s": 0.2, "rally_min_s": 1.0, "rally_max_gap_s": 1.0})
    visible = np.zeros(600, dtype=int)
    visible[100:200] = 1      # rally 1
    visible[210:300] = 1      # same rally: 10-frame gap (< 30) is bridged
    visible[400:410] = 1      # a toss, too short
    visible[450:540] = 1      # rally 2
    spans = rally_spans(visible, first_frame=1000, fps=fps, cfg_seg=cfg)
    assert len(spans) == 2
    (a0, b0), (a1, b1) = spans
    assert abs(a0 - 1100) <= 3 and abs(b0 - 1300) <= 3
    assert abs(a1 - 1450) <= 3 and abs(b1 - 1540) <= 3


def test_split_rallies_tiles_the_range_and_numbers_rallies():
    fps = 30.0
    cfg = Config({"rally_smooth_s": 0.2, "rally_min_s": 1.0, "rally_max_gap_s": 1.0})
    segments = pd.DataFrame(
        {"segment_id": [0, 1, 2], "start_frame": [0, 300, 400], "end_frame": [300, 400, 1000], "is_play": [1, 0, 1]}
    )
    visible = np.zeros(1000, dtype=int)
    visible[50:150] = 1
    visible[500:600] = 1
    visible[700:800] = 1
    shuttle = pd.DataFrame({"frame": range(1000), "visible": visible})
    df = split_rallies(segments, shuttle, fps, cfg)
    assert list(df.columns) == SEGMENT_COLUMNS
    assert df["start_frame"].iloc[0] == 0 and df["end_frame"].iloc[-1] == 1000
    assert (df["start_frame"].to_numpy()[1:] == df["end_frame"].to_numpy()[:-1]).all()
    assert df.loc[df.rally_id >= 0, "rally_id"].tolist() == [0, 1, 2]
    assert (df.loc[df.rally_id >= 0, "is_play"] == 1).all()
    # The non-play segment is untouched.
    row = df[df.segment_id == 1].iloc[0]
    assert (row.start_frame, row.end_frame, row.rally_id) == (300, 400, -1)


def test_rallies_bridge_a_short_cut_away_from_the_main_camera():
    """2018-19 broadcasts cut to a side camera for ~1 s mid-rally; between
    points the main camera is away for 10 s or more."""
    fps = 25.0
    cfg = Config({"rally_smooth_s": 0.2, "rally_min_s": 1.0, "rally_max_gap_s": 4.0})
    s = lambda sec: int(sec * fps)  # noqa: E731
    segments = pd.DataFrame({
        "segment_id": [0, 1, 2, 3, 4],
        "start_frame": [0, s(8), s(9), s(30), s(42)],
        "end_frame": [s(8), s(9), s(30), s(42), s(60)],
        "is_play": [1, 0, 1, 0, 1],
    })
    visible = np.zeros(s(60), dtype=int)
    visible[s(2):s(8)] = 1     # rally A, main camera ...
    visible[s(9):s(16)] = 1    # ... a 1 s side-camera cut, rally A continues
    visible[s(22):s(29)] = 1   # rally B, same play span, after a 6 s pause
    visible[s(43):s(55)] = 1   # rally C, after a 12 s cut-away
    shuttle = pd.DataFrame({"frame": range(s(60)), "visible": visible})
    df = split_rallies(segments, shuttle, fps, cfg)
    rallies = df[df.rally_id >= 0]
    assert rallies["rally_id"].tolist() == [0, 0, 1, 2]
    assert rallies["segment_id"].tolist() == [0, 2, 2, 4]


def test_split_rallies_without_shuttle_keeps_segments_whole():
    segments = pd.DataFrame({"segment_id": [0, 1], "start_frame": [0, 90], "end_frame": [90, 150], "is_play": [1, 0]})
    df = split_rallies(segments, None, 30.0, CFG_SEG)
    assert df["rally_id"].tolist() == [-1, -1]
    assert len(df) == 2


def test_play_precision_recall():
    segments = pd.DataFrame(
        {"segment_id": [0, 1, 2], "start_frame": [0, 100, 200], "end_frame": [100, 200, 300],
         "is_play": [1, 0, 1], "rally_id": [-1, -1, -1]}
    )
    p, r = play_precision_recall(segments, [(0, 100), (200, 300)])
    assert (p, r) == (1.0, 1.0)
    p, r = play_precision_recall(segments, [(0, 50), (200, 300), (150, 200)])
    assert p == pytest.approx(150 / 200)
    assert r == pytest.approx(150 / 200)


def test_segment_video_end_to_end(broadcast_video, tmp_path, monkeypatch):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}"])
    df = segment_video(cfg, "synthetic", broadcast_video["path"])
    out_dir = tmp_path / "cache" / "synthetic"
    for name in ["segments.csv", "segments.meta.json", "view_segments.csv", "view_scores.parquet",
                 "line_masks.npz", "line_template.png"]:
        assert (out_dir / name).exists(), name
    # No shuttle.csv: play spans are whole, no rallies. The other camera is not play.
    assert list(zip(df.start_frame, df.end_frame, df.is_play, df.rally_id)) == [
        (0, 90, 1, -1), (90, 150, 0, -1), (150, 260, 1, -1), (260, 300, 0, -1)
    ]
    precision, recall = play_precision_recall(df, broadcast_video["play_spans"])
    assert (precision, recall) == (1.0, 1.0)

    # From here on, nothing may decode again: the masks are cached.
    def no_decode(*args, **kwargs):
        raise AssertionError("decoded again")

    monkeypatch.setattr(segment, "frame_line_masks", no_decode)

    # A trajectory appearing in the cache invalidates segments.csv by itself —
    # no --force — and play spans split into rallies.
    visible = np.zeros(300, dtype=int)
    visible[10:70] = 1
    visible[160:250] = 1
    pd.DataFrame({"frame": range(300), "x": 0.0, "y": 0.0, "visible": visible, "confidence": 0.0}).to_csv(
        out_dir / "shuttle.csv", index=False
    )
    df = segment_video(cfg, "synthetic", broadcast_video["path"])
    # The two pieces are 3 s apart across a 2 s crowd cut — under
    # rally_max_gap_s (4 s), so they are one rally spanning both play spans.
    assert df.loc[df.rally_id >= 0, "rally_id"].tolist() == [0, 0]
    assert df.loc[df.rally_id >= 0, "segment_id"].tolist() == [0, 2]

    # Cached: a second call returns the same rows.
    again = segment_video(cfg, "synthetic", broadcast_video["path"])
    pd.testing.assert_frame_equal(again, df)

    # A changed setting reruns stages 2-3 from the cached masks (this was the
    # old trap: `-o segment.x=...` silently reused segments.csv).
    loose = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}", "segment.play_score=0.0"])
    everything = segment_video(loose, "synthetic", broadcast_video["path"])
    assert everything["is_play"].all()


def test_segment_video_rejects_masks_built_differently(broadcast_video, tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}"])
    segment_video(cfg, "synthetic", broadcast_video["path"], 0, 200)
    other = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}", "segment.white_min_val=200"])
    with pytest.raises(SystemExit, match="built with"):
        segment_video(other, "synthetic", broadcast_video["path"], 0, 200)
    with pytest.raises(SystemExit, match="covers frames"):
        segment_video(cfg, "synthetic", broadcast_video["path"], 0, 280)


def test_contact_sheet_writes_an_image(broadcast_video, tmp_path):
    import cv2

    views = pd.DataFrame(
        {"segment_id": [0, 1, 2], "start_frame": [0, 90, 150], "end_frame": [90, 150, 260],
         "is_play": [1, 0, 1], "score": [0.97, 0.02, 0.96]}
    )
    out = write_contact_sheet(broadcast_video["path"], views, tmp_path / "sheet.png", 30.0, thumb_width=160, columns=2)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] == 320 and img.shape[0] > 2 * 90
