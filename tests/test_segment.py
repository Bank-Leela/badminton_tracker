import numpy as np
import pandas as pd
import pytest

from config import Config, load_config
from segment import (
    SEGMENT_COLUMNS,
    _runs,
    classify_segments,
    detect_cuts,
    frame_signatures,
    merge_same_verdict,
    play_precision_recall,
    rally_spans,
    segment_video,
    split_rallies,
    write_contact_sheet,
)

CFG_SEG = load_config().segment


@pytest.fixture
def signatures(broadcast_video):
    return frame_signatures(broadcast_video["path"], load_config(), 0, broadcast_video["n_frames"])


def test_signatures_cover_every_frame(signatures, broadcast_video):
    assert signatures["frame"].tolist() == list(range(broadcast_video["n_frames"]))
    assert signatures["hist_diff"].iloc[0] == 0.0
    assert signatures["hist_diff"].between(0, 1).all()


def test_hist_diff_spikes_only_at_cuts(signatures, broadcast_video):
    spikes = signatures.loc[signatures["hist_diff"] > CFG_SEG.cut_threshold, "frame"].tolist()
    flash_a, flash_b = broadcast_video["flash"]
    assert spikes == broadcast_video["cuts"] + [flash_a, flash_b]


def test_detect_cuts_merges_the_flash(signatures, broadcast_video):
    spans = detect_cuts(signatures, broadcast_video["fps"], CFG_SEG)
    # Flash [200, 203) is shorter than min_segment_s and folds into its predecessor.
    assert spans == [(0, 90), (90, 150), (150, 203), (203, 260)]
    assert all(b > a for a, b in spans)


def test_detect_cuts_short_first_segment_folds_forward():
    sig = pd.DataFrame({"frame": range(100), "hist_diff": 0.0, "court_frac": 0.0, "line_frac": 0.0})
    sig.loc[3, "hist_diff"] = 0.9
    sig.loc[50, "hist_diff"] = 0.9
    assert detect_cuts(sig, 30.0, CFG_SEG) == [(0, 50), (50, 100)]


def test_play_view_classified_by_court_and_lines(signatures, broadcast_video):
    spans = detect_cuts(signatures, broadcast_video["fps"], CFG_SEG)
    camera = classify_segments(signatures, spans, CFG_SEG)
    assert camera["is_play"].tolist() == [1, 0, 1, 1]
    play, crowd = camera[camera.is_play == 1], camera[camera.is_play == 0]
    assert (play["court_frac"] > crowd["court_frac"].max()).all()
    assert (play["line_frac"] > crowd["line_frac"].max()).all()


def test_merge_same_verdict_renumbers_and_tiles():
    camera = pd.DataFrame(
        {"segment_id": [0, 1, 2, 3, 4], "start_frame": [0, 90, 150, 203, 300],
         "end_frame": [90, 150, 203, 300, 320], "is_play": [1, 0, 1, 1, 0]}
    )
    merged = merge_same_verdict(camera)
    assert merged["segment_id"].tolist() == [0, 1, 2, 3]
    assert list(zip(merged.start_frame, merged.end_frame)) == [(0, 90), (90, 150), (150, 300), (300, 320)]
    assert merged["is_play"].tolist() == [1, 0, 1, 0]


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


def test_segment_video_end_to_end(broadcast_video, tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}"])
    df = segment_video(cfg, "synthetic", broadcast_video["path"])
    out_dir = tmp_path / "cache" / "synthetic"
    assert (out_dir / "segments.csv").exists()
    assert (out_dir / "camera_segments.csv").exists()
    assert (out_dir / "frame_signatures.parquet").exists()
    # No shuttle.csv: play segments are whole, no rallies.
    assert list(zip(df.start_frame, df.end_frame, df.is_play, df.rally_id)) == [
        (0, 90, 1, -1), (90, 150, 0, -1), (150, 260, 1, -1)
    ]
    precision, recall = play_precision_recall(df, broadcast_video["play_spans"])
    assert (precision, recall) == (1.0, 1.0)

    # With a trajectory in the cache, play segments split into rallies. The
    # cached signatures are reused, so this does not decode again.
    visible = np.zeros(260, dtype=int)
    visible[10:70] = 1
    visible[160:250] = 1
    pd.DataFrame({"frame": range(260), "x": 0.0, "y": 0.0, "visible": visible, "confidence": 0.0}).to_csv(
        out_dir / "shuttle.csv", index=False
    )
    df = segment_video(cfg, "synthetic", broadcast_video["path"], force=True)
    assert df.loc[df.rally_id >= 0, "rally_id"].tolist() == [0, 1]
    assert (df.loc[df.rally_id >= 0, "segment_id"].tolist()) == [0, 2]

    # Cached: a second call returns the same rows without recomputing.
    again = segment_video(cfg, "synthetic", broadcast_video["path"])
    pd.testing.assert_frame_equal(again, df)


def test_contact_sheet_writes_an_image(broadcast_video, tmp_path):
    import cv2

    camera = pd.DataFrame(
        {"segment_id": [0, 1, 2], "start_frame": [0, 90, 150], "end_frame": [90, 150, 260],
         "court_frac": [0.5, 0.01, 0.5], "line_frac": [0.02, 0.0, 0.02], "is_play": [1, 0, 1]}
    )
    out = write_contact_sheet(broadcast_video["path"], camera, tmp_path / "sheet.png", 30.0, thumb_width=160, columns=2)
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] == 320 and img.shape[0] > 2 * 90
