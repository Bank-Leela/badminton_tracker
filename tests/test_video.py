import numpy as np
import pandas as pd
import pytest

from video import extract_clip, iter_frames, probe_video, read_frames, sample_frames, write_overlay_video


def dot_x(frame):
    """Recover the dot's x position, which encodes the frame index."""
    ys, xs = np.where(frame[:, :, 0] > 200)
    return int(round(xs.mean()))


def test_probe(synthetic_video):
    info = probe_video(synthetic_video["path"])
    assert (info.width, info.height) == (synthetic_video["width"], synthetic_video["height"])
    assert info.fps == pytest.approx(synthetic_video["fps"], abs=0.01)
    assert info.frame_count == synthetic_video["n_frames"]


def test_iter_frames_indices_are_absolute(synthetic_video):
    pairs = list(iter_frames(synthetic_video["path"], 10, 15))
    assert [idx for idx, _ in pairs] == [10, 11, 12, 13, 14]
    # The dot encodes the frame index; this catches a seek landing elsewhere.
    assert [dot_x(f) for _, f in pairs] == [20 + 4 * i for i in range(10, 15)]


def test_iter_frames_past_seek_threshold_stays_aligned(synthetic_video):
    idx, frame = next(iter(iter_frames(synthetic_video["path"], 50)))
    assert idx == 50
    assert dot_x(frame) == 20 + 4 * 50


def test_iter_frames_to_end(synthetic_video):
    assert len(list(iter_frames(synthetic_video["path"], 55))) == 5


def test_empty_range_rejected(synthetic_video):
    with pytest.raises(ValueError):
        list(iter_frames(synthetic_video["path"], 10, 10))


def test_sample_frames_returns_exactly_the_requested_frames(synthetic_video):
    wanted = [0, 3, 4, 17, 40, 59]
    frames = sample_frames(synthetic_video["path"], wanted)
    assert [dot_x(f) for f in frames] == [20 + 4 * i for i in wanted]


def test_sample_frames_rejects_unsorted(synthetic_video):
    with pytest.raises(ValueError):
        sample_frames(synthetic_video["path"], [5, 2])
    assert sample_frames(synthetic_video["path"], []) == []


def test_read_frames_shape(synthetic_video):
    frames, first = read_frames(synthetic_video["path"], 4, 12)
    assert first == 4
    assert frames.shape == (8, synthetic_video["height"], synthetic_video["width"], 3)


def test_extract_clip(synthetic_video, tmp_path):
    out = extract_clip(synthetic_video["path"], tmp_path / "clip.mp4", 10, 20)
    assert probe_video(out).frame_count == 10


def test_overlay_writes_video(synthetic_video, tmp_path):
    traj = pd.DataFrame(
        {
            "frame": range(60),
            "x": [20.0 + 4 * i for i in range(60)],
            "y": [90.0] * 60,
            "visible": [1] * 60,
            "confidence": [0.9] * 60,
        }
    )
    out = write_overlay_video(synthetic_video["path"], traj, tmp_path / "ov.mp4", 0, 30)
    assert probe_video(out).frame_count == 30


def test_overlay_rejects_bad_columns(synthetic_video, tmp_path):
    with pytest.raises(ValueError):
        write_overlay_video(
            synthetic_video["path"], pd.DataFrame({"frame": [0]}), tmp_path / "ov.mp4"
        )
