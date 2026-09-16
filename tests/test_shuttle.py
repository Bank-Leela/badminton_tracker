from pathlib import Path

import numpy as np
import pytest
import torch

from config import load_config
from shuttle import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SHUTTLE_COLUMNS,
    TrackNetModel,
    _background_median,
    _build_input,
    _ensemble_heatmaps,
    _ensemble_weight,
    _localise,
    _median_to_input,
    _read_frames_resized,
    _resize_bgr_to_chw,
    _tracknet_importable,
    _window_indices,
    load_tracknet,
    resolve_device,
    track_shuttle,
)

CFG = load_config()
CKPT = Path(CFG.paths.tracknet_ckpt)
REPO = Path(CFG.paths.tracknet_repo)
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="TrackNet checkpoint not downloaded")
needs_repo = pytest.mark.skipif(not REPO.is_dir(), reason="TrackNetV3 not cloned")


def test_ensemble_weight_is_symmetric_and_normalised():
    w = _ensemble_weight(8, "weight")
    assert w.sum() == pytest.approx(1.0)
    # Symmetry is what lets the accumulator index the weight by window position.
    assert np.allclose(w, w[::-1])
    assert w.argmax() in (3, 4)


def test_ensemble_weight_modes():
    assert np.allclose(_ensemble_weight(8, "nonoverlap"), np.ones(8))
    assert _ensemble_weight(8, "average").sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        _ensemble_weight(8, "bogus")


def test_localise_finds_the_blob_centre():
    hm = np.zeros((288, 512), dtype=np.float64)
    hm[100:106, 200:210] = 0.9
    cx, cy, conf, vis = _localise(hm, 0.5)
    assert vis == 1
    assert (cx, cy) == pytest.approx((205.0, 103.0), abs=1.0)
    assert conf == pytest.approx(0.9)


def test_localise_reports_no_detection_but_keeps_the_peak():
    hm = np.full((288, 512), 0.3)
    cx, cy, conf, vis = _localise(hm, 0.5)
    assert (vis, cx, cy) == (0, 0.0, 0.0)
    assert conf == pytest.approx(0.3)


def test_localise_picks_the_largest_blob():
    hm = np.zeros((288, 512))
    hm[10:12, 10:12] = 0.99      # small but brighter
    hm[100:120, 100:130] = 0.6   # large
    cx, cy, _, vis = _localise(hm, 0.5)
    assert vis == 1
    assert (cx, cy) == pytest.approx((115.0, 110.0), abs=1.0)


def test_resolve_device_honours_explicit_name():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}


# --- Preprocessing equivalence with TrackNetV3's own dataset -----------------
#
# The wrapper builds network inputs itself instead of going through their
# Shuttlecock_Trajectory_Dataset (which is ~5x too slow). These tests are the
# contract: the tensors handed to the network must match theirs.


def _textured_frames(n=12, h=180, w=320, seed=0):
    """RGB uint8 frames with structure, so a resize mismatch cannot hide."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        f = np.roll(base, shift=3 * i, axis=1).copy()
        f[40:60, 20 + 5 * i : 30 + 5 * i] = 255
        frames.append(f)
    return np.stack(frames)


def test_window_indices_overlap_and_nonoverlap():
    assert _window_indices(10, 8, "weight").tolist() == [list(range(0, 8)), list(range(1, 9)), list(range(2, 10))]
    nonoverlap = _window_indices(10, 8, "nonoverlap")
    assert nonoverlap.shape == (2, 8)
    assert nonoverlap[0].tolist() == list(range(8))
    # The last window is padded by repeating the final frame.
    assert nonoverlap[1].tolist() == [8, 9, 9, 9, 9, 9, 9, 9]
    with pytest.raises(ValueError):
        _window_indices(5, 8, "weight")


@needs_repo
@pytest.mark.parametrize("eval_mode,padding", [("weight", False), ("nonoverlap", True)])
def test_window_indices_match_their_dataset(eval_mode, padding):
    rgb = _textured_frames(n=19)
    with _tracknet_importable(REPO):
        from dataset import Shuttlecock_Trajectory_Dataset

        ds = Shuttlecock_Trajectory_Dataset(
            seq_len=8, sliding_step=8 if eval_mode == "nonoverlap" else 1, data_mode="heatmap",
            bg_mode="", frame_arr=rgb, padding=padding,
        )
    theirs = ds.data_dict["id"][:, :, 1]
    assert _window_indices(len(rgb), 8, eval_mode).tolist() == theirs.tolist()


@needs_repo
@pytest.mark.parametrize("bg_mode", ["concat", ""])
def test_build_input_matches_their_dataset(bg_mode):
    rgb = _textured_frames(n=12)
    median_rgb = np.median(rgb, axis=0)
    with _tracknet_importable(REPO):
        from dataset import Shuttlecock_Trajectory_Dataset

        ds = Shuttlecock_Trajectory_Dataset(
            seq_len=8, sliding_step=1, data_mode="heatmap", bg_mode=bg_mode,
            frame_arr=rgb, padding=False, median=median_rgb if bg_mode else None,
        )

    bgr = rgb[..., ::-1]
    frames_small = np.stack([_resize_bgr_to_chw(f) for f in bgr])
    assert frames_small.shape == (12, 3, INPUT_HEIGHT, INPUT_WIDTH)
    frames = torch.from_numpy(frames_small)
    median = torch.from_numpy(_median_to_input(median_rgb)).float() / 255.0 if bg_mode else None
    idx = torch.from_numpy(_window_indices(12, 8, "weight"))
    ours = _build_input(frames, median, idx).numpy()

    for i in range(len(ds)):
        _, theirs = ds[i]
        assert ours[i].shape == theirs.shape
        np.testing.assert_allclose(ours[i], theirs.astype(np.float32), atol=1e-6)


def test_read_frames_resized_is_decode_then_resize(synthetic_video):
    import cv2

    small = _read_frames_resized(synthetic_video["path"], 5, 9)
    assert small.shape == (4, 3, INPUT_HEIGHT, INPUT_WIDTH)
    cap = cv2.VideoCapture(str(synthetic_video["path"]))
    for _ in range(6):
        ok, frame = cap.read()
    cap.release()
    np.testing.assert_array_equal(small[0], _resize_bgr_to_chw(frame))


class _FakeNet(torch.nn.Module):
    """Heatmap = window-frame brightness + a per-position offset.

    The offset makes the temporal weights matter, so the ensemble arithmetic is
    actually tested rather than cancelled out.
    """

    def __init__(self, seq_len, n_median_ch):
        super().__init__()
        self.seq_len = seq_len
        self.skip = n_median_ch

    def forward(self, x):
        b, c, h, w = x.shape
        frames = x[:, self.skip :].reshape(b, self.seq_len, 3, h, w).mean(dim=2)
        pos = torch.arange(self.seq_len, device=x.device, dtype=x.dtype) * 0.01
        return frames + pos[None, :, None, None]


@pytest.mark.parametrize("eval_mode", ["weight", "average", "nonoverlap"])
def test_ensemble_matches_numpy_reference(eval_mode):
    seq_len, n = 4, 11
    rng = np.random.default_rng(1)
    frames_small = rng.integers(0, 256, size=(n, 3, 6, 8), dtype=np.uint8)
    median = rng.integers(0, 256, size=(3, 6, 8), dtype=np.uint8)
    model = TrackNetModel(_FakeNet(seq_len, 3), seq_len, "concat", torch.device("cpu"))

    got = _ensemble_heatmaps(model, frames_small, median, batch_size=3, eval_mode=eval_mode)

    weight = _ensemble_weight(seq_len, eval_mode)
    brightness = frames_small.astype(np.float64).mean(axis=1) / 255.0  # (n, h, w)
    acc = np.zeros_like(brightness)
    acc_w = np.zeros(n)
    for window in _window_indices(n, seq_len, eval_mode):
        for pos, f in enumerate(window):
            acc[f] += weight[pos] * (brightness[f] + 0.01 * pos)
            acc_w[f] += weight[pos]
    np.testing.assert_allclose(got, acc / acc_w[:, None, None], atol=1e-5)


# --- Needs the checkpoint ----------------------------------------------------


@needs_ckpt
def test_checkpoint_loads_on_cpu():
    model = load_tracknet(load_config(), device=resolve_device("cpu"))
    assert model.seq_len > 0
    assert model.bg_mode in {"", "concat"}


@needs_ckpt
def test_track_shuttle_covers_the_range_and_holds_invariants(synthetic_video):
    cfg = load_config(overrides=["device=cpu", "shuttle.batch_size=4"])
    df = track_shuttle(
        synthetic_video["path"], cfg, start_frame=0, end_frame=48, chunk_frames=32
    )
    assert list(df.columns) == SHUTTLE_COLUMNS
    # One row per frame, no gaps, no duplicates across the chunk seam.
    assert df["frame"].tolist() == list(range(48))
    assert df["confidence"].between(0.0, 1.0).all()


@needs_ckpt
def test_chunking_does_not_change_the_result(synthetic_video):
    """Interior frames must get the same answer whether or not a chunk seam falls on them."""
    cfg = load_config(overrides=["device=cpu", "shuttle.batch_size=4"])
    model = load_tracknet(cfg, device=resolve_device("cpu"))
    one = track_shuttle(synthetic_video["path"], cfg, 0, 48, model=model, chunk_frames=48)
    two = track_shuttle(synthetic_video["path"], cfg, 0, 48, model=model, chunk_frames=32)
    np.testing.assert_allclose(one[["x", "y", "confidence"]].to_numpy(), two[["x", "y", "confidence"]].to_numpy(), atol=1e-4)
    assert one["visible"].tolist() == two["visible"].tolist()


@needs_ckpt
def test_track_shuttle_rejects_undersized_chunks(synthetic_video):
    cfg = load_config(overrides=["device=cpu"])
    with pytest.raises(ValueError):
        track_shuttle(synthetic_video["path"], cfg, 0, 40, chunk_frames=8)
