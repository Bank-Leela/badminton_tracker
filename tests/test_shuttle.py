from pathlib import Path

import numpy as np
import pytest

from config import load_config
from shuttle import (
    SHUTTLE_COLUMNS,
    _ensemble_weight,
    _localise,
    load_tracknet,
    resolve_device,
    track_shuttle,
)

CKPT = Path(load_config().paths.tracknet_ckpt)
needs_ckpt = pytest.mark.skipif(not CKPT.exists(), reason="TrackNet checkpoint not downloaded")


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


@needs_ckpt
def test_checkpoint_loads_on_cpu():
    model = load_tracknet(load_config(), device=resolve_device("cpu"))
    assert model.seq_len > 0
    assert model.bg_mode in {"", "subtract", "subtract_concat", "concat"}


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
def test_track_shuttle_rejects_undersized_chunks(synthetic_video):
    cfg = load_config(overrides=["device=cpu"])
    with pytest.raises(ValueError):
        track_shuttle(synthetic_video["path"], cfg, 0, 40, chunk_frames=8)
