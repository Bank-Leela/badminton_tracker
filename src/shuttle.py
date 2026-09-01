"""TrackNetV3 wrapper.

`external/TrackNetV3` is a frozen dependency: it is never edited and never
imported by anything except this module. Everything the rest of the pipeline
knows about the shuttle comes out of `track_shuttle()` as a DataFrame with
columns `frame, x, y, visible, confidence`, in source-video pixel coordinates.

Two deliberate differences from their `predict.py`:

* Confidence. Their `predict()` thresholds the heatmap and reports only a 0/1
  visibility flag. Contact detection in phase 5 needs to know how sure the
  model was, so the peak heatmap response at the detected blob is kept.
* Edge frames. Their temporal ensemble divides early frames by an occurrence
  count and later frames by the full weight sum. Here every frame is divided
  by the weight actually accumulated for it, which is the same in the interior
  and better defined at the ends.

InpaintNet (the trajectory-inpainting half of TrackNetV3) is not run. It
replaces missing detections with interpolated coordinates that carry no
heatmap evidence, and phase 5 wants the gaps visible rather than filled in.
"""

from __future__ import annotations

import os

# Must be set before torch is imported: lets unsupported ops fall back to CPU
# instead of raising, which matters on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import contextlib
import sys
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, cache_dir
from video import iter_frames, probe_video, read_frames

SHUTTLE_COLUMNS = ["frame", "x", "y", "visible", "confidence"]

# Frames decoded and held in memory at once. A 1080p frame is ~6 MB, so this
# is the main memory knob. Consecutive chunks overlap by seq_len - 1 frames so
# the temporal ensemble has full context across the seam.
DEFAULT_CHUNK_FRAMES = 300

# Frames sampled across the requested range to build the background median
# image. Odd, so np.median never has to average two middle values.
MEDIAN_SAMPLE_NUM = 121


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device name, preferring CUDA, then MPS, then CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextlib.contextmanager
def _tracknet_importable(repo_dir: str | Path) -> Iterator[None]:
    """Put the TrackNetV3 checkout on sys.path for the duration of the block.

    Their modules import each other by bare top-level name (`model`,
    `dataset`, `utils.general`), so the checkout root has to be importable.
    It is inserted at the front and removed afterwards; module names imported
    from it stay in sys.modules, which is why nothing in this repo may take
    those names.
    """
    repo_dir = str(Path(repo_dir).resolve())
    if not Path(repo_dir).is_dir():
        raise FileNotFoundError(
            f"TrackNetV3 checkout not found at {repo_dir}. "
            "Clone it: git clone https://github.com/qaz812345/TrackNetV3.git external/TrackNetV3"
        )
    sys.path.insert(0, repo_dir)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(repo_dir)


class TrackNetModel:
    """A loaded TrackNet checkpoint plus the input parameters it was trained with."""

    def __init__(self, net, seq_len: int, bg_mode: str, device: torch.device):
        self.net = net
        self.seq_len = seq_len
        self.bg_mode = bg_mode
        self.device = device


def load_tracknet(cfg: Config, device: torch.device | None = None) -> TrackNetModel:
    ckpt_path = Path(cfg.paths.tracknet_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"TrackNet checkpoint not found at {ckpt_path}. "
            "Download TrackNetV3_ckpts.zip (see external/TrackNetV3/README.md) "
            "and unzip it into external/TrackNetV3/ckpts/"
        )
    device = device or resolve_device(cfg.get("device", "auto"))

    with _tracknet_importable(cfg.paths.tracknet_repo):
        from utils.general import get_model

        # weights_only=False: the checkpoint stores a param_dict alongside the
        # weights. The file is the project's own published checkpoint.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        seq_len = int(ckpt["param_dict"]["seq_len"])
        bg_mode = ckpt["param_dict"]["bg_mode"]
        net = get_model("TrackNet", seq_len, bg_mode)
        net.load_state_dict(ckpt["model"])

    net.to(device).eval()
    return TrackNetModel(net, seq_len, bg_mode, device)


def _ensemble_weight(seq_len: int, eval_mode: str) -> np.ndarray:
    """Per-position weight for the temporal ensemble.

    `nonoverlap` sees each frame once, so every position weighs the same.
    """
    if eval_mode == "nonoverlap":
        return np.ones(seq_len, dtype=np.float64)
    with_center_peak = np.ones(seq_len, dtype=np.float64)
    for i in range((seq_len + 1) // 2):
        with_center_peak[i] = i + 1
        with_center_peak[seq_len - i - 1] = i + 1
    if eval_mode == "average":
        return np.ones(seq_len, dtype=np.float64) / seq_len
    if eval_mode == "weight":
        return with_center_peak / with_center_peak.sum()
    raise ValueError(f"unknown eval_mode {eval_mode!r}")


def _background_median(
    video_path: str | Path, start_frame: int, end_frame: int
) -> np.ndarray:
    """Median image over frames sampled evenly across the range.

    Computed once for the whole range and reused for every chunk, so chunking
    cannot change the model's input.
    """
    n = end_frame - start_frame
    step = max(1, n // MEDIAN_SAMPLE_NUM)
    wanted = set(range(start_frame, end_frame, step))
    samples = [f for i, f in iter_frames(video_path, start_frame, end_frame) if i in wanted]
    if not samples:
        raise RuntimeError(f"no frames sampled for median over [{start_frame}, {end_frame})")
    # BGR -> RGB, matching the frame order handed to the dataset.
    return np.median(np.stack(samples)[..., ::-1], axis=0)


def _localise(heatmap: np.ndarray, threshold: float) -> tuple[float, float, float, int]:
    """Blob centre, peak response and visibility from one heatmap.

    Mirrors TrackNetV3's `predict_location`: binarise, take the largest
    connected component by bounding-box area, report its centre.
    """
    mask = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, float(heatmap.max()), 0
    x, y, w, h = max((cv2.boundingRect(c) for c in contours), key=lambda r: r[2] * r[3])
    peak = float(heatmap[y : y + h, x : x + w].max())
    return x + w / 2, y + h / 2, peak, 1


def _run_chunk(
    model: TrackNetModel,
    frames_bgr: np.ndarray,
    chunk_start: int,
    median: np.ndarray | None,
    batch_size: int,
    eval_mode: str,
    threshold: float,
    img_scaler: tuple[float, float],
    dataset_cls,
    progress: tqdm | None,
) -> dict[int, tuple[float, float, float, int]]:
    """Run TrackNet over one decoded chunk, returning per-absolute-frame results.

    Heatmaps are accumulated per frame across every window that frame appears
    in, then localised once the frame can no longer be touched by a later
    window. That keeps memory at O(seq_len) heatmaps rather than O(chunk).
    """
    seq_len = model.seq_len
    sliding_step = seq_len if eval_mode == "nonoverlap" else 1
    weight = _ensemble_weight(seq_len, eval_mode)

    dataset = dataset_cls(
        seq_len=seq_len,
        sliding_step=sliding_step,
        data_mode="heatmap",
        bg_mode=model.bg_mode,
        # Their dataset expects RGB.
        frame_arr=frames_bgr[..., ::-1],
        padding=(eval_mode == "nonoverlap"),
        median=median,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    acc: dict[int, np.ndarray] = {}
    acc_w: dict[int, float] = {}
    results: dict[int, tuple[float, float, float, int]] = {}

    def flush(up_to_local: int) -> None:
        for local in sorted(k for k in acc if k <= up_to_local):
            hm = acc.pop(local) / acc_w.pop(local)
            cx, cy, conf, vis = _localise(hm, threshold)
            results[chunk_start + local] = (
                cx * img_scaler[0],
                cy * img_scaler[1],
                conf,
                vis,
            )

    for indices, x in loader:
        with torch.no_grad():
            y_pred = model.net(x.float().to(model.device)).detach().cpu().numpy()
        indices = indices.numpy()

        last_window_start = -1
        for n in range(indices.shape[0]):
            local_frames = indices[n][:, 1]
            last_window_start = max(last_window_start, int(local_frames[0]))
            for pos in range(seq_len):
                local = int(local_frames[pos])
                hm = y_pred[n][pos].astype(np.float64)
                if local in acc:
                    acc[local] += weight[pos] * hm
                    acc_w[local] += weight[pos]
                else:
                    acc[local] = weight[pos] * hm
                    acc_w[local] = weight[pos]
        # A frame is final once the window that starts on it has been seen:
        # no later window can contain it.
        flush(last_window_start)
        if progress is not None:
            progress.update(indices.shape[0] * sliding_step)

    flush(max(acc) if acc else -1)
    return results


def track_shuttle(
    video_path: str | Path,
    cfg: Config,
    start_frame: int = 0,
    end_frame: int | None = None,
    model: TrackNetModel | None = None,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> pd.DataFrame:
    """Track the shuttle over `[start_frame, end_frame)` of a video.

    Returns one row per frame in the range, with `x`, `y` in source-video
    pixels. `visible == 0` means no detection; `x`, `y` are 0 there and must
    not be read. `confidence` is the peak heatmap response, and is meaningful
    even when `visible == 0` (a near-miss reads high).
    """
    info = probe_video(video_path)
    end_frame = info.frame_count if end_frame is None else end_frame
    if end_frame <= start_frame:
        raise ValueError(f"empty frame range [{start_frame}, {end_frame})")

    model = model or load_tracknet(cfg)
    eval_mode = cfg.get("shuttle.eval_mode", "weight")
    batch_size = int(cfg.get("shuttle.batch_size", 8))
    threshold = float(cfg.get("shuttle.heatmap_threshold", 0.5))
    seq_len = model.seq_len

    # Each chunk's outermost seq_len - 1 frames are discarded (see below), so a
    # chunk has to be comfortably larger than that to make progress.
    min_chunk = 4 * seq_len
    if chunk_frames < min_chunk:
        raise ValueError(f"chunk_frames must be at least {min_chunk}, got {chunk_frames}")

    with _tracknet_importable(cfg.paths.tracknet_repo):
        from dataset import Shuttlecock_Trajectory_Dataset
        from utils.general import HEIGHT, WIDTH

        img_scaler = (info.width / WIDTH, info.height / HEIGHT)
        median = (
            _background_median(video_path, start_frame, end_frame)
            if model.bg_mode
            else None
        )

        results: dict[int, tuple[float, float, float, int]] = {}
        # A frame at the edge of a chunk sees only part of its temporal window,
        # so each chunk keeps only its interior and consecutive chunks overlap
        # by twice the margin. The first and last chunk keep their outer edge,
        # since there is no more video to give those frames context.
        margin = seq_len - 1
        stride = chunk_frames - 2 * margin
        started = time.perf_counter()
        with tqdm(total=end_frame - start_frame, unit="frame", desc="tracknet") as progress:
            for chunk_start in range(start_frame, end_frame, stride):
                chunk_end = min(chunk_start + chunk_frames, end_frame)
                if chunk_end - chunk_start < seq_len:
                    break
                frames, first = read_frames(video_path, chunk_start, chunk_end)
                chunk_results = _run_chunk(
                    model=model,
                    frames_bgr=frames,
                    chunk_start=first,
                    median=median,
                    batch_size=batch_size,
                    eval_mode=eval_mode,
                    threshold=threshold,
                    img_scaler=img_scaler,
                    dataset_cls=Shuttlecock_Trajectory_Dataset,
                    progress=progress,
                )
                keep_lo = chunk_start if chunk_start == start_frame else chunk_start + margin
                keep_hi = chunk_end if chunk_end == end_frame else chunk_end - margin
                results.update(
                    {f: v for f, v in chunk_results.items() if keep_lo <= f < keep_hi}
                )
                del frames
                if chunk_end == end_frame:
                    break
        elapsed = time.perf_counter() - started

    n_frames = end_frame - start_frame
    print(
        f"tracknet: {n_frames} frames in {elapsed:.1f}s "
        f"({n_frames / elapsed:.1f} fps, device={model.device.type}, mode={eval_mode})"
    )

    rows = [
        (frame, *results.get(frame, (0.0, 0.0, 0.0, 0)))
        for frame in range(start_frame, end_frame)
    ]
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "confidence", "visible"])
    df = df[SHUTTLE_COLUMNS]
    _assert_trajectory_sane(df, info)
    return df


def _assert_trajectory_sane(df: pd.DataFrame, info) -> None:
    """Invariants that catch frame-index and coordinate-system mistakes."""
    assert list(df.columns) == SHUTTLE_COLUMNS, df.columns
    assert df["frame"].is_monotonic_increasing, "frame indices must increase"
    assert df["frame"].is_unique, "duplicate frame indices"
    seen = df[df["visible"] == 1]
    if not seen.empty:
        assert seen["x"].between(0, info.width).all(), "x outside frame width"
        assert seen["y"].between(0, info.height).all(), "y outside frame height"
    assert df["confidence"].between(0.0, 1.0).all(), "confidence outside [0, 1]"


def track_shuttle_cached(
    cfg: Config,
    match_id: str,
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """`track_shuttle` with the caching rule: never redo work unless forced.

    The cache key is the match id alone, so re-running with a different frame
    range needs `force=True`; the range that produced the file is recorded in a
    sidecar so the mismatch is at least visible.
    """
    out_dir = cache_dir(cfg, match_id)
    out_file = out_dir / "shuttle.csv"
    if out_file.exists() and not force:
        print(f"shuttle: reusing {out_file} (pass --force to recompute)")
        return pd.read_csv(out_file)

    df = track_shuttle(video_path, cfg, start_frame, end_frame)
    df.to_csv(out_file, index=False)
    (out_dir / "shuttle.meta.json").write_text(
        pd.Series(
            {
                "video": str(Path(video_path).resolve()),
                "start_frame": start_frame,
                "end_frame": end_frame if end_frame is not None else "",
                "eval_mode": cfg.get("shuttle.eval_mode", "weight"),
            }
        ).to_json(),
        encoding="utf-8",
    )
    print(f"shuttle: wrote {out_file} ({len(df)} frames, {int(df.visible.sum())} detected)")
    return df
