"""TrackNetV3 wrapper.

`external/TrackNetV3` is a frozen dependency: it is never edited and never
imported by anything except this module. Everything the rest of the pipeline
knows about the shuttle comes out of `track_shuttle()` as a DataFrame with
columns `frame, x, y, visible, confidence`, in source-video pixel coordinates.

Three deliberate differences from their `predict.py`:

* Confidence. Their `predict()` thresholds the heatmap and reports only a 0/1
  visibility flag. Contact detection in phase 5 needs to know how sure the
  model was, so the peak heatmap response at the detected blob is kept.
* Edge frames. Their temporal ensemble divides early frames by an occurrence
  count and later frames by the full weight sum. Here every frame is divided
  by the weight actually accumulated for it, which is the same in the interior
  and better defined at the ends.
* Input preprocessing. Their `Shuttlecock_Trajectory_Dataset` resizes every
  full-resolution frame once per sliding window it appears in (`seq_len`
  times) and concatenates float64 arrays per window, which starves the GPU:
  ~5 fps on an RTX 5070. This module resizes each frame once with the same
  PIL call, keeps the small frames on the device, and builds windows and the
  temporal ensemble there. `tests/test_shuttle.py` asserts the tensors fed to
  the network match their dataset's output.

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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import Config, cache_dir
from video import iter_frames, probe_video, sample_frames

SHUTTLE_COLUMNS = ["frame", "x", "y", "visible", "confidence"]

# Model input size. Hardcoded in TrackNetV3's utils.general; repeated here so
# the resize can run before their code is imported. Checked against theirs at
# load time.
INPUT_WIDTH, INPUT_HEIGHT = 512, 288

# Frames held on the device at once, already resized to the model input size
# (~0.44 MB each as uint8, plus one float32 heatmap each for the ensemble).
# Consecutive chunks overlap by 2 * (seq_len - 1) frames so the temporal
# ensemble has full context across the seam.
DEFAULT_CHUNK_FRAMES = 1200

# Frames sampled across the requested range to build the background median
# image. Odd, so np.median never has to average two middle values.
MEDIAN_SAMPLE_NUM = 121

# Threads resizing decoded frames. PIL releases the GIL inside resize.
RESIZE_WORKERS = min(8, os.cpu_count() or 1)


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
        from utils.general import HEIGHT, WIDTH, get_model

        assert (WIDTH, HEIGHT) == (INPUT_WIDTH, INPUT_HEIGHT), (
            f"TrackNetV3 input size changed to {WIDTH}x{HEIGHT}; update INPUT_WIDTH/INPUT_HEIGHT"
        )
        # weights_only=False: the checkpoint stores a param_dict alongside the
        # weights. The file is the project's own published checkpoint.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        seq_len = int(ckpt["param_dict"]["seq_len"])
        bg_mode = ckpt["param_dict"]["bg_mode"]
        net = get_model("TrackNet", seq_len, bg_mode)
        net.load_state_dict(ckpt["model"])

    if bg_mode not in ("", "concat"):
        # The published checkpoint is `concat`. The subtract modes need a
        # per-frame difference image that this wrapper does not build.
        raise NotImplementedError(f"bg_mode {bg_mode!r} is not supported by this wrapper")

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


def _resize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Resize one RGB uint8 frame to the model input size, as their dataset does.

    Must stay a PIL resize with default resampling: that is what the model
    was trained on, and cv2's bicubic is not the same filter.
    """
    return np.asarray(Image.fromarray(rgb).resize((INPUT_WIDTH, INPUT_HEIGHT)))


def _resize_bgr_to_chw(bgr: np.ndarray) -> np.ndarray:
    """Decoded BGR frame -> RGB, resized, channels-first uint8 `(3, H, W)`."""
    return np.ascontiguousarray(np.moveaxis(_resize_rgb(np.ascontiguousarray(bgr[..., ::-1])), -1, 0))


def _background_median(
    video_path: str | Path, start_frame: int, end_frame: int
) -> np.ndarray:
    """Median image over frames sampled evenly across the range.

    Computed once at source resolution for the whole range, so chunking cannot
    change the model's input. Returned as RGB float `(H, W, 3)`, which is what
    their dataset expects to be handed.
    """
    n = end_frame - start_frame
    step = max(1, n // MEDIAN_SAMPLE_NUM)
    # Seek to each sample rather than decoding the whole range: on a full
    # match that is ~120 seeks instead of a second pass over 100k frames.
    samples = sample_frames(video_path, range(start_frame, end_frame, step))
    if not samples:
        raise RuntimeError(f"no frames sampled for median over [{start_frame}, {end_frame})")
    # BGR -> RGB, matching the frame order handed to the dataset.
    return np.median(np.stack(samples)[..., ::-1], axis=0)


def _median_to_input(median_rgb: np.ndarray) -> np.ndarray:
    """Their `concat` preprocessing of the median: uint8, resize, channels first."""
    return np.ascontiguousarray(np.moveaxis(_resize_rgb(median_rgb.astype("uint8")), -1, 0))


def _read_frames_resized(
    video_path: str | Path, start_frame: int, end_frame: int
) -> np.ndarray:
    """Decode `[start_frame, end_frame)` straight into model-sized uint8 `(N, 3, H, W)`.

    Full-resolution frames never accumulate: each is resized as it is decoded,
    on a thread pool, so memory is ~0.44 MB per frame regardless of source size.
    """
    # Executor.map submits its whole input up front, which would hold every
    # full-resolution frame of the chunk at once; feed it in small batches.
    batch = 4 * RESIZE_WORKERS
    small: list[np.ndarray] = []
    pending: list[np.ndarray] = []
    with ThreadPoolExecutor(RESIZE_WORKERS) as pool:
        for _, frame in iter_frames(video_path, start_frame, end_frame):
            pending.append(frame)
            if len(pending) == batch:
                small.extend(pool.map(_resize_bgr_to_chw, pending))
                pending = []
        if pending:
            small.extend(pool.map(_resize_bgr_to_chw, pending))
    if not small:
        raise RuntimeError(f"no frames decoded from {video_path} at [{start_frame}, {end_frame})")
    return np.stack(small)


def _window_indices(n_frames: int, seq_len: int, eval_mode: str) -> np.ndarray:
    """Local frame indices per sliding window, `(num_windows, seq_len)`.

    Mirrors their `_gen_input_from_frame_arr`: stride 1 for the overlapping
    modes; stride `seq_len` for `nonoverlap`, with the last window padded by
    repeating the final frame so every frame is covered.
    """
    if n_frames < seq_len:
        raise ValueError(f"need at least seq_len={seq_len} frames, got {n_frames}")
    if eval_mode == "nonoverlap":
        starts = np.arange(0, n_frames, seq_len)
        return np.minimum(starts[:, None] + np.arange(seq_len)[None, :], n_frames - 1)
    starts = np.arange(0, n_frames - seq_len + 1)
    return starts[:, None] + np.arange(seq_len)[None, :]


def _build_input(
    frames: torch.Tensor, median: torch.Tensor | None, idx: torch.Tensor
) -> torch.Tensor:
    """Network input for a batch of windows, `(B, C, H, W)` float32 in [0, 1].

    `frames` is `(N, 3, H, W)` uint8 on the device; `idx` is `(B, seq_len)`.
    Channel order matches their dataset: median RGB first (in `concat` mode),
    then each window frame's RGB in sequence order.
    """
    b, seq_len = idx.shape
    x = frames[idx].reshape(b, seq_len * 3, frames.shape[2], frames.shape[3]).float() / 255.0
    if median is not None:
        x = torch.cat([median.expand(b, -1, -1, -1), x], dim=1)
    return x


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


def _ensemble_heatmaps(
    model: TrackNetModel,
    frames_small: np.ndarray,
    median_input: np.ndarray | None,
    batch_size: int,
    eval_mode: str,
    progress: tqdm | None = None,
    precision: str = "fp32",
) -> np.ndarray:
    """Weighted temporal-ensemble heatmap per frame, `(N, H, W)` float32.

    Windows are gathered and the ensemble accumulated on the device; only the
    finished per-frame heatmaps come back to the CPU, once. Each frame is
    divided by the weight actually accumulated for it.

    `precision="fp16"` runs the network under autocast on CUDA: ~1.6x faster,
    heatmaps differ from fp32 by <1e-3, which can flip a borderline pixel
    across the detection threshold. The ensemble itself stays float32.
    """
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"precision must be fp32 or fp16, got {precision!r}")
    seq_len = model.seq_len
    device = model.device
    n, _, h, w = frames_small.shape

    windows = torch.from_numpy(_window_indices(n, seq_len, eval_mode)).to(device)
    weight = torch.from_numpy(_ensemble_weight(seq_len, eval_mode)).float().to(device)

    frames = torch.from_numpy(frames_small).to(device)
    median = (
        torch.from_numpy(median_input).to(device).float() / 255.0
        if median_input is not None
        else None
    )
    acc = torch.zeros((n, h, w), dtype=torch.float32, device=device)
    acc_w = torch.zeros(n, dtype=torch.float32, device=device)

    autocast = torch.autocast(
        device.type, dtype=torch.float16, enabled=(precision == "fp16" and device.type == "cuda")
    )
    with torch.no_grad(), autocast:
        for b0 in range(0, len(windows), batch_size):
            idx = windows[b0 : b0 + batch_size]
            y_pred = model.net(_build_input(frames, median, idx)).float()  # (B, seq_len, H, W)
            flat = idx.reshape(-1)
            acc.index_add_(0, flat, (y_pred * weight[None, :, None, None]).reshape(-1, h, w))
            acc_w.index_add_(0, flat, weight.repeat(idx.shape[0]))
            if progress is not None:
                progress.update(int(idx.shape[0]) * (seq_len if eval_mode == "nonoverlap" else 1))

    assert bool((acc_w > 0).all()), "a frame received no ensemble weight"
    return (acc / acc_w[:, None, None]).cpu().numpy()


def _run_chunk(
    model: TrackNetModel,
    frames_small: np.ndarray,
    chunk_start: int,
    median_input: np.ndarray | None,
    batch_size: int,
    eval_mode: str,
    threshold: float,
    img_scaler: tuple[float, float],
    progress: tqdm | None,
    precision: str = "fp32",
) -> dict[int, tuple[float, float, float, int]]:
    """Run TrackNet over one chunk of resized frames, returning per-absolute-frame results."""
    heatmaps = _ensemble_heatmaps(
        model, frames_small, median_input, batch_size, eval_mode, progress, precision
    )
    results: dict[int, tuple[float, float, float, int]] = {}
    for local in range(len(heatmaps)):
        cx, cy, conf, vis = _localise(heatmaps[local], threshold)
        results[chunk_start + local] = (cx * img_scaler[0], cy * img_scaler[1], conf, vis)
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
    precision = str(cfg.get("shuttle.precision", "fp32"))
    seq_len = model.seq_len

    # Each chunk's outermost seq_len - 1 frames are discarded (see below), so a
    # chunk has to be comfortably larger than that to make progress.
    min_chunk = 4 * seq_len
    if chunk_frames < min_chunk:
        raise ValueError(f"chunk_frames must be at least {min_chunk}, got {chunk_frames}")

    img_scaler = (info.width / INPUT_WIDTH, info.height / INPUT_HEIGHT)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    t0 = time.perf_counter()
    median_input = (
        _median_to_input(_background_median(video_path, start_frame, end_frame))
        if model.bg_mode
        else None
    )
    timings["median"] = time.perf_counter() - t0

    results: dict[int, tuple[float, float, float, int]] = {}
    # A frame at the edge of a chunk sees only part of its temporal window,
    # so each chunk keeps only its interior and consecutive chunks overlap
    # by twice the margin. The first and last chunk keep their outer edge,
    # since there is no more video to give those frames context.
    margin = seq_len - 1
    stride = chunk_frames - 2 * margin
    timings["decode"] = timings["infer"] = 0.0
    with tqdm(total=end_frame - start_frame, unit="frame", desc="tracknet") as progress:
        for chunk_start in range(start_frame, end_frame, stride):
            chunk_end = min(chunk_start + chunk_frames, end_frame)
            if chunk_end - chunk_start < seq_len:
                break
            t0 = time.perf_counter()
            frames_small = _read_frames_resized(video_path, chunk_start, chunk_end)
            timings["decode"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            chunk_results = _run_chunk(
                model=model,
                frames_small=frames_small,
                chunk_start=chunk_start,
                median_input=median_input,
                batch_size=batch_size,
                eval_mode=eval_mode,
                threshold=threshold,
                img_scaler=img_scaler,
                progress=progress,
                precision=precision,
            )
            timings["infer"] += time.perf_counter() - t0
            keep_lo = chunk_start if chunk_start == start_frame else chunk_start + margin
            keep_hi = chunk_end if chunk_end == end_frame else chunk_end - margin
            results.update(
                {f: v for f, v in chunk_results.items() if keep_lo <= f < keep_hi}
            )
            del frames_small
            if chunk_end == end_frame:
                break
    elapsed = time.perf_counter() - started

    n_frames = end_frame - start_frame
    print(
        f"tracknet: {n_frames} frames in {elapsed:.1f}s ({n_frames / elapsed:.1f} fps; "
        f"median {timings['median']:.1f}s, decode+resize {timings['decode']:.1f}s, "
        f"infer {timings['infer']:.1f}s; device={model.device.type}, {eval_mode}, {precision})"
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
