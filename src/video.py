"""Video decode, frame iteration, clip extraction, overlay rendering.

Frame indices are absolute indices into the source video and are the only
identifier used to join any two stages together. Everything in this module
counts frames itself rather than trusting a decoder position, because an
off-by-one here silently poisons every downstream stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import pandas as pd

# Above this many frames it is worth asking the decoder to seek rather than
# decoding and discarding. Below it, sequential decode is cheap and exact.
SEEK_THRESHOLD = 240


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        info = VideoInfo(
            path=path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            # Container metadata; treated as an estimate, never as ground truth
            # for the last frame index.
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()
    if info.width == 0 or info.height == 0:
        raise RuntimeError(f"video reports zero dimensions: {path}")
    return info


def _seek(cap: cv2.VideoCapture, start_frame: int) -> None:
    """Position the capture at `start_frame`, exactly.

    Container seeking lands on a keyframe and decodes forward, which is usually
    frame-accurate but not guaranteed. Verify, and fall back to a sequential
    decode when the decoder disagrees.
    """
    if start_frame <= 0:
        return
    if start_frame >= SEEK_THRESHOLD:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == start_frame:
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(start_frame):
        if not cap.grab():
            raise RuntimeError(f"video ended before frame {start_frame}")


def iter_frames(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(absolute_frame_index, bgr_frame)` over the half-open range.

    `end_frame` is exclusive; None means decode to the end of the video.
    """
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    if end_frame is not None and end_frame <= start_frame:
        raise ValueError(f"empty frame range [{start_frame}, {end_frame})")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    try:
        _seek(cap, start_frame)
        idx = start_frame
        while end_frame is None or idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


def sample_frames(path: str | Path, indices: Sequence[int]) -> list[np.ndarray]:
    """Decode just the given absolute frame indices, in ascending order.

    One capture, moved forward between samples: a short hop is grabbed
    through, a long one is seeked (and verified, as in `_seek`). Cheap enough
    to pull ~100 frames out of a full match without a second decode pass.
    """
    indices = list(indices)
    if not indices:
        return []
    if any(b <= a for a, b in zip(indices, indices[1:])) or indices[0] < 0:
        raise ValueError("indices must be strictly increasing and non-negative")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        pos = 0  # index of the next frame cap.read() would return
        for target in indices:
            hop = target - pos
            if hop >= SEEK_THRESHOLD:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == target:
                    pos = target
                    hop = 0
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            for _ in range(hop):
                if not cap.grab():
                    raise RuntimeError(f"video ended before frame {target}")
                pos += 1
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {target}")
            pos += 1
            frames.append(frame)
    finally:
        cap.release()
    return frames


def read_frames(
    path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[np.ndarray, int]:
    """Decode a frame range into one BGR array `(N, H, W, 3)`.

    Returns the array and the index of its first frame. Memory scales with the
    range: 1080p costs ~6 MB per frame, so ~370 MB for a 60 s clip at 60 fps.
    Callers processing a whole match must chunk.
    """
    frames = [frame for _, frame in iter_frames(path, start_frame, end_frame)]
    if not frames:
        raise RuntimeError(f"no frames decoded from {path} at [{start_frame}, {end_frame})")
    return np.stack(frames), start_frame


def repeated_frames(path: str | Path, spans, threshold: float = 0.05) -> np.ndarray:
    """Frame indices, within `spans` (`[(start, end), ...)`), that repeat the frame before.

    The 2025-26 broadcasts are 25 fps content at 30 fps: every 6th frame is
    an exact copy (mean absolute difference 0.0 against 0.2-0.8 otherwise).
    Counting a copy as 1/30 s of motion puts a fast shuttle ~1 m out, so
    time-sensitive stages measure time in unique frames (`FrameClock`).
    """
    out = []
    for a, b in spans:
        prev = None
        for idx, frame in iter_frames(path, int(a), int(b)):
            g = cv2.cvtColor(cv2.resize(frame, (192, 108), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.int16)
            if prev is not None and np.abs(g - prev).mean() < threshold:
                out.append(idx)
            prev = g
    return np.array(out, dtype=np.int64)


class FrameClock:
    """Seconds between frames of one span, counting a repeated frame as no time.

    `source_fps` is the content's own rate: 25 for 30 fps video that repeats
    one frame in six, else the container's.
    """

    def __init__(self, fps: float, repeats: np.ndarray, n_frames: int):
        self.repeats = np.sort(np.asarray(repeats, dtype=np.int64))
        frac = len(self.repeats) / max(1, n_frames)
        self.source_fps = fps * (1 - frac) if frac > 0.05 else fps
        for common in (24.0, 25.0, 30.0, 50.0, 60.0):
            if abs(self.source_fps - common) < 0.5:
                self.source_fps = common
        self.uses_repeats = frac > 0.05

    def seconds(self, f0: float, f: np.ndarray) -> np.ndarray:
        """Time of frames `f` after frame `f0` (same span)."""
        f = np.asarray(f, dtype=np.float64)
        if not self.uses_repeats:
            return (f - f0) / self.source_fps
        rep = lambda x: np.searchsorted(self.repeats, np.floor(x), side="right")
        return ((f - f0) - (rep(f) - rep(f0))) / self.source_fps


def extract_clip(
    path: str | Path,
    out_path: str | Path,
    start_frame: int,
    end_frame: int,
) -> Path:
    """Write frames `[start_frame, end_frame)` to a new video file."""
    info = probe_video(path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")
    try:
        for _, frame in iter_frames(path, start_frame, end_frame):
            writer.write(frame)
    finally:
        writer.release()
    return out_path


def write_overlay_video(
    path: str | Path,
    traj: pd.DataFrame,
    out_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    traj_len: int = 8,
    radius: int = 3,
    color: Sequence[int] = (0, 0, 255),
) -> Path:
    """Draw the shuttle trajectory onto the video for visual checking.

    `traj` is the shuttle DataFrame (`frame, x, y, visible, confidence`) in
    source-video pixel coordinates. The current detection is drawn filled; the
    previous `traj_len - 1` detections trail behind it, fading out.
    """
    required = {"frame", "x", "y", "visible"}
    missing = required - set(traj.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")

    info = probe_video(path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_frame = {
        int(row.frame): (float(row.x), float(row.y))
        for row in traj.itertuples()
        if int(row.visible) == 1
    }

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (info.width, info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")

    color = tuple(int(c) for c in color)
    try:
        for idx, frame in iter_frames(path, start_frame, end_frame):
            for age in range(traj_len - 1, -1, -1):
                point = by_frame.get(idx - age)
                if point is None:
                    continue
                # Oldest point in the trail is dimmest.
                fade = 1.0 - age / max(traj_len, 1)
                shade = tuple(int(c * fade) for c in color)
                thickness = -1 if age == 0 else 1
                cv2.circle(frame, (int(point[0]), int(point[1])), radius, shade, thickness)
            cv2.putText(
                frame,
                f"frame {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
    return out_path
