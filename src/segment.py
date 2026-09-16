"""Rally / camera-cut segmentation.

Broadcast footage cuts between the play camera, replays, close-ups, crowd
shots and graphics. Everything downstream must only see play. This module
answers, for every frame in a range, "which camera segment is this, is it the
play view, and which rally (if any) is in progress".

Three stages, each cached separately:

1. `frame_signatures`: one cheap colour signature per frame, from a thumbnail.
   The only stage that decodes video, so it dominates the cost (~decode
   speed, a few hundred fps at 1080p).
2. `detect_cuts` / `classify_segments`: hard cuts from the frame-to-frame
   histogram distance; play view from court-colour coverage and white-line
   coverage inside a central region of interest. Heuristic thresholds live in
   `configs/default.yaml` under `segment:` and are expected to need tuning
   against real footage — `bda segment --sheet` renders a contact sheet with
   the per-segment numbers for exactly that.
3. `split_rallies`: within play segments, a rally is a run of frames where the
   shuttle trajectory (phase 1) is consistently visible. Between points the
   shuttle is in a hand or out of frame and TrackNet goes quiet.

Output `segments.csv` has one row per contiguous span and covers the range
exactly once: `segment_id, start_frame, end_frame, is_play, rally_id`.
Non-play camera segments are one row each with `rally_id = -1`. Play
segments are split into rally rows (`rally_id >= 0`, numbered across the
whole range) and the dead time between them (`rally_id = -1`).
`end_frame` is exclusive, as everywhere in this repo.

Known limitation: a replay shot from the play camera looks like play to the
heuristic. Replays are usually slow motion, so phase 5's shuttle-speed
invariant is the place that will catch them; if that is not enough, this is
where a classifier replaces the heuristic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from config import Config, cache_dir
from video import iter_frames, probe_video, sample_frames

SEGMENT_COLUMNS = ["segment_id", "start_frame", "end_frame", "is_play", "rally_id"]
SIGNATURE_COLUMNS = ["frame", "hist_diff", "court_frac", "line_frac"]


# --- Stage 1: per-frame signatures --------------------------------------------


def _hsv_hist(hsv: np.ndarray, bins: Sequence[int]) -> np.ndarray:
    """Normalised hue x saturation histogram, flattened. L1 norm is 1."""
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256]).ravel()
    total = hist.sum()
    return hist / total if total else hist


def _roi(hsv: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    h, w = hsv.shape[:2]
    x0, y0, x1, y1 = roi
    return hsv[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def _court_frac(hsv_roi: np.ndarray, hue_ranges: Sequence[Sequence[int]]) -> float:
    """Fraction of ROI pixels whose hue is in one of the court-colour ranges."""
    hue, sat, val = hsv_roi[..., 0], hsv_roi[..., 1], hsv_roi[..., 2]
    coloured = (sat > 40) & (val > 40)
    in_range = np.zeros(hue.shape, dtype=bool)
    for lo, hi in hue_ranges:
        in_range |= (hue >= lo) & (hue <= hi)
    return float((coloured & in_range).mean())


def _line_frac(hsv_roi: np.ndarray) -> float:
    """Fraction of ROI pixels that are white-ish: bright and unsaturated."""
    sat, val = hsv_roi[..., 1], hsv_roi[..., 2]
    return float(((sat < 60) & (val > 180)).mean())


def _thumb_hsv(frame_bgr: np.ndarray, width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if width >= w:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    thumb = cv2.resize(frame_bgr, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)


def frame_signature(
    frame_bgr: np.ndarray, cfg_seg: Config
) -> tuple[np.ndarray, float, float]:
    """`(histogram, court_frac, line_frac)` for one decoded frame.

    Two thumbnails: a tiny one for the histogram (composition only), and a
    wider one for the court features, because court lines are a few pixels
    wide at 1080p and average away into the mat below ~600 px.
    """
    hsv_small = _thumb_hsv(frame_bgr, int(cfg_seg.get("thumb_width", 160)))
    hsv_wide = _thumb_hsv(frame_bgr, int(cfg_seg.get("feature_width", 640)))
    roi = _roi(hsv_wide, cfg_seg.get("court_roi", [0.1, 0.25, 0.9, 0.95]))
    return (
        _hsv_hist(hsv_small, cfg_seg.get("hist_bins", [16, 8])),
        _court_frac(roi, cfg_seg.get("court_hue_ranges", [[35, 85], [90, 130]])),
        _line_frac(roi),
    )


def frame_signatures(
    video_path: str | Path,
    cfg: Config,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """One row per frame: `frame, hist_diff, court_frac, line_frac`.

    `hist_diff` is half the L1 distance between this frame's histogram and the
    previous one's, in [0, 1]; it is 0 on the first frame of the range.
    """
    info = probe_video(video_path)
    end_frame = info.frame_count if end_frame is None else end_frame
    if end_frame <= start_frame:
        raise ValueError(f"empty frame range [{start_frame}, {end_frame})")
    cfg_seg = cfg.get("segment", Config({}))

    rows = []
    prev_hist = None
    started = time.perf_counter()
    for idx, frame in iter_frames(video_path, start_frame, end_frame):
        hist, court, line = frame_signature(frame, cfg_seg)
        diff = 0.0 if prev_hist is None else 0.5 * float(np.abs(hist - prev_hist).sum())
        rows.append((idx, diff, court, line))
        prev_hist = hist
    elapsed = time.perf_counter() - started
    print(f"signatures: {len(rows)} frames in {elapsed:.1f}s ({len(rows) / elapsed:.0f} fps)")

    df = pd.DataFrame(rows, columns=SIGNATURE_COLUMNS)
    if df.empty:
        raise RuntimeError(f"no frames decoded from {video_path} at [{start_frame}, {end_frame})")
    if len(df) != end_frame - start_frame:
        # Container frame counts overestimate by a few frames on some files.
        # The rows tile what was actually decoded; downstream reads the range
        # from them, not from the probe.
        print(
            f"signatures: decoded {len(df)} frames, container promised {end_frame - start_frame}; "
            f"range ends at {int(df['frame'].iloc[-1]) + 1}"
        )
    return df


# --- Stage 2: cuts and play-view classification -------------------------------


def detect_cuts(sig: pd.DataFrame, fps: float, cfg_seg: Config) -> list[tuple[int, int]]:
    """Camera segments as `(start_frame, end_frame)` spans covering the range.

    A cut is a frame whose histogram distance to the previous frame exceeds
    `cut_threshold`. Segments shorter than `min_segment_s` (a flash, a
    two-frame transition) are merged into the segment before them.
    """
    threshold = float(cfg_seg.get("cut_threshold", 0.35))
    min_len = max(1, int(round(float(cfg_seg.get("min_segment_s", 0.5)) * fps)))

    frames = sig["frame"].to_numpy()
    first, last = int(frames[0]), int(frames[-1]) + 1
    cut_frames = frames[sig["hist_diff"].to_numpy() > threshold]
    # The first frame of the range can never be a cut: nothing precedes it.
    cut_frames = [int(f) for f in cut_frames if f != first]

    bounds = [first, *cut_frames, last]
    spans = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]

    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and (span[1] - span[0]) < min_len:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    # A short first segment has no predecessor; fold it into the next one.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_len:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return merged


def classify_segments(
    sig: pd.DataFrame, spans: Sequence[tuple[int, int]], cfg_seg: Config
) -> pd.DataFrame:
    """Per camera segment: median court/line coverage and the play-view verdict."""
    min_court = float(cfg_seg.get("min_court_frac", 0.25))
    min_line = float(cfg_seg.get("min_line_frac", 0.005))
    by_frame = sig.set_index("frame")
    rows = []
    for seg_id, (a, b) in enumerate(spans):
        part = by_frame.loc[a : b - 1]
        court = float(part["court_frac"].median())
        line = float(part["line_frac"].median())
        rows.append((seg_id, a, b, court, line, int(court >= min_court and line >= min_line)))
    return pd.DataFrame(rows, columns=["segment_id", "start_frame", "end_frame", "court_frac", "line_frac", "is_play"])


def merge_same_verdict(camera: pd.DataFrame) -> pd.DataFrame:
    """Merge runs of consecutive camera segments with the same `is_play` verdict.

    A one-frame flash or a two-frame transition splits the play camera into
    two segments; left as they are, a rally crossing that boundary would be
    split too. Two consecutive play segments are, for everything downstream,
    one span of play. `segment_id` is renumbered; the pre-merge rows stay in
    `camera_segments.csv` for the contact sheet.
    """
    rows = []
    for seg in camera.itertuples():
        if rows and rows[-1][3] == int(seg.is_play):
            prev = rows[-1]
            rows[-1] = (prev[0], prev[1], int(seg.end_frame), prev[3])
        else:
            rows.append((len(rows), int(seg.start_frame), int(seg.end_frame), int(seg.is_play)))
    return pd.DataFrame(rows, columns=["segment_id", "start_frame", "end_frame", "is_play"])


# --- Stage 3: rallies from the shuttle trajectory ------------------------------


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open `[start, end)` index runs where `mask` is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def rally_spans(
    visible: np.ndarray, first_frame: int, fps: float, cfg_seg: Config
) -> list[tuple[int, int]]:
    """Rally spans within one play segment from its per-frame shuttle visibility.

    Visibility is box-smoothed over `rally_smooth_s`; runs above 0.5 shorter
    than `rally_min_s` are dropped and gaps shorter than `rally_max_gap_s`
    are bridged (the shuttle vanishes for a few frames at the top of a clear).
    """
    smooth = max(1, int(round(float(cfg_seg.get("rally_smooth_s", 1.0)) * fps)))
    min_len = max(1, int(round(float(cfg_seg.get("rally_min_s", 1.5)) * fps)))
    max_gap = max(0, int(round(float(cfg_seg.get("rally_max_gap_s", 2.0)) * fps)))

    v = visible.astype(np.float64)
    kernel = np.ones(smooth) / smooth
    active = np.convolve(v, kernel, mode="same") > 0.5

    runs = _runs(active)
    bridged: list[tuple[int, int]] = []
    for run in runs:
        if bridged and run[0] - bridged[-1][1] <= max_gap:
            bridged[-1] = (bridged[-1][0], run[1])
        else:
            bridged.append(run)
    return [(first_frame + a, first_frame + b) for a, b in bridged if b - a >= min_len]


def split_rallies(
    segments: pd.DataFrame, shuttle: pd.DataFrame | None, fps: float, cfg_seg: Config
) -> pd.DataFrame:
    """Expand play segments into rally / dead-time rows. See the module docstring."""
    rows = []
    rally_id = 0
    by_frame = shuttle.set_index("frame")["visible"] if shuttle is not None else None
    for seg in segments.itertuples():
        a, b = int(seg.start_frame), int(seg.end_frame)
        if not seg.is_play or by_frame is None:
            rows.append((seg.segment_id, a, b, int(seg.is_play), -1))
            continue
        visible = by_frame.reindex(range(a, b), fill_value=0).to_numpy()
        cursor = a
        for ra, rb in rally_spans(visible, a, fps, cfg_seg):
            if ra > cursor:
                rows.append((seg.segment_id, cursor, ra, 1, -1))
            rows.append((seg.segment_id, ra, rb, 1, rally_id))
            rally_id += 1
            cursor = rb
        if cursor < b:
            rows.append((seg.segment_id, cursor, b, 1, -1))
    df = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
    _assert_segments_sane(df)
    return df


def _assert_segments_sane(df: pd.DataFrame) -> None:
    """Rows tile the range exactly once; rallies only inside play."""
    assert list(df.columns) == SEGMENT_COLUMNS, df.columns
    assert (df["end_frame"] > df["start_frame"]).all(), "empty span"
    assert (df["start_frame"].to_numpy()[1:] == df["end_frame"].to_numpy()[:-1]).all(), "spans must tile the range"
    assert df["segment_id"].is_monotonic_increasing, "segments out of order"
    rallies = df[df["rally_id"] >= 0]
    assert (rallies["is_play"] == 1).all(), "rally outside a play segment"
    assert rallies["rally_id"].tolist() == list(range(len(rallies))), "rally ids must be 0..n-1 in order"


# --- Driver -------------------------------------------------------------------


def segment_video(
    cfg: Config,
    match_id: str,
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Run all three stages with the caching rule; write `segments.csv`.

    Uses the cached shuttle trajectory for rally boundaries when present;
    otherwise every play segment is one row with `rally_id = -1` and a
    warning is printed.
    """
    out_dir = cache_dir(cfg, match_id)
    out_file = out_dir / "segments.csv"
    if out_file.exists() and not force:
        print(f"segments: reusing {out_file} (pass --force to recompute)")
        return pd.read_csv(out_file)

    info = probe_video(video_path)
    end_frame = info.frame_count if end_frame is None else end_frame
    cfg_seg = cfg.get("segment", Config({}))

    sig_file = out_dir / "frame_signatures.parquet"
    if sig_file.exists() and not force:
        sig = pd.read_parquet(sig_file)
        covered = sig["frame"].min() <= start_frame and sig["frame"].max() >= end_frame - 1
        if not covered:
            raise SystemExit(
                f"{sig_file} covers frames {sig['frame'].min()}-{sig['frame'].max()}, "
                f"not [{start_frame}, {end_frame}); pass --force"
            )
        sig = sig[(sig["frame"] >= start_frame) & (sig["frame"] < end_frame)]
        print(f"signatures: reusing {sig_file}")
    else:
        sig = frame_signatures(video_path, cfg, start_frame, end_frame)
        sig.to_parquet(sig_file, index=False)

    spans = detect_cuts(sig, info.fps, cfg_seg)
    camera = classify_segments(sig, spans, cfg_seg)
    camera.to_csv(out_dir / "camera_segments.csv", index=False)

    shuttle_file = out_dir / "shuttle.csv"
    shuttle = None
    if shuttle_file.exists():
        shuttle = pd.read_csv(shuttle_file)
    else:
        print(f"segments: no {shuttle_file}; rally boundaries skipped (run `bda track` first)")

    merged = merge_same_verdict(camera)
    df = split_rallies(merged, shuttle, info.fps, cfg_seg)
    df.to_csv(out_file, index=False)
    n_play = int(merged["is_play"].sum())
    n_rally = int((df["rally_id"] >= 0).sum())
    print(
        f"segments: wrote {out_file} — {len(camera)} camera segments merged to "
        f"{len(merged)} ({n_play} play), {n_rally} rallies"
    )
    return df


# --- Tuning aids ----------------------------------------------------------------


def write_contact_sheet(
    video_path: str | Path,
    camera: pd.DataFrame,
    out_path: str | Path,
    fps: float,
    thumb_width: int = 320,
    columns: int = 5,
) -> Path:
    """One thumbnail per camera segment, annotated with the numbers the heuristic used.

    The frame shown is the segment's midpoint. Green border = classified play,
    red = not. This is how the `segment:` thresholds get tuned.
    """
    mids = [int((a + b) // 2) for a, b in zip(camera["start_frame"], camera["end_frame"])]
    frames = sample_frames(video_path, mids)
    if not frames:
        raise RuntimeError("no segments to draw")
    h0, w0 = frames[0].shape[:2]
    tw, th = thumb_width, round(h0 * thumb_width / w0)
    text_h = 52
    rows = (len(frames) + columns - 1) // columns
    sheet = np.full((rows * (th + text_h), columns * tw, 3), 20, dtype=np.uint8)

    for i, (frame, seg) in enumerate(zip(frames, camera.itertuples())):
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        colour = (0, 200, 0) if seg.is_play else (0, 0, 220)
        cv2.rectangle(thumb, (0, 0), (tw - 1, th - 1), colour, 3)
        r, c = divmod(i, columns)
        y, x = r * (th + text_h), c * tw
        sheet[y : y + th, x : x + tw] = thumb
        secs = (seg.end_frame - seg.start_frame) / fps
        lines = [
            f"#{seg.segment_id} f{seg.start_frame}-{seg.end_frame} {secs:.1f}s",
            f"court {seg.court_frac:.2f} line {seg.line_frac:.3f}",
        ]
        for j, text in enumerate(lines):
            cv2.putText(sheet, text, (x + 4, y + th + 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def play_precision_recall(
    segments: pd.DataFrame, truth_spans: Sequence[tuple[int, int]]
) -> tuple[float, float]:
    """Frame-level precision and recall of `is_play` against hand-marked play spans.

    `truth_spans` are `(start_frame, end_frame)` half-open ranges covering the
    play view; everything else in the segmented range counts as not-play.
    """
    lo = int(segments["start_frame"].min())
    hi = int(segments["end_frame"].max())
    pred = np.zeros(hi - lo, dtype=bool)
    for seg in segments.itertuples():
        if seg.is_play:
            pred[seg.start_frame - lo : seg.end_frame - lo] = True
    truth = np.zeros(hi - lo, dtype=bool)
    for a, b in truth_spans:
        truth[max(a, lo) - lo : max(min(b, hi), lo) - lo] = True
    tp = int((pred & truth).sum())
    precision = tp / pred.sum() if pred.any() else 0.0
    recall = tp / truth.sum() if truth.any() else 0.0
    return float(precision), float(recall)
