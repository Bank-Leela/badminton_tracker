"""Play-view and rally segmentation.

Broadcast footage cuts between the play camera, replays, close-ups, crowd
shots and graphics. Everything downstream must only see play. This module
answers, for every frame in a range, "is this the play view, and which rally
(if any) is in progress".

"Play view" means the broadcast's main camera: the fixed high shot from behind
one baseline. Live play that the director shows from another camera is
deliberately *not* play — homography and court positions downstream are built
for the one view, and a replay must never be counted as a new rally. Losing
the occasional side-camera rally costs data; letting replays in corrupts it.

Three stages:

1. `frame_line_masks`: one small white-pixel mask per frame. The only stage
   that decodes video, so it dominates the cost (~200 fps at 1080p). Cached.
2. `play_view_scores` / `play_spans`: the main camera does not move, so its
   court lines land on the same pixels in every play-view frame. The line
   template is learned from the range itself — pixels white in a large share
   of frames — and each frame is scored by how well its white pixels match it.
   No colours, no cut detection, nothing to tune per venue. On real broadcasts
   the score is cleanly bimodal (play view >= 0.85, everything else < 0.4,
   crossfades and wipes in between), so the threshold sits in an empty gap.
3. `split_rallies`: within play spans, a rally is a run of frames where the
   shuttle trajectory (phase 1) is consistently visible. Between points the
   shuttle is in a hand or out of frame and TrackNet goes quiet.

Output `segments.csv` has one row per contiguous span and covers the range
exactly once: `segment_id, start_frame, end_frame, is_play, rally_id`.
Non-play spans are one row each with `rally_id = -1`. Play spans are split
into rally rows (`rally_id >= 0`, numbered across the whole range) and the
dead time between them (`rally_id = -1`). `end_frame` is exclusive, as
everywhere in this repo.

Assumptions, both true of the BWF broadcasts this was built on:
- The main camera is static. A camera that pans or zooms would score low and
  its frames would be dropped as not-play, never mislabelled as play. The run
  summary reports how many frames scored near the threshold; a large share
  means this assumption is breaking.
- The play view is the largest group of frames sharing a fixed layout of
  white pixels (`_dominant_layout`), and at least 3% of the range. True of
  match footage; a range that is mostly intro or interval fails loudly
  instead of guessing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from config import Config, cache_dir
from video import iter_frames, probe_video, sample_frames

SEGMENT_COLUMNS = ["segment_id", "start_frame", "end_frame", "is_play", "rally_id"]
VIEW_COLUMNS = ["segment_id", "start_frame", "end_frame", "is_play", "score"]
SCORE_COLUMNS = ["frame", "recall", "precision", "score"]

# Frames unpacked at once when scoring: 2048 masks at 320x180 is ~120 MB.
_CHUNK = 2048
# Pass 2 of the template: a line pixel is white in most play-view frames and
# few others; an overlay (the score graphic) is white in most frames of both.
_PLAY_MAJORITY = 0.5
_OTHER_MINORITY = 0.3
# Fewer template pixels than this share of the mask means no fixed play
# camera was found. Real broadcasts give 1-3%.
_MIN_TEMPLATE_FRAC = 0.002
# Frames scoring within this distance of `play_score` are reported: on a
# static camera they are only crossfades and wipes, well under 1%. Narrow on
# purpose: the play-view mode sits anywhere from ~0.75 (score graphic on
# screen, All England 2019) to ~0.95, the rest below ~0.4.
_NEAR_THRESHOLD = 0.1
# Part of the segments.csv cache key: bump when stages 2-3 change what they
# compute, so cached results are redone (from the cached masks, in seconds).
# 3: pass-1 template from the dominant shared layout, not pixel frequency.
# 4: play threshold per match (valley of the score histogram); play spans
#    shorter than play_min_s dropped.
# 5: only the main cluster of play kept (another match in the stream).
# 6: rallies bridge short cuts away from the main camera (one id, several rows).
_METHOD = 6


# --- Stage 1: per-frame white-pixel masks --------------------------------------


def _mask_params(cfg_seg: Config) -> dict[str, int]:
    return {
        "mask_width": int(cfg_seg.get("mask_width", 320)),
        "white_max_sat": int(cfg_seg.get("white_max_sat", 60)),
        "white_min_val": int(cfg_seg.get("white_min_val", 170)),
    }


def white_mask(frame_bgr: np.ndarray, params: dict[str, int]) -> np.ndarray:
    """Bright, unsaturated pixels of a downscaled frame: court lines, mostly."""
    h, w = frame_bgr.shape[:2]
    width = params["mask_width"]
    small = cv2.resize(frame_bgr, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < params["white_max_sat"]) & (hsv[..., 2] > params["white_min_val"])


def frame_line_masks(
    video_path: str | Path,
    cfg: Config,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """`(frames, packed_masks, width)` over the half-open range.

    Masks are bit-packed along the last axis (`np.packbits`), so ten minutes
    at 320x180 is ~130 MB in memory; `width` unpacks them.
    """
    info = probe_video(video_path)
    end_frame = info.frame_count if end_frame is None else end_frame
    if end_frame <= start_frame:
        raise ValueError(f"empty frame range [{start_frame}, {end_frame})")
    params = _mask_params(cfg.get("segment", Config({})))

    frames, packed = [], []
    started = time.perf_counter()
    for idx, frame in iter_frames(video_path, start_frame, end_frame):
        frames.append(idx)
        packed.append(np.packbits(white_mask(frame, params), axis=-1))
    elapsed = time.perf_counter() - started
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path} at [{start_frame}, {end_frame})")
    print(f"masks: {len(frames)} frames in {elapsed:.1f}s ({len(frames) / elapsed:.0f} fps)")
    if len(frames) != end_frame - start_frame:
        # Container frame counts overestimate by a few frames on some files.
        # Downstream reads the range from the decoded frames, not the probe.
        print(
            f"masks: decoded {len(frames)} frames, container promised {end_frame - start_frame}; "
            f"range ends at {frames[-1] + 1}"
        )
    return np.asarray(frames, dtype=np.int64), np.stack(packed), params["mask_width"]


def _unpack(packed: np.ndarray, width: int) -> np.ndarray:
    return np.unpackbits(packed, axis=-1, count=width).astype(bool)


# --- Stage 2: play-view scores and spans -----------------------------------------


def _white_share(packed: np.ndarray, width: int, select: np.ndarray) -> np.ndarray:
    """Per pixel: the share of the selected frames in which it is white."""
    idx = np.flatnonzero(select)
    total = np.zeros((packed.shape[1], width), dtype=np.int64)
    for a in range(0, idx.size, _CHUNK):
        total += _unpack(packed[idx[a : a + _CHUNK]], width).sum(axis=0)
    return total / max(1, idx.size)


def _match(packed: np.ndarray, width: int, template: np.ndarray, overlay: np.ndarray):
    """Per frame: template recall, on-template precision, and their geometric mean.

    Recall: share of template pixels white in the frame (within one pixel, to
    absorb compression jitter). Precision: share of the frame's white pixels
    that lie on the template. A close-up has neither; a crowd of white shirts
    or the Hawk-Eye graphic can have recall but not precision.
    """
    near = cv2.dilate(template.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n_template = max(1, int(template.sum()))
    recall = np.empty(len(packed))
    precision = np.empty(len(packed))
    for a in range(0, len(packed), _CHUNK):
        white = _unpack(packed[a : a + _CHUNK], width) & ~overlay
        on = (white & near).sum(axis=(1, 2))
        recall[a : a + _CHUNK] = np.minimum(on / n_template, 1.0)
        precision[a : a + _CHUNK] = on / np.maximum(1, white.sum(axis=(1, 2)))
    return recall, precision, np.sqrt(recall * precision)


def _dominant_layout(
    packed: np.ndarray, width: int, n_sample: int = 2000, pool: int = 4, min_group: float = 0.03
) -> np.ndarray:
    """Pixels white in most frames of the group that shares the largest fixed layout.

    Why not just "pixels white in many frames": a persistent graphic can be
    white in more frames than faint court lines are. On All England 2019 the
    score bug is white in 54% of frames and the lines in 15-25%, so a
    frequency template was mostly score bug and matched dark close-ups best.

    Instead, on `n_sample` evenly spaced frames (masks OR-pooled `pool` x
    `pool` to absorb jitter): each frame's group is the frames whose pooled
    mask overlaps it with IoU >= 0.5; the group's layout is the cells white in
    at least half of it. Close-ups showing only the score bug form a large
    group with a tiny layout; crowd shots form no group; the play view forms a
    large group with a large layout (lines, boards, the bug). The group
    maximising size x layout, ignoring groups under `min_group` of the sample,
    gives the template at full mask resolution.
    """
    idx = np.unique(np.linspace(0, len(packed) - 1, min(n_sample, len(packed))).astype(int))
    full = _unpack(packed[idx], width)
    h, w = full.shape[1] // pool * pool, full.shape[2] // pool * pool
    pooled = full[:, :h, :w].reshape(len(idx), h // pool, pool, w // pool, pool).any(axis=(2, 4))
    pooled = pooled.reshape(len(idx), -1).astype(np.float32)

    area = pooled.sum(axis=1)
    inter = pooled @ pooled.T
    union = area[:, None] + area[None, :] - inter
    group = (inter >= 0.5 * np.maximum(union, 1)) & (area[:, None] > 0)
    size = group.sum(axis=1)
    layout = ((group.astype(np.float32) @ pooled) >= 0.5 * np.maximum(size, 1)[:, None]).sum(axis=1)
    value = np.where(size >= max(2, min_group * len(idx)), size * layout, 0)
    best = int(np.argmax(value))
    if value[best] == 0:
        raise RuntimeError("no group of frames shares a fixed layout; is there play view in this range?")
    return full[group[best]].mean(axis=0) > _PLAY_MAJORITY


def play_view_scores(
    frames: np.ndarray, packed: np.ndarray, width: int, cfg_seg: Config
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """`(scores, template, overlay)`: per-frame match to the learned line template.

    Pass 1: `_dominant_layout` finds the group of frames sharing the largest
    fixed layout of white pixels; its consensus is the pass-1 template.
    Frames scoring at least `seed_score` against it are taken as play view.
    Pass 2: overlay = pixels white in most non-seed frames as well; template =
    pixels white in most seed frames and few others. Final scores use these.
    """
    seed_score = float(cfg_seg.get("seed_score", 0.75))
    no_overlay = np.zeros((packed.shape[1], width), dtype=bool)
    _, _, first = _match(packed, width, _dominant_layout(packed, width), no_overlay)
    seed = first >= seed_score
    if not seed.any():
        raise RuntimeError(
            f"no frame in [{frames[0]}, {frames[-1] + 1}) matches a dominant line layout; "
            "is there play view in this range?"
        )

    in_play = _white_share(packed, width, seed)
    elsewhere = _white_share(packed, width, ~seed)
    overlay = elsewhere > _PLAY_MAJORITY
    template = (in_play > _PLAY_MAJORITY) & (elsewhere < _OTHER_MINORITY)
    if template.mean() < _MIN_TEMPLATE_FRAC:
        raise RuntimeError(
            f"line template has {int(template.sum())} pixels ({template.mean():.2%} of the mask); "
            "no fixed play camera found in this range"
        )

    recall, precision, score = _match(packed, width, template, overlay)
    scores = pd.DataFrame({"frame": frames, "recall": recall, "precision": precision, "score": score})
    return scores, template, overlay


def play_threshold(score: np.ndarray, cfg_seg: Config) -> float:
    """`play_score` from the config, or with `auto` the valley between the score modes.

    Scores are bimodal — play view high, everything else low — but where the
    modes sit varies by broadcast: play view at 0.65-0.85 with the rest under
    0.40 (2018-19, score graphic on screen during play), play at 0.75-0.90
    with a tail of non-play up to 0.55 (All England 2026). No fixed value sits
    mid-gap in both. `auto`: histogram in 40 bins, 3-bin smoothing, the lowest
    point between 0.3 and 0.8; when the valley is flat, the middle of the run
    of bins within 1.5x (+1) of that lowest count.
    """
    setting = cfg_seg.get("play_score", "auto")
    if setting != "auto":
        return float(setting)
    hist, edges = np.histogram(score, bins=40, range=(0.0, 1.0))
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    lo, hi = int(np.searchsorted(centres, 0.3)), int(np.searchsorted(centres, 0.8))
    window = smooth[lo:hi]
    i = int(np.argmin(window))
    low = window <= window[i] * 1.5 + 1
    a = b = i
    while a > 0 and low[a - 1]:
        a -= 1
    while b < len(window) - 1 and low[b + 1]:
        b += 1
    return float((centres[lo + a] + centres[lo + b]) / 2)


def play_spans(scores: pd.DataFrame, fps: float, cfg_seg: Config) -> pd.DataFrame:
    """Contiguous play / not-play spans tiling the range, from per-frame scores.

    A frame is play view when its score is at least `play_threshold`; the
    verdict is then a majority vote over `play_smooth_s`, which absorbs a flash
    or a few frames of a wipe, and play runs shorter than `play_min_s` are
    dropped — on All England 2026 those were 0.2-1.7 s glimpses of close-ups
    scoring just over the line, and no rally fits in one anyway. `score` in
    the output is the span's median score; the threshold used is in
    `.attrs["play_score"]`.
    """
    threshold = play_threshold(scores["score"].to_numpy(), cfg_seg)
    k = max(1, int(round(float(cfg_seg.get("play_smooth_s", 0.5)) * fps))) | 1  # odd window
    min_len = int(round(float(cfg_seg.get("play_min_s", 2.0)) * fps))

    raw = (scores["score"].to_numpy() >= threshold).astype(np.float64)
    vote = np.convolve(np.pad(raw, k // 2, mode="edge"), np.ones(k) / k, mode="valid") > 0.5
    for a, b in _runs(vote):
        if b - a < min_len:
            vote[a:b] = False
    vote, dropped = _keep_main_match(vote, fps, cfg_seg)

    frames = scores["frame"].to_numpy()
    change = np.flatnonzero(vote[1:] != vote[:-1]) + 1
    bounds = [0, *change.tolist(), len(vote)]
    rows = []
    for seg_id, (a, b) in enumerate(zip(bounds, bounds[1:])):
        median = float(np.median(scores["score"].to_numpy()[a:b]))
        rows.append((seg_id, int(frames[a]), int(frames[b - 1]) + 1, int(vote[a]), median))
    views = pd.DataFrame(rows, columns=VIEW_COLUMNS)
    views.attrs["play_score"] = threshold
    views.attrs["dropped"] = [(int(frames[a]), int(frames[b - 1]) + 1) for a, b in dropped]
    return views


def _keep_main_match(vote: np.ndarray, fps: float, cfg_seg: Config) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Keep only the cluster of play that is the match; return it and the dropped spans.

    A stream can open or close on another match on the same court — India
    Open 2023 starts with the end of the men's doubles final, shot by the same
    main camera, so it passes as play view. Within a match, play view never
    pauses longer than an interval or a challenge (at most 4.7 min across 32
    matches); between matches there are ceremonies and walk-ons. Play runs
    separated by more than `match_gap_min` form clusters; the one with the most
    play is kept.
    """
    gap = int(round(float(cfg_seg.get("match_gap_min", 6.0)) * 60 * fps))
    runs = _runs(vote)
    if len(runs) < 2:
        return vote, []
    clusters = [[runs[0]]]
    for run in runs[1:]:
        if run[0] - clusters[-1][-1][1] > gap:
            clusters.append([run])
        else:
            clusters[-1].append(run)
    if len(clusters) == 1:
        return vote, []
    main = max(clusters, key=lambda c: sum(b - a for a, b in c))
    vote = vote.copy()
    dropped = [run for c in clusters if c is not main for run in c]
    for a, b in dropped:
        vote[a:b] = False
    return vote, dropped


def write_template_image(template: np.ndarray, overlay: np.ndarray, out_path: str | Path, scale: int = 3) -> Path:
    """White = learned court lines, red = overlay pixels excluded. For eyeballing."""
    vis = np.zeros(template.shape + (3,), dtype=np.uint8)
    vis[template] = (255, 255, 255)
    vis[overlay] = (0, 0, 255)
    h, w = template.shape
    out_path = Path(out_path)
    cv2.imwrite(str(out_path), cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST))
    return out_path


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
    if by_frame is not None:
        df = _bridge_rallies(df, fps, cfg_seg)
    _assert_segments_sane(df)
    return df


def _bridge_rallies(df: pd.DataFrame, fps: float, cfg_seg: Config) -> pd.DataFrame:
    """Give one rally id to rally pieces split by a short cut away from the main camera.

    2018-19 broadcasts cut to a side camera for ~1 s in the middle of a rally
    and back; each main-camera piece became its own rally (All England 2019:
    170 rallies for 104 points). Between points the main camera is away for
    10 s or more. So rally rows less than `rally_max_gap_s` apart keep the
    same id: a rally may span several play segments, with the off-camera
    seconds between them as non-play rows (`rally_id = -1`, no tracking).
    Within one play segment this changes nothing — `rally_spans` already
    bridged those gaps.
    """
    max_gap = float(cfg_seg.get("rally_max_gap_s", 2.0)) * fps
    ids = df["rally_id"].to_numpy().copy()
    current, prev_end = -1, None
    for i in np.flatnonzero(ids >= 0):
        start = df["start_frame"].iat[i]
        if prev_end is None or start - prev_end > max_gap:
            current += 1
        ids[i] = current
        prev_end = df["end_frame"].iat[i]
    return df.assign(rally_id=ids)


def _assert_segments_sane(df: pd.DataFrame) -> None:
    """Rows tile the range exactly once; rallies only inside play."""
    assert list(df.columns) == SEGMENT_COLUMNS, df.columns
    assert (df["end_frame"] > df["start_frame"]).all(), "empty span"
    assert (df["start_frame"].to_numpy()[1:] == df["end_frame"].to_numpy()[:-1]).all(), "spans must tile the range"
    assert df["segment_id"].is_monotonic_increasing, "segments out of order"
    rallies = df[df["rally_id"] >= 0]
    assert (rallies["is_play"] == 1).all(), "rally outside a play segment"
    # One rally may span several rows (pieces either side of a short cut).
    ids = rallies["rally_id"].to_numpy()
    assert (np.diff(ids) >= 0).all() and (np.diff(ids) <= 1).all() and (len(ids) == 0 or ids[0] == 0), \
        "rally ids must run 0..n-1 in order"


# --- Driver -------------------------------------------------------------------


def _load_masks(mask_file: Path, start_frame: int, end_frame: int, params: dict[str, int]):
    """Cached masks restricted to the range, or SystemExit if they cannot serve it."""
    with np.load(mask_file) as z:
        built_with = {key: int(z[key]) for key in params}
        lo, hi = int(z["range_start"]), int(z["range_end"])
        frames, packed = z["frames"], z["packed"]
    if built_with != params:
        raise SystemExit(f"{mask_file} was built with {built_with}, config says {params}; pass --force")
    if lo > start_frame or hi < end_frame:
        raise SystemExit(f"{mask_file} covers frames [{lo}, {hi}), not [{start_frame}, {end_frame}); pass --force")
    keep = (frames >= start_frame) & (frames < end_frame)
    return frames[keep], packed[keep]


def _file_stamp(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def segment_video(
    cfg: Config,
    match_id: str,
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Run all three stages with the caching rule; write `segments.csv`.

    Only the decode is expensive, and its masks are cached in
    `line_masks.npz`. `segments.csv` is reused only while the range, the
    `segment:` config and `shuttle.csv` are unchanged (recorded in
    `segments.meta.json`); otherwise stages 2-3 rerun from the cached masks,
    which takes seconds. `force` also redoes the decode.

    Uses the cached shuttle trajectory for rally boundaries when present;
    otherwise every play span is one row with `rally_id = -1` and a warning is
    printed.
    """
    out_dir = cache_dir(cfg, match_id)
    out_file = out_dir / "segments.csv"
    meta_file = out_dir / "segments.meta.json"
    shuttle_file = out_dir / "shuttle.csv"

    info = probe_video(video_path)
    end_frame = info.frame_count if end_frame is None else end_frame
    cfg_seg = cfg.get("segment", Config({}))
    key = {
        "method": _METHOD,
        "range": [start_frame, end_frame],
        "segment": cfg_seg.to_dict(),
        "shuttle": _file_stamp(shuttle_file),
    }
    if out_file.exists() and meta_file.exists() and not force:
        if json.loads(meta_file.read_text()) == key:
            print(f"segments: reusing {out_file} (pass --force to recompute)")
            return pd.read_csv(out_file)

    params = _mask_params(cfg_seg)
    mask_file = out_dir / "line_masks.npz"
    if mask_file.exists() and not force:
        frames, packed = _load_masks(mask_file, start_frame, end_frame, params)
        print(f"masks: reusing {mask_file}")
    else:
        frames, packed, _ = frame_line_masks(video_path, cfg, start_frame, end_frame)
        np.savez_compressed(
            mask_file, frames=frames, packed=packed, range_start=start_frame, range_end=end_frame, **params
        )
    width = params["mask_width"]

    scores, template, overlay = play_view_scores(frames, packed, width, cfg_seg)
    scores.to_parquet(out_dir / "view_scores.parquet", index=False)
    write_template_image(template, overlay, out_dir / "line_template.png")
    views = play_spans(scores, info.fps, cfg_seg)
    views.to_csv(out_dir / "view_segments.csv", index=False)

    shuttle = None
    if shuttle_file.exists():
        shuttle = pd.read_csv(shuttle_file)
    else:
        print(f"segments: no {shuttle_file}; rally boundaries skipped (run `bda track` first)")

    df = split_rallies(views, shuttle, info.fps, cfg_seg)
    df.to_csv(out_file, index=False)
    meta_file.write_text(json.dumps(key))

    threshold = views.attrs["play_score"]
    near = float((scores["score"] - threshold).abs().lt(_NEAR_THRESHOLD).mean())
    n_play = int(views["is_play"].sum())
    play_frac = float((views["end_frame"] - views["start_frame"])[views["is_play"] == 1].sum()) / len(scores)
    n_rally = int(df.loc[df["rally_id"] >= 0, "rally_id"].nunique())
    print(
        f"segments: wrote {out_file} — {n_play} play spans ({play_frac:.0%} of frames), {n_rally} rallies; "
        f"{near:.1%} of frames scored within {_NEAR_THRESHOLD} of play_score {threshold:.2f} "
        f"(template {int(template.sum())} px)"
    )
    if near > 0.02:
        print("segments: WARNING many frames near the threshold — does the play camera pan or zoom?")
    for a, b in views.attrs["dropped"]:
        print(f"segments: dropped play view at f{a}-{b} ({(b - a) / info.fps:.0f} s): separate from the "
              f"main match by more than match_gap_min — another match in the stream?")
    return df


# --- Tuning aids ----------------------------------------------------------------


def write_contact_sheet(
    video_path: str | Path,
    views: pd.DataFrame,
    out_path: str | Path,
    fps: float,
    thumb_width: int = 320,
    columns: int = 5,
) -> Path:
    """One thumbnail per play / not-play span, annotated with its median score.

    The frame shown is the span's midpoint. Green border = play view, red =
    not. A not-play span may hold several shots (close-up, replay, crowd);
    only one of them is shown.
    """
    mids = [int((a + b) // 2) for a, b in zip(views["start_frame"], views["end_frame"])]
    frames = sample_frames(video_path, mids)
    if not frames:
        raise RuntimeError("no segments to draw")
    h0, w0 = frames[0].shape[:2]
    tw, th = thumb_width, round(h0 * thumb_width / w0)
    text_h = 52
    rows = (len(frames) + columns - 1) // columns
    sheet = np.full((rows * (th + text_h), columns * tw, 3), 20, dtype=np.uint8)

    for i, (frame, seg) in enumerate(zip(frames, views.itertuples())):
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        colour = (0, 200, 0) if seg.is_play else (0, 0, 220)
        cv2.rectangle(thumb, (0, 0), (tw - 1, th - 1), colour, 3)
        r, c = divmod(i, columns)
        y, x = r * (th + text_h), c * tw
        sheet[y : y + th, x : x + tw] = thumb
        secs = (seg.end_frame - seg.start_frame) / fps
        lines = [
            f"#{seg.segment_id} f{seg.start_frame}-{seg.end_frame} {secs:.1f}s",
            f"score {seg.score:.2f}",
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
