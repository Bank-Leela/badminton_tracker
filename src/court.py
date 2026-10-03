"""Court model and the image -> court homography (phase 3).

Court coordinates are metres on the floor, origin at the centre of the court
(under the net), x across the court increasing to the right as seen from the
main camera, y along it increasing away from the camera. So the near baseline
is y = -6.70, the far one y = +6.70, the doubles sidelines x = -/+3.05.

The broadcast's main camera is static (phase 2 depends on that too), so the
court is fitted once per match on an empty-court background — the per-pixel
median of play-view frames, where players and shuttle vanish — and then
checked against every play span on its own frames. A span where the camera
has moved is refitted on its own background; one that still fails gets no
matrix and is left for `bda court-click`.

Fitting:
1. `line_mask`: court lines are thin and brighter than the mat around them.
   A brightness top-hat finds them; a saturation test does not, because
   4:2:0 chroma smears 2-3 px horizontal lines into the green.
2. `detect_lines`: Hough segments merged into infinite lines, split into
   cross lines (near horizontal in the image) and lengthwise lines.
3. `fit_court`: every pairing of two cross and two lengthwise candidates with
   two cross and two lengthwise model lines gives a homography; each is
   scored by how far *all* projected model lines land from line pixels
   (a consensus score, so a wrong pairing leaves model lines on bare mat).
4. `refine_court`: line pixels near each projected model line are fitted
   with a robust line; all cross x lengthwise intersections give a
   least-squares homography. Repeated with a narrowing search band.
5. `court_dimensions` / `check_dimensions`: length and width measured from
   the fitted image lines through the homography must be 13.40 m and 6.10 m
   within `length_tolerance_m`. A real check, raised as `CourtCheckError`,
   not an `assert` statement that -O would strip.

Model lines are placed at the nominal distances, which BWF measures to the
outer edges of 40 mm lines: a 2 cm ambiguity, a tenth of the tolerance.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from config import Config, cache_dir
from video import probe_video, sample_frames

# --- Court model ---------------------------------------------------------------

COURT_LENGTH = 13.40
DOUBLES_WIDTH = 6.10
SINGLES_WIDTH = 5.18
LONG_SERVICE_DOUBLES = 0.76  # from each baseline
SHORT_SERVICE = 1.98  # from the net

HALF_L, HALF_W, HALF_S = COURT_LENGTH / 2, DOUBLES_WIDTH / 2, SINGLES_WIDTH / 2

# Cross lines (constant y), in image order top to bottom: far to near.
CROSS_LINES = {
    "far_baseline": HALF_L,
    "far_long_service": HALF_L - LONG_SERVICE_DOUBLES,
    "far_short_service": SHORT_SERVICE,
    "near_short_service": -SHORT_SERVICE,
    "near_long_service": -(HALF_L - LONG_SERVICE_DOUBLES),
    "near_baseline": -HALF_L,
}
# Lengthwise lines (constant x), in image order left to right.
LONG_LINES = {
    "left_doubles": -HALF_W,
    "left_singles": -HALF_S,
    "centre": 0.0,
    "right_singles": HALF_S,
    "right_doubles": HALF_W,
}
# Doubles court corners, in the order `bda court-click` asks for them.
CORNERS = np.array([[-HALF_W, -HALF_L], [HALF_W, -HALF_L], [HALF_W, HALF_L], [-HALF_W, HALF_L]])
CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]


def painted_segments() -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """`(line_name, start, end)` for every painted stretch, in court metres.

    The centre line is two stretches: it runs from each short service line to
    its baseline, not across the middle of the court.
    """
    segs = [(name, (-HALF_W, y), (HALF_W, y)) for name, y in CROSS_LINES.items()]
    for name, x in LONG_LINES.items():
        if name == "centre":
            segs.append((name, (0.0, SHORT_SERVICE), (0.0, HALF_L)))
            segs.append((name, (0.0, -HALF_L), (0.0, -SHORT_SERVICE)))
        else:
            segs.append((name, (x, -HALF_L), (x, HALF_L)))
    return segs


def court_points(step: float) -> np.ndarray:
    """Points every `step` metres along all painted lines, `(N, 2)`."""
    pts = []
    for _, (x0, y0), (x1, y1) in painted_segments():
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1))
    return np.concatenate(pts)


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to `(N, 2)` points."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


class CourtCheckError(AssertionError):
    """A fitted court failed a geometric invariant. Subclasses AssertionError
    on purpose: these are the plan's invariants, but raised explicitly so
    `python -O` cannot disable them."""


# --- Background and line mask ---------------------------------------------------


def play_background(video_path: str | Path, views: pd.DataFrame, n_frames: int, margin: int = 15) -> np.ndarray:
    """Per-pixel median of `n_frames` frames spread evenly over the play spans.

    The camera is static and players move, so the median is the empty court.
    `margin` frames at each span edge are skipped (crossfades).
    """
    play = views[views["is_play"] == 1]
    pool = np.concatenate(
        [np.arange(a + margin, b - margin) for a, b in zip(play["start_frame"], play["end_frame"]) if b - a > 2 * margin]
        or [np.array([], dtype=np.int64)]
    )
    if pool.size == 0:
        raise RuntimeError("no play-view frames to build a court background from")
    n = min(n_frames, pool.size) | 1  # odd, so the median is an actual sample
    n = min(n, pool.size)
    idx = np.unique(pool[np.linspace(0, pool.size - 1, n).astype(int)])
    stack = np.stack(sample_frames(video_path, idx.tolist()))
    k = len(stack) // 2
    return np.partition(stack, k, axis=0)[k]


def line_mask(bgr: np.ndarray, cfg_court: Config) -> np.ndarray:
    """Thin structures brighter than their surroundings: court lines, mostly."""
    h = bgr.shape[0]
    k = max(3, int(round(float(cfg_court.get("tophat_px", 15)) * h / 1080))) | 1
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(luma, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return tophat > int(cfg_court.get("tophat_min", 30))


# --- Line candidates ---------------------------------------------------------------


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton: each stroke thinned to roughly its centreline.

    Hough on the raw mask puts segments along both *edges* of every 3-8 px
    stroke, so candidate lines sit off-centre; on the skeleton they sit on
    the centreline, where "mat on both sides" can be tested.
    """
    img = mask.astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros_like(img)
    while img.any():
        eroded = cv2.erode(img, cross)
        skel |= img - cv2.dilate(eroded, cross)  # opening is a subset of img
        img = eroded
    return skel.astype(bool)


def _hline(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two points, normalised so (a, b) is a unit normal."""
    line = np.cross(np.r_[p, 1.0], np.r_[q, 1.0])
    return line / np.hypot(line[0], line[1])


def detect_lines(mask: np.ndarray, cfg_court: Config) -> tuple[np.ndarray, np.ndarray]:
    """Candidate court lines as homogeneous `(a, b, c)` rows: `(cross, lengthwise)`.

    Hough runs on the skeleton of the mask. Segments are merged when their
    normals agree within 1 degree and their offsets within `merge_px`; the
    merged line is the longest segment's.
    Cross lines are within 20 degrees of horizontal; lengthwise lines are
    steeper than 25 degrees. Each set keeps the `max_candidates` best-supported
    lines, cross sorted top to bottom, lengthwise left to right.
    """
    h, w = mask.shape
    scale = h / 1080
    min_len = max(10, int(float(cfg_court.get("hough_min_len_px", 80)) * scale))
    merge_px = float(cfg_court.get("merge_px", 5)) * scale
    max_cand = int(cfg_court.get("max_candidates", 10))

    skel = skeleton(mask)
    segs = cv2.HoughLinesP(skel.astype(np.uint8) * 255, 1, np.pi / 720, threshold=max(20, min_len // 2),
                           minLineLength=min_len, maxLineGap=max(3, int(10 * scale)))
    if segs is None:
        return np.zeros((0, 3)), np.zeros((0, 3))
    segs = segs.reshape(-1, 4).astype(np.float64)  # (N, 1, 4) before OpenCV 5, (N, 4) after
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)

    clusters: list[dict] = []
    for i in order:
        line = _hline(segs[i, :2], segs[i, 2:])
        if line[1] < 0:  # canonical sign: normal points down/right
            line = -line
        for c in clusters:
            cos = abs(float(line[:2] @ c["line"][:2]))
            if cos > np.cos(np.radians(1.0)):
                mid = (segs[i, :2] + segs[i, 2:]) / 2
                if abs(float(c["line"] @ np.r_[mid, 1.0])) < merge_px:
                    c["support"] += lengths[i]
                    break
        else:
            clusters.append({"line": line, "support": float(lengths[i])})

    def angle_from_horizontal(line):  # direction (-b, a)
        return np.degrees(np.arctan2(abs(line[0]), abs(line[1])))

    def y_at(line, x):
        return -(line[0] * x + line[2]) / line[1]

    def x_at(line, y):
        return -(line[1] * y + line[2]) / line[0]

    # Rank by the longest unbroken run along each line of pixels that are
    # line pixels *with mat on both sides* (`side` px away across the line),
    # gaps up to `gap` px bridged. A painted court line is one long such run;
    # texture — crowd, ad lettering — gives Hough plenty of segments, but its
    # neighbours are bright too.
    near = cv2.dilate(skel.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gap = max(2, int(round(4 * scale)))
    side = max(3, int(round(6 * scale)))  # beyond half the widest stroke

    def longest_run(line, horizontal):
        if horizontal:
            t = np.arange(w, dtype=np.float64)
            x, y = t, y_at(line, t)
            dx, dy = 0, side
        else:
            t = np.arange(h, dtype=np.float64)
            x, y = x_at(line, t), t
            dx, dy = side, 0
        finite = np.isfinite(x) & np.isfinite(y) & (np.abs(x) < 4 * w) & (np.abs(y) < 4 * h)
        xr = np.where(finite, np.round(np.where(finite, x, 0)), -1).astype(int)
        yr = np.where(finite, np.round(np.where(finite, y, 0)), -1).astype(int)
        inside = finite & (xr >= side) & (xr < w - side) & (yr >= side) & (yr < h - side)
        xi, yi = xr[inside], yr[inside]
        on = np.zeros(len(t), dtype=bool)
        on[inside] = near[yi, xi] & ~mask[yi - dy, xi - dx] & ~mask[yi + dy, xi + dx]
        if not on.any():
            return 0.0
        idx = np.flatnonzero(on)
        breaks = np.flatnonzero(np.diff(idx) > gap + 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        step = np.hypot(1.0, (line[0] / line[1]) if horizontal else (line[1] / line[0]))  # px per sample
        return float((ends - starts + 1).max() * step)

    cross = [c for c in clusters if angle_from_horizontal(c["line"]) < 20]
    longw = [c for c in clusters if angle_from_horizontal(c["line"]) > 25]
    for c in cross:
        c["support"] = longest_run(c["line"], True)
    for c in longw:
        c["support"] = longest_run(c["line"], False)
    cross = sorted((c for c in cross if c["support"] >= min_len), key=lambda c: -c["support"])[:max_cand]
    longw = sorted((c for c in longw if c["support"] >= min_len), key=lambda c: -c["support"])[:max_cand]

    cross = sorted(cross, key=lambda c: y_at(c["line"], w / 2))
    longw = sorted(longw, key=lambda c: x_at(c["line"], 0.75 * h))
    as_array = lambda cs: np.array([c["line"] for c in cs]).reshape(-1, 3)  # noqa: E731
    return as_array(cross), as_array(longw)


# --- Fit -------------------------------------------------------------------------


def _batch_homographies(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Homographies mapping 4 `src` points to 4 `dst` points, batched: `(N, 4, 2)` each.

    Returns `(H, ok)`; rows where the system is singular are not ok.
    """
    n = len(src)
    X, Y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    A = np.zeros((n, 8, 8))
    A[:, 0::2, 0], A[:, 0::2, 1], A[:, 0::2, 2] = X, Y, 1
    A[:, 1::2, 3], A[:, 1::2, 4], A[:, 1::2, 5] = X, Y, 1
    A[:, 0::2, 6], A[:, 0::2, 7] = -u * X, -u * Y
    A[:, 1::2, 6], A[:, 1::2, 7] = -v * X, -v * Y
    b = np.empty((n, 8))
    b[:, 0::2], b[:, 1::2] = u, v
    ok = np.abs(np.linalg.det(A)) > 1e-9
    H = np.full((n, 3, 3), np.nan)
    if ok.any():
        h = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        H[ok] = np.c_[h, np.ones(len(h))].reshape(-1, 3, 3)
    return H, ok


def _project_batch(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(B, N, 2)` image points and `(B, N)` denominators for `(B, 3, 3)` homographies."""
    p = np.einsum("bij,nj->bni", H, np.c_[pts, np.ones(len(pts))])
    return p[..., :2] / p[..., 2:3], p[..., 2]


def _cost(xy: np.ndarray, dist: np.ndarray, cap: float) -> np.ndarray:
    """Mean distance-to-line-pixel of projected points; off-image points cost `cap`."""
    h, w = dist.shape
    xi = np.round(xy[..., 0]).astype(np.int64)
    yi = np.round(xy[..., 1]).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & np.isfinite(xy).all(axis=-1)
    d = np.full(xi.shape, cap)
    d[inside] = np.minimum(dist[yi[inside], xi[inside]], cap)
    return d.mean(axis=-1)


def distance_to_lines(mask: np.ndarray) -> np.ndarray:
    """Per pixel: distance in px to the nearest line pixel."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def line_cost(H: np.ndarray, dist: np.ndarray, step: float = 0.05, cap: float = 10.0) -> float:
    """How far, on average, the projected painted lines land from line pixels (px)."""
    xy, _ = _project_batch(H[None], court_points(step))
    return float(_cost(xy, dist, cap)[0])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[..., :2] / p[..., 2:3]


def fit_court(mask: np.ndarray, cfg_court: Config) -> tuple[np.ndarray, float]:
    """Best court -> image homography over all candidate pairings, and its cost (px)."""
    h, w = mask.shape
    cross, longw = detect_lines(mask, cfg_court)
    if len(cross) < 2 or len(longw) < 2:
        raise CourtCheckError(f"found {len(cross)} cross and {len(longw)} lengthwise line candidates; need 2 of each")
    dist = distance_to_lines(mask)
    cap = 10.0 * h / 1080

    model_y = np.array(list(CROSS_LINES.values()))
    model_x = np.array(list(LONG_LINES.values()))
    img_pairs = [(i, j, k, l) for i, j in itertools.combinations(range(len(cross)), 2)
                 for k, l in itertools.combinations(range(len(longw)), 2)]
    mod_pairs = [(a, b, c, d) for a, b in itertools.combinations(range(len(model_y)), 2)
                 for c, d in itertools.combinations(range(len(model_x)), 2)]
    ip = np.array(img_pairs)
    mp = np.array(mod_pairs)

    # Image intersections for every image pairing: corners (i,k) (i,l) (j,k) (j,l).
    def corners(ci, lk):
        return _intersect(cross[ci], longw[lk])

    img = np.stack([corners(ip[:, 0], ip[:, 2]), corners(ip[:, 0], ip[:, 3]),
                    corners(ip[:, 1], ip[:, 2]), corners(ip[:, 1], ip[:, 3])], axis=1)
    mod = np.stack([np.c_[model_x[mp[:, 2]], model_y[mp[:, 0]]], np.c_[model_x[mp[:, 3]], model_y[mp[:, 0]]],
                    np.c_[model_x[mp[:, 2]], model_y[mp[:, 1]]], np.c_[model_x[mp[:, 3]], model_y[mp[:, 1]]]], axis=1)
    finite = np.isfinite(img).all(axis=(1, 2)) & (np.abs(img) < 10 * max(h, w)).all(axis=(1, 2))
    img = img[finite]

    coarse = court_points(0.25)
    best_H, best_cost = None, np.inf
    survivors = []
    for m in range(len(mod)):
        H, ok = _batch_homographies(np.broadcast_to(mod[m], img.shape), img)
        H = H[ok]
        if not len(H):
            continue
        quad, den = _project_batch(H, CORNERS)
        nl, nr, fr, fl = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        d1, d2 = fr - nl, fl - nr  # diagonals
        area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        valid = (
            (np.sign(den) == np.sign(den[:, :1])).all(axis=1)
            & (quad[..., 0] > -0.25 * w).all(axis=1) & (quad[..., 0] < 1.25 * w).all(axis=1)
            & (quad[..., 1] > -0.25 * h).all(axis=1) & (quad[..., 1] < 1.25 * h).all(axis=1)
            & (nl[:, 1] > fl[:, 1]) & (nr[:, 1] > fr[:, 1])  # near side lower in the image
            & (nl[:, 0] < nr[:, 0]) & (fl[:, 0] < fr[:, 0])  # left is left
            & (area > 0.05 * w * h)
        )
        H = H[valid]
        if not len(H):
            continue
        xy, _ = _project_batch(H, coarse)
        cost = _cost(xy, dist, cap)
        keep = np.argsort(cost)[:5]
        survivors.extend(zip(cost[keep], H[keep]))

    if not survivors:
        raise CourtCheckError("no candidate pairing gives a plausible court")
    survivors.sort(key=lambda s: s[0])
    for _, H in survivors[:20]:
        c = line_cost(H, dist, cap=cap)
        if c < best_cost:
            best_H, best_cost = H, c
    return best_H / best_H[2, 2], best_cost


def refine_court(
    H: np.ndarray, mask: np.ndarray, bands_px: Sequence[float] = (8, 4, 2.5)
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Least-squares homography from robust line fits near each projected model line.

    Returns the refined court -> image homography and the fitted image lines
    (homogeneous, by model line name). A model line with too few pixels near
    it is left out of the fit.
    """
    h = mask.shape[0]
    ys, xs = np.nonzero(mask)
    pix = np.c_[xs, ys].astype(np.float64)
    fitted: dict[str, np.ndarray] = {}
    for band in bands_px:
        band = band * h / 1080
        Hinv = np.linalg.inv(H)
        court_xy = project(Hinv, pix)
        fitted = {}
        for name, (x0, y0), (x1, y1) in painted_segments():
            a, b = project(H, np.array([[x0, y0], [x1, y1]]))
            line = _hline(a, b)
            near = np.abs(np.c_[pix, np.ones(len(pix))] @ line) < band
            # Only pixels whose court position lies along this painted stretch.
            along = (court_xy[:, 0] >= min(x0, x1) - 0.3) & (court_xy[:, 0] <= max(x0, x1) + 0.3) \
                & (court_xy[:, 1] >= min(y0, y1) - 0.3) & (court_xy[:, 1] <= max(y0, y1) + 0.3)
            sel = pix[near & along]
            if name in fitted:  # second stretch of the centre line
                sel = np.r_[fitted[name + "_pts"], sel]
            if len(sel) < 30:
                continue
            vx, vy, x, y = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fitted[name] = _hline(np.array([x, y]), np.array([x + vx, y + vy]))
            fitted[name + "_pts"] = sel
        lines = {k: v for k, v in fitted.items() if not k.endswith("_pts")}
        src, dst = [], []
        for cname, cy in CROSS_LINES.items():
            for lname, lx in LONG_LINES.items():
                if cname in lines and lname in lines:
                    src.append((lx, cy))
                    dst.append(_intersect(lines[cname], lines[lname]))
        if len(src) < 4:
            raise CourtCheckError(f"only {len(src)} line intersections found near the fitted court")
        H, _ = cv2.findHomography(np.array(src), np.array(dst), 0)
        H = H / H[2, 2]
    return H, {k: v for k, v in fitted.items() if not k.endswith("_pts")}


def court_dimensions(H: np.ndarray, lines: dict[str, np.ndarray]) -> tuple[float, float]:
    """Court length and width (m) measured from the fitted image lines.

    Length: each fitted lengthwise line meets both baselines; those two image
    points, mapped to the court, are a length apart. Width: each fitted cross
    line meets both doubles sidelines. Medians over the available lines;
    NaN when the needed lines were not fitted.
    """
    Hinv = np.linalg.inv(H)
    lengths, widths = [], []
    if "near_baseline" in lines and "far_baseline" in lines:
        for name in LONG_LINES:
            if name in lines:
                p = project(Hinv, np.array([_intersect(lines["near_baseline"], lines[name]),
                                            _intersect(lines["far_baseline"], lines[name])]))
                lengths.append(float(np.linalg.norm(p[1] - p[0])))
    if "left_doubles" in lines and "right_doubles" in lines:
        for name in CROSS_LINES:
            if name in lines:
                p = project(Hinv, np.array([_intersect(lines[name], lines["left_doubles"]),
                                            _intersect(lines[name], lines["right_doubles"])]))
                widths.append(float(np.linalg.norm(p[1] - p[0])))
    med = lambda v: float(np.median(v)) if v else float("nan")  # noqa: E731
    return med(lengths), med(widths)


def check_dimensions(length: float, width: float, tolerance: float) -> None:
    """The plan's invariant: court length 13.40 m (and width 6.10 m) within tolerance."""
    if not abs(length - COURT_LENGTH) <= tolerance:
        raise CourtCheckError(f"court length {length:.3f} m, expected {COURT_LENGTH} +/- {tolerance}")
    if not abs(width - DOUBLES_WIDTH) <= tolerance:
        raise CourtCheckError(f"court width {width:.3f} m, expected {DOUBLES_WIDTH} +/- {tolerance}")


def solve_background(bgr: np.ndarray, cfg_court: Config, H0: np.ndarray | None = None) -> dict:
    """Fit (or, given `H0`, only refine) the court on one background. Checked.

    Returns `{"court_to_image", "cost_px", "length_m", "width_m"}`.
    """
    mask = line_mask(bgr, cfg_court)
    H = H0 if H0 is not None else fit_court(mask, cfg_court)[0]
    H, lines = refine_court(H, mask)
    length, width = court_dimensions(H, lines)
    check_dimensions(length, width, float(cfg_court.get("length_tolerance_m", 0.2)))
    cost = line_cost(H, distance_to_lines(mask), cap=10.0 * bgr.shape[0] / 1080)
    max_cost = float(cfg_court.get("max_cost_px", 2.0)) * bgr.shape[0] / 1080
    if cost > max_cost:
        raise CourtCheckError(f"projected court lines sit {cost:.2f} px from line pixels on average (max {max_cost:.2f})")
    return {"court_to_image": H, "cost_px": cost, "length_m": length, "width_m": width}


def homography_from_corners(image_corners: Sequence[Sequence[float]]) -> np.ndarray:
    """Court -> image homography from the four doubles corners, in `CORNER_NAMES` order."""
    H = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), np.asarray(image_corners, dtype=np.float32))
    return H / H[2, 2]


# --- Drawing ------------------------------------------------------------------------


def draw_court(bgr: np.ndarray, H: np.ndarray, label: str | None = None) -> np.ndarray:
    """All painted lines projected onto `bgr` (cyan) with the corners circled (red)."""
    out = bgr.copy()
    t = max(1, round(bgr.shape[0] / 540))
    for _, a, b in painted_segments():
        p, q = project(H, np.array([a, b]))
        cv2.line(out, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (255, 255, 0), t, cv2.LINE_AA)
    for (x, y), name in zip(project(H, CORNERS), CORNER_NAMES):
        cv2.circle(out, (int(round(x)), int(round(y))), 6 * t, (0, 0, 255), t + 1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (20, bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (255, 255, 255), t, cv2.LINE_AA)
    return out


# --- Driver ----------------------------------------------------------------------------


def _views_digest(views_file: Path) -> str:
    """Fingerprint of the play spans and their ids — what `homography.json` is keyed on.

    Content, not size/mtime: `bda segment` rewrites view_segments.csv whenever
    its settings change, usually with identical spans.
    """
    views = pd.read_csv(views_file)
    play = views.loc[views["is_play"] == 1, ["segment_id", "start_frame", "end_frame"]].to_numpy(dtype=np.int64)
    return hashlib.sha1(play.tobytes()).hexdigest()


def _span_background(video_path, a: int, b: int, n: int) -> np.ndarray:
    margin = min(15, max(0, (b - a - n) // 2))
    idx = np.unique(np.linspace(a + margin, b - 1 - margin, n).astype(int))
    stack = np.stack(sample_frames(video_path, idx.tolist()))
    k = len(stack) // 2
    return np.partition(stack, k, axis=0)[k]


def solve_homographies(
    cfg: Config, match_id: str, video_path: str | Path, force: bool = False
) -> dict:
    """Fit the court for a match's play spans and write `homography.json`.

    Needs phase 2's `view_segments.csv`. Reused while that file and the
    `court:` config are unchanged; manual entries from `bda court-click`
    survive reruns unless `force` is given.
    """
    out_dir = cache_dir(cfg, match_id)
    views_file = out_dir / "view_segments.csv"
    if not views_file.exists():
        raise SystemExit(f"no {views_file}; run `bda segment` first")
    out_file = out_dir / "homography.json"
    cfg_court = cfg.get("court", Config({}))
    inputs = {"play_digest": _views_digest(views_file), "court": cfg_court.to_dict()}
    if out_file.exists() and not force:
        cached = json.loads(out_file.read_text())
        if cached.get("inputs") == inputs:
            print(f"court: reusing {out_file} (pass --force to recompute)")
            return cached

    views = pd.read_csv(views_file)
    info = probe_video(video_path)
    started = time.perf_counter()
    bg = play_background(video_path, views, int(cfg_court.get("background_frames", 61)))
    cv2.imwrite(str(out_dir / "court_background.png"), bg)
    cv2.imwrite(str(out_dir / "court_lines.png"), line_mask(bg, cfg_court).astype(np.uint8) * 255)
    try:
        base = solve_background(bg, cfg_court)
    except CourtCheckError as exc:
        raise SystemExit(f"court: automatic fit failed on the match background: {exc}\n"
                         f"court: run `bda court-click --video {video_path} --match-id {match_id}`")
    label = f"length {base['length_m']:.2f} m  width {base['width_m']:.2f} m  cost {base['cost_px']:.2f} px"
    cv2.imwrite(str(out_dir / "court_overlay.png"), draw_court(bg, base["court_to_image"], label))
    print(f"court: match fit in {time.perf_counter() - started:.1f}s — {label}")

    n_span = int(cfg_court.get("span_frames", 7))
    max_cost = float(cfg_court.get("max_cost_px", 2.0)) * info.height / 1080
    cap = 10.0 * info.height / 1080
    segments = []
    for seg in views[views["is_play"] == 1].itertuples():
        a, b = int(seg.start_frame), int(seg.end_frame)
        span_bg = _span_background(video_path, a, b, n_span)
        dist = distance_to_lines(line_mask(span_bg, cfg_court))
        entry = {"segment_id": int(seg.segment_id), "start_frame": a, "end_frame": b}
        cost = line_cost(base["court_to_image"], dist, cap=cap)
        if cost <= max_cost:
            fit, status = base, "match"
        else:
            try:  # the camera moved: refit on this span alone
                fit, status = solve_background(span_bg, cfg_court), "span"
            except CourtCheckError as exc:
                print(f"court: span {a}-{b} needs manual corners ({exc})")
                entry.update({"image_to_court": None, "source": "needs_manual", "cost_px": round(cost, 3)})
                segments.append(entry)
                continue
            cost = fit["cost_px"]
        entry.update({
            "image_to_court": np.linalg.inv(fit["court_to_image"]).tolist(),
            "source": f"auto:{status}",
            "cost_px": round(cost, 3),
            "length_m": round(fit["length_m"], 3),
            "width_m": round(fit["width_m"], 3),
        })
        segments.append(entry)

    doc = {
        "match_id": match_id,
        "video": str(video_path),
        "image_size": [info.width, info.height],
        "court": "metres; origin at court centre; x right as seen from the main camera, y away from it",
        "inputs": inputs,
        "segments": segments,
    }
    out_file.write_text(json.dumps(doc, indent=1))
    n_ok = sum(s["image_to_court"] is not None for s in segments)
    n_refit = sum(s["source"] == "auto:span" for s in segments)
    print(f"court: wrote {out_file} — {n_ok}/{len(segments)} play spans with a homography"
          f" ({n_refit} refitted on their own frames)")
    return doc


# --- Manual fallback -------------------------------------------------------------------


def click_corners(bgr: np.ndarray, window: str = "bda court-click") -> list[tuple[float, float]] | None:
    """Ask for the four doubles corners by mouse. Enter accepts, Backspace undoes, Esc aborts."""
    pts: list[tuple[float, float]] = []
    scale = min(1.0, 1600 / bgr.shape[1])
    shown = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            img = shown.copy()
            for i, (x, y) in enumerate(pts):
                c = (int(x * scale), int(y * scale))
                cv2.circle(img, c, 6, (0, 0, 255), 2)
                cv2.putText(img, CORNER_NAMES[i], (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            prompt = (f"click the {CORNER_NAMES[len(pts)]} doubles corner" if len(pts) < 4
                      else "Enter to accept, Backspace to undo")
            cv2.putText(img, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(window, img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (8, 127) and pts:
                pts.pop()
            if key in (13, 10) and len(pts) == 4:
                return pts
    finally:
        cv2.destroyWindow(window)


def apply_manual_corners(
    cfg: Config, match_id: str, video_path: str | Path, corners: Sequence[Sequence[float]],
    segment_ids: Sequence[int] | None = None,
) -> dict:
    """Write a homography from four clicked corners into `homography.json`.

    The corners are refined against the line pixels when that passes the
    checks; otherwise the raw four-point homography is used. Applies to the
    given play segments, or to every one marked `needs_manual` (all play
    segments if there is no file yet).
    """
    out_dir = cache_dir(cfg, match_id)
    cfg_court = cfg.get("court", Config({}))
    views = pd.read_csv(out_dir / "view_segments.csv")
    out_file = out_dir / "homography.json"
    doc = json.loads(out_file.read_text()) if out_file.exists() else None

    H = homography_from_corners(corners)
    bg_file = out_dir / "court_background.png"
    bg = cv2.imread(str(bg_file)) if bg_file.exists() else play_background(
        video_path, views, int(cfg_court.get("background_frames", 61)))
    try:
        fit, refined = solve_background(bg, cfg_court, H0=H), True
    except CourtCheckError as exc:
        print(f"court-click: refinement failed ({exc}); using the clicked corners as they are")
        fit = {"court_to_image": H, "cost_px": line_cost(H, distance_to_lines(line_mask(bg, cfg_court))),
               "length_m": COURT_LENGTH, "width_m": DOUBLES_WIDTH}
        refined = False
    cv2.imwrite(str(out_dir / "court_overlay.png"), draw_court(bg, fit["court_to_image"], "manual corners"))

    if doc is None:
        info = probe_video(video_path)
        doc = {"match_id": match_id, "video": str(video_path), "image_size": [info.width, info.height],
               "court": "metres; origin at court centre; x right as seen from the main camera, y away from it",
               "inputs": {"play_digest": _views_digest(out_dir / "view_segments.csv"), "court": cfg_court.to_dict()},
               "segments": [{"segment_id": int(s.segment_id), "start_frame": int(s.start_frame),
                             "end_frame": int(s.end_frame), "image_to_court": None, "source": "needs_manual"}
                            for s in views[views["is_play"] == 1].itertuples()]}
    targets = set(segment_ids) if segment_ids else {
        s["segment_id"] for s in doc["segments"] if s["image_to_court"] is None}
    for s in doc["segments"]:
        if s["segment_id"] in targets:
            s.update({"image_to_court": np.linalg.inv(fit["court_to_image"]).tolist(),
                      "source": "manual:refined" if refined else "manual:corners",
                      "cost_px": round(fit["cost_px"], 3), "length_m": round(fit["length_m"], 3),
                      "width_m": round(fit["width_m"], 3)})
    out_file.write_text(json.dumps(doc, indent=1))
    print(f"court-click: wrote {len(targets)} segment(s) to {out_file}")
    return doc


def load_homographies(path: str | Path) -> pd.DataFrame:
    """`homography.json` as rows: `segment_id, start_frame, end_frame, H` (image -> court, or None)."""
    doc = json.loads(Path(path).read_text())
    return pd.DataFrame(
        [(s["segment_id"], s["start_frame"], s["end_frame"],
          None if s["image_to_court"] is None else np.array(s["image_to_court"]))
         for s in doc["segments"]],
        columns=["segment_id", "start_frame", "end_frame", "H"],
    )
