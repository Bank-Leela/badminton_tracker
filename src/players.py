"""Phase 4: both players — box, pose, and where they stand on the court.

Two stages, cached separately in `data/cache/<match_id>/`:

* `detect_poses` runs YOLO-pose over every play-span frame, with ByteTrack
  inside each span, and keeps every person whose feet land near the court:
  `poses.parquet`. The expensive half (~40 fps at 1280 px on an RTX 5070).
* `select_players` picks the two players out of those, puts their feet on the
  court through the span's homography, and asserts the speed invariant:
  `players.parquet`. Seconds; reruns whenever its settings change.

Pose runs at 1280 px, not the model's native 640: at 640 the far player
(150-220 px tall at 1080p) is missed in some frames and found at confidence
0.3-0.8; at 1280 they are found in every frame sampled, ankles confident in
99%+. 1920 adds nothing but crowd. So the crop-and-upscale second pass the
plan allows for is not needed.

Keypoints are stored raw. Smoothing is left to the reader (`smooth_keypoints`)
so that no stored measurement carries a smoothing window's lag or flattened
peaks — phase 5 needs wrist-speed peaks intact.
"""

from __future__ import annotations

import os

# Before ultralytics is first imported: no update checks, analytics or
# pip installs from inside a batch run.
os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import hashlib
import itertools
import json
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import Config, cache_dir
from court import load_homographies
from video import iter_frames, probe_video

# COCO keypoint order, as YOLO-pose emits it.
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
L_ANKLE, R_ANKLE = KEYPOINTS.index("left_ankle"), KEYPOINTS.index("right_ankle")

L_SHOULDER, R_SHOULDER = KEYPOINTS.index("left_shoulder"), KEYPOINTS.index("right_shoulder")
L_HIP, R_HIP = KEYPOINTS.index("left_hip"), KEYPOINTS.index("right_hip")
L_KNEE, R_KNEE = KEYPOINTS.index("left_knee"), KEYPOINTS.index("right_knee")

# Median CIELAB colour (OpenCV 8-bit scaling) of the shirt and of the shorts:
# what tells the two players apart after they change ends.
COLOUR_COLUMNS = ["shirt_l", "shirt_a", "shirt_b", "shorts_l", "shorts_a", "shorts_b"]
POSE_COLUMNS = ["frame", "segment_id", "track_id", "conf", "x1", "y1", "x2", "y2",
                "foot_x", "foot_y", "ankles", "court_x", "court_y", *COLOUR_COLUMNS, "keypoints"]


# --- Geometry --------------------------------------------------------------------------


def foot_points(boxes: np.ndarray, kpts: np.ndarray, min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """Image point each person stands on, `(N, 2)`, and whether it came from the ankles.

    The midpoint of the ankles when both are confident; otherwise the bottom
    centre of the box, which is lower than the ankles by a shoe's height but
    never lands on the wrong person.
    """
    ank = kpts[:, [L_ANKLE, R_ANKLE]]
    both = (ank[:, :, 2] >= min_conf).all(axis=1)
    box_foot = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    return np.where(both[:, None], ank[:, :, :2].mean(axis=1), box_foot), both


def _median_lab(frame: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, x1 = int(max(0, x0)), int(min(w, x1))
    y0, y1 = int(max(0, y0)), int(min(h, y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return np.full(3, np.nan)
    bgr = np.median(frame[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(np.uint8)
    return cv2.cvtColor(bgr[None, None], cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def clothing_colours(frame: np.ndarray, kpts: np.ndarray, min_conf: float) -> np.ndarray:
    """Median Lab colour of each person's shirt and shorts, `(N, 6)`; NaN where unseen.

    Shirt: the middle of the shoulder-hip quad. Shorts: hips down to a third
    of the way to the knees. Inset so arms, skin and court stay out.
    """
    out = np.full((len(kpts), 6), np.nan, np.float32)
    for i, k in enumerate(kpts):
        torso = k[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
        if (torso[:, 2] >= min_conf).all():
            xa, xb = torso[:, 0].min(), torso[:, 0].max()
            ya, yb = torso[:2, 1].mean(), torso[2:, 1].mean()
            dx, dy = 0.25 * (xb - xa), 0.15 * (yb - ya)
            out[i, :3] = _median_lab(frame, xa + dx, ya + dy, xb - dx, yb - dy)
        legs = k[[L_HIP, R_HIP, L_KNEE, R_KNEE]]
        if (legs[:, 2] >= min_conf).all():
            xa, xb = legs[:2, 0].min(), legs[:2, 0].max()
            ya = legs[:2, 1].mean()
            yb = ya + (legs[2:, 1].mean() - ya) / 3
            dx = 0.1 * (xb - xa)
            out[i, 3:] = _median_lab(frame, xa + dx, ya, xb - dx, yb)
    return out


def to_court(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Image points to court metres through `image_to_court`; NaN beyond the horizon.

    A point above the horizon line maps through w < 0 (taking the court
    centre's sign as positive, since H is defined up to scale) to a
    finite-looking point behind the camera. Those are NaN here.
    """
    centre = np.linalg.inv(H) @ np.array([0.0, 0.0, 1.0])  # court centre, homogeneous image point
    sign = np.sign(H[2] @ np.r_[centre[:2] / centre[2], 1.0])
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    out = p[:, :2] / p[:, 2:3]
    out[p[:, 2] * sign <= 0] = np.nan
    return out


# --- Stage 1: pose and tracks over the play spans ----------------------------------------


def _ultralytics_device(name: str) -> str:
    from shuttle import resolve_device

    dev = resolve_device(name)
    return f"cuda:{dev.index or 0}" if dev.type == "cuda" else dev.type


def load_pose_model(cfg: Config):
    from ultralytics import YOLO

    path = Path(cfg.paths.pose_model)
    if not path.exists():
        raise SystemExit(f"no pose model at {path}; see README (phase 4) for the download")
    return YOLO(str(path))


def _prefetch(gen: Iterator, depth: int) -> Iterator:
    """Run a generator in a thread so decoding overlaps inference."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()
    error: list[BaseException] = []

    def work():
        try:
            for item in gen:
                q.put(item)
        except BaseException as exc:  # re-raised in the consumer
            error.append(exc)
        finally:
            q.put(done)

    threading.Thread(target=work, daemon=True).start()
    while (item := q.get()) is not done:
        yield item
    if error:
        raise error[0]


def _batches(video_path, a: int, b: int, n: int) -> Iterator[tuple[int, list[np.ndarray]]]:
    batch: list[np.ndarray] = []
    first = a
    for idx, frame in iter_frames(video_path, a, b):
        if not batch:
            first = idx
        batch.append(frame)
        if len(batch) == n:
            yield first, batch
            batch = []
    if batch:
        yield first, batch


def _tracker(cfg_det: Config):
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.utils import IterableSimpleNamespace

    return BYTETracker(args=IterableSimpleNamespace(tracker_type="bytetrack", **cfg_det.tracker.to_dict()))


_INT_COLUMNS = {"frame", "segment_id", "track_id", "player_id"}


def _keypoint_array(kp: np.ndarray) -> pa.Array:
    flat = pa.array(np.ascontiguousarray(kp, dtype=np.float32).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(pa.FixedSizeListArray.from_arrays(flat, 3), len(KEYPOINTS))


def _table(cols: dict[str, np.ndarray], kp: np.ndarray, order: list[str]) -> pa.Table:
    """Columns plus `keypoints[17][3]` (float32 `x, y, conf`) as a parquet table."""
    def typed(name, v):
        if name in _INT_COLUMNS:
            return v.astype(np.int32)
        return v if v.dtype == np.bool_ or v.dtype == object else v.astype(np.float32)

    arrays = {k: pa.array(typed(k, np.asarray(v))) for k, v in cols.items()}
    arrays["keypoints"] = _keypoint_array(kp)
    return pa.table({c: arrays[c] for c in order})


def _pose_table(rows: dict[str, list]) -> pa.Table:
    if not rows["frame"]:
        cols = {k: np.zeros(0, bool if k == "ankles" else np.float32) for k in POSE_COLUMNS if k != "keypoints"}
        return _table(cols, np.zeros((0, len(KEYPOINTS), 3), np.float32), POSE_COLUMNS)
    cols = {k: np.concatenate(v) for k, v in rows.items() if k != "keypoints"}
    return _table(cols, np.concatenate(rows["keypoints"]), POSE_COLUMNS)


def keypoints_array(table_or_df) -> np.ndarray:
    """The `keypoints` column as an `(N, 17, 3)` float array of `x, y, conf`."""
    if isinstance(table_or_df, pd.DataFrame):
        if len(table_or_df) == 0:
            return np.zeros((0, len(KEYPOINTS), 3), np.float32)
        return np.stack([np.stack(k) for k in table_or_df["keypoints"]]).astype(np.float32)
    col = table_or_df.column("keypoints").combine_chunks()
    return col.flatten().flatten().to_numpy().reshape(-1, len(KEYPOINTS), 3)


def _detect_inputs(cfg: Config, views: pd.DataFrame) -> dict:
    from shuttle import _play_digest

    model = Path(cfg.paths.pose_model)
    det = cfg.players.detect.to_dict()
    return {"play_digest": _play_digest(views), "model": [model.name, model.stat().st_size], "detect": det}


def detect_poses(cfg: Config, match_id: str, video_path: str | Path, force: bool = False) -> Path:
    """YOLO-pose + ByteTrack over every play span; write `poses.parquet`.

    Every tracked person whose feet project inside `region_m` of the court
    centre is kept (players, line judges, umpire, anyone walking on); the
    choice of the two players is `select_players`'. The tracker is reset at
    each play span: a camera cut breaks any identity it could carry.

    Cached on the play spans and the `players.detect` settings, not on
    `homography.json`: the court only decides who is kept, through a region
    far wider than the court, so a refit court does not need hours of
    re-detection. If the play spans change, this stops (like `bda track`).
    """
    out_dir = cache_dir(cfg, match_id)
    views_file = out_dir / "view_segments.csv"
    hom_file = out_dir / "homography.json"
    for f, cmd in [(views_file, "segment"), (hom_file, "court")]:
        if not f.exists():
            raise SystemExit(f"no {f}; run `bda {cmd}` first")
    out_file = out_dir / "poses.parquet"
    meta_file = out_dir / "poses.meta.json"
    views = pd.read_csv(views_file)
    inputs = _detect_inputs(cfg, views)
    if out_file.exists() and not force:
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        if meta.get("inputs") == inputs:
            print(f"poses: reusing {out_file} (pass --force to recompute)")
            return out_file
        what = "the play spans" if meta.get("inputs", {}).get("play_digest") != inputs["play_digest"] \
            else "the model or players.detect settings"
        raise SystemExit(f"{out_file} exists but {what} changed since; pass --force to redetect (~20 min a match)")

    cfg_det = cfg.players.detect
    homs = load_homographies(hom_file).set_index("segment_id")
    info = probe_video(video_path)
    model = load_pose_model(cfg)
    device = _ultralytics_device(cfg.get("device", "auto"))
    half = device.startswith("cuda") and cfg_det.get("precision", "fp16") == "fp16"
    predict_kw = dict(imgsz=int(cfg_det.imgsz), conf=float(cfg_det.conf), device=device,
                      quantize=16 if half else None, verbose=False)
    rx, ry = cfg_det.region_m
    kpt_conf = float(cfg_det.kpt_conf)
    bs = int(cfg_det.batch_size)

    play = views[views["is_play"] == 1]
    rows: dict[str, list] = {c: [] for c in POSE_COLUMNS}
    skipped = []
    started = time.perf_counter()
    total = int((play["end_frame"] - play["start_frame"]).sum())
    with tqdm(total=total, unit="frame", desc=f"pose {match_id}") as bar:
        for span in play.itertuples():
            a, b, sid = int(span.start_frame), int(span.end_frame), int(span.segment_id)
            H = homs.H.get(sid)
            if H is None:  # no homography (`bda court-click` it): nobody can be placed
                skipped.append(sid)
                bar.update(b - a)
                continue
            tracker = _tracker(cfg_det)
            for first, frames in _prefetch(_batches(video_path, a, b, bs), depth=3):
                results = model.predict(frames, **predict_kw)
                if len(results) != len(frames):
                    raise RuntimeError(f"{len(results)} results for {len(frames)} frames at {first}")
                for i, (res, frame) in enumerate(zip(results, frames)):
                    boxes = res.boxes.cpu().numpy()
                    if len(boxes) == 0:
                        tracker.update(boxes, frame)
                        continue
                    kpts = res.keypoints.data.cpu().numpy()
                    foot, ankles = foot_points(boxes.xyxy, kpts, kpt_conf)
                    court = to_court(H, foot)
                    keep = (np.abs(court[:, 0]) <= rx) & (np.abs(court[:, 1]) <= ry)  # NaN fails both
                    tracks = tracker.update(boxes[keep], frame)
                    if len(tracks) == 0:
                        continue
                    # Rows are the detections themselves (not the Kalman boxes);
                    # `idx` indexes into what the tracker was given.
                    det = np.flatnonzero(keep)[tracks[:, -1].astype(int)]
                    n = len(det)
                    rows["frame"].append(np.full(n, first + i))
                    rows["segment_id"].append(np.full(n, sid))
                    rows["track_id"].append(tracks[:, 4])
                    rows["conf"].append(boxes.conf[det])
                    for j, c in enumerate(["x1", "y1", "x2", "y2"]):
                        rows[c].append(boxes.xyxy[det, j])
                    rows["foot_x"].append(foot[det, 0])
                    rows["foot_y"].append(foot[det, 1])
                    rows["ankles"].append(ankles[det])
                    rows["court_x"].append(court[det, 0])
                    rows["court_y"].append(court[det, 1])
                    colours = clothing_colours(frame, kpts[det], kpt_conf)
                    for j, c in enumerate(COLOUR_COLUMNS):
                        rows[c].append(colours[:, j])
                    rows["keypoints"].append(kpts[det])
                bar.update(len(frames))
    elapsed = time.perf_counter() - started

    table = _pose_table(rows)
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_file)
    meta_file.write_text(json.dumps({
        "inputs": inputs, "video": str(Path(video_path).resolve()), "fps": info.fps,
        "frames": total, "seconds": round(elapsed, 1), "skipped_segments": skipped,
    }, indent=1))
    print(f"poses: wrote {out_file} — {table.num_rows} tracked people over {total} play frames "
          f"in {elapsed:.0f}s ({total / elapsed:.1f} fps)"
          + (f"; {len(skipped)} span(s) without a homography skipped" if skipped else ""))
    return out_file


def load_poses(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """`poses.parquet` as a DataFrame without the keypoints, plus the `(N, 17, 3)` keypoints."""
    table = pq.read_table(path)
    return table.drop(["keypoints"]).to_pandas(), keypoints_array(table)


# --- Stage 2: the two players ------------------------------------------------------------


class PlayerCheckError(AssertionError):
    """A player trajectory broke a physical invariant. Raised explicitly, like
    `court.CourtCheckError`, so `python -O` cannot disable it."""


SIDES = ("near", "far")
PLAYER_COLUMNS = ["frame", "player_id", "side", "segment_id", "track_id", "conf", "x1", "y1", "x2", "y2",
                  "foot_x", "foot_y", "foot_src", "court_x", "court_y", "speed", "keypoints"]


_HEAD = [KEYPOINTS.index(n) for n in ("nose", "left_eye", "right_eye", "left_ear", "right_ear")]
_BODY = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]


def body_extent(kp: np.ndarray, boxes: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
    """`(N, 3)`: left, top, right of each person's head and body — arms and
    racket left out. What can hide someone standing behind them. The head
    reaches at least a fifth of the box height above the shoulders (a player
    facing away has no confident face keypoints). Falls back to the box
    where the keypoints are unsure."""
    body = kp[:, _BODY]
    ok = body[:, :, 2] >= min_conf
    x = np.where(ok, body[:, :, 0], np.nan)
    head, shoulders = kp[:, _HEAD], kp[:, [L_SHOULDER, R_SHOULDER]]
    head_y = np.where(head[:, :, 2] >= min_conf, head[:, :, 1], np.nan)
    shoulder_y = np.where(shoulders[:, :, 2] >= min_conf, shoulders[:, :, 1], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows fall back below
        x1, x2 = np.nanmin(x, axis=1), np.nanmax(x, axis=1)
        head_top = np.nanmin(head_y, axis=1)
        shoulder_top = np.nanmin(shoulder_y, axis=1) - 0.2 * (boxes[:, 3] - boxes[:, 1])
    top = np.maximum(boxes[:, 1], np.fmin(head_top, shoulder_top))  # fmin: NaN-aware
    pad = 0.2 * (boxes[:, 2] - boxes[:, 0])
    x1 = np.where(np.isfinite(x1), x1 - pad, boxes[:, 0])
    x2 = np.where(np.isfinite(x2), x2 + pad, boxes[:, 2])
    top = np.where(np.isfinite(top), top, boxes[:, 1])
    return np.c_[x1, top, x2]


def place_feet(poses: pd.DataFrame, kp: np.ndarray, homs: pd.DataFrame, height: int,
               ankle_conf: float) -> pd.DataFrame:
    """Recompute feet and court position with the current homographies.

    `foot_src`: `ankles` (both confident), `box` (bottom centre of the box),
    or `edge` — the box is cut by the bottom of the frame, so the feet are out
    of shot and `court_x/y` is NaN (`edge_x/y` keeps the box-bottom point,
    good enough to say which half someone is on). `choose_players` adds
    `occluded`, `select_players` `flagged` (`blank_flagged`).

    The ankle midpoint, not a point weighted towards the planted foot: on
    twelve matches weighting by each leg's knee-to-ankle drop tripled the
    single-frame jumps (the weight swaps feet during ordinary footwork).
    """
    boxes = poses[["x1", "y1", "x2", "y2"]].to_numpy(np.float64)
    foot, ankles = foot_points(boxes, kp, ankle_conf)
    edge = ~ankles & (boxes[:, 3] >= height - 2)
    court = np.full((len(poses), 2), np.nan)
    H_of = dict(zip(homs.segment_id, homs.H))
    for sid, idx in poses.groupby("segment_id").indices.items():
        if H_of.get(sid) is not None:
            court[idx] = to_court(H_of[sid], foot[idx])
    out = poses.copy()
    out[["body_x1", "body_y1", "body_x2"]] = body_extent(kp, boxes)
    out["foot_x"], out["foot_y"] = foot[:, 0], foot[:, 1]
    out["ankle_conf"] = kp[:, [L_ANKLE, R_ANKLE], 2].min(axis=1)
    out["foot_src"] = np.where(ankles, "ankles", np.where(edge, "edge", "box"))
    out["edge_x"], out["edge_y"] = court[:, 0], court[:, 1]
    court[edge] = np.nan
    out["court_x"], out["court_y"] = court[:, 0], court[:, 1]
    return out


def choose_players(placed: pd.DataFrame, cfg_sel: Config) -> pd.DataFrame:
    """At most one person per half per frame: the player.

    Candidates stand within the court plus margins (which leaves out line
    judges and the umpire). Their half is the sign of y — except within
    `net_band_m` of the net, where feet by the post or ankles in a lunge can
    read a few cm over it: there a person's half is where their track spends
    most of the span. Per play span and half, the track seen there most often
    wins a frame; anyone else on court (a mop, a referee checking the
    shuttle) is seen for far fewer frames.
    """
    from court import COURT_LENGTH, DOUBLES_WIDTH

    hx = DOUBLES_WIDTH / 2 + float(cfg_sel.side_margin_m)
    hy = COURT_LENGTH / 2 + float(cfg_sel.back_margin_m)
    x, y = placed["edge_x"], placed["edge_y"]
    cand = placed[(x.abs() <= hx) & (y.abs() <= hy)].copy()
    usual = np.sign(cand.groupby(["segment_id", "track_id"])["edge_y"].transform("median"))
    near_net = cand["edge_y"].abs() < float(cfg_sel.net_band_m)
    cand["side"] = np.where(np.where(near_net, usual, np.sign(cand["edge_y"])) < 0, "near", "far")
    cand["n_track"] = cand.groupby(["segment_id", "side", "track_id"])["frame"].transform("size")
    cand = cand.sort_values(["frame", "side", "n_track", "conf"], ascending=[True, True, False, False])
    chosen = cand.drop_duplicates(["frame", "side"]).drop(columns="n_track")
    return mark_occluded(chosen, float(cfg_sel.hidden_ankle_conf))


def mark_occluded(chosen: pd.DataFrame, min_ankle_conf: float) -> pd.DataFrame:
    """Blank the far player's court position when the near player hides their feet.

    From the main camera the near player often stands right in front of the
    far one. The far player's ankles are then guessed on the near player's
    legs or head, which reads 2-3 m too close. A far foot point on the near
    player (`body_extent`: head to the bottom of the box, arms left out) is
    `foot_src = occluded`, `court_x/y` NaN — unless both ankles are still
    seen with `min_ankle_conf`: on four matches those read as well as feet
    in the open (median 0.17 m from the path either side), while less sure
    ones were off by 0.3-1 m at the median.
    """
    near = chosen[chosen["side"] == "near"].set_index("frame")
    far = chosen[chosen["side"] == "far"]
    fx, fy = far["foot_x"].to_numpy(), far["foot_y"].to_numpy()

    def within(cols):
        b = near[cols].reindex(far["frame"]).to_numpy()
        return (fx >= b[:, 0]) & (fx <= b[:, 2]) & (fy >= b[:, 1]) & (fy <= b[:, 3])  # NaN box: False

    on_body = within(["body_x1", "body_y1", "body_x2", "y2"])
    hidden = on_body & (far["ankle_conf"].to_numpy() < min_ankle_conf)
    # No ankles at all, feet from the bottom of the box: a box cut short by
    # the near player in front (the far player bending for the shuttle
    # behind them) — its bottom ends above the near player's feet, and a
    # third or more of its width is across the near player's box.
    nb = near[["x1", "x2", "y2"]].reindex(far["frame"]).to_numpy()
    fb = far[["x1", "x2", "y2"]].to_numpy()
    overlap = np.minimum(fb[:, 1], nb[:, 1]) - np.maximum(fb[:, 0], nb[:, 0])
    cut_short = (overlap >= (fb[:, 1] - fb[:, 0]) / 3) & (fb[:, 2] < nb[:, 2])  # NaN: False
    hidden |= (far["foot_src"].to_numpy() == "box") & (within(["x1", "y1", "x2", "y2"]) | cut_short)
    out = chosen.copy()
    rows = far.index[hidden]
    out.loc[rows, "foot_src"] = "occluded"
    out.loc[rows, ["court_x", "court_y"]] = np.nan
    return out


def _colour_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Lab distance over the colours both have (shirt and/or shorts)."""
    d = (a - b) ** 2
    ok = ~np.isnan(d)
    return float(np.sqrt(d[ok].sum() * len(d) / ok.sum())) if ok.any() else 0.0


def _wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean over rows, per column, ignoring NaN; NaN where a column has nothing."""
    ok = np.isfinite(x)
    ww = np.where(ok, w[:, None], 0.0)
    with np.errstate(invalid="ignore"):
        return np.where(ww.sum(0) > 0, (np.where(ok, x, 0.0) * ww).sum(0) / ww.sum(0), np.nan)


def _fit_appearance(near: np.ndarray, far: np.ndarray, w: np.ndarray, state: np.ndarray,
                    shift: np.ndarray) -> tuple[float, list]:
    """Fit two players' colours to an assignment of blocks; return the weighted misfit and `A`.

    `state[b]` is the player near in block b. Model: the near player looks like
    `A[state]`, the far player like `A[1 - state] + shift`.
    """
    A = [_wmean(np.concatenate([near[state == k], far[state == 1 - k] - shift]),
                np.concatenate([w[state == k], w[state == 1 - k]])) for k in (0, 1)]
    A = [np.where(np.isfinite(a), a, 0.0) for a in A]
    cost = sum(w[b] * (_colour_distance(near[b], A[state[b]]) + _colour_distance(far[b], A[1 - state[b]] + shift))
               for b in range(len(state)))
    return float(cost), A


def assign_identity(chosen: pd.DataFrame, views: pd.DataFrame, fps: float, cfg_sel: Config) -> tuple[dict, dict]:
    """Which player is near in each play span: `{segment_id: near_player_id}`, plus a report.

    Players change ends only in a break — between games, and at 11 in the
    third — so only gaps between play spans of at least `min_break_s` can be
    a change of ends. The spans between such gaps form blocks with the same
    player near. Every way of changing ends at up to `max_end_changes` of
    those gaps is tried; each is scored by how well two players' shirt and
    shorts colours (plus one far-end colour shift) explain every block's
    near and far player, plus `switch_cost` per change. The best wins. Whole
    blocks are compared, so noisy spans and players dressed alike (white
    shirts at All England 2026, told apart by their shorts) still separate.

    Player 0 is whoever is near in the first play span.
    """
    play = views[views["is_play"] == 1].sort_values("start_frame")
    have = set(chosen["segment_id"].unique())
    play = play[play["segment_id"].isin(have)]
    if play.empty:
        return {}, {"switches": [], "separation": None, "colours": {}}
    gap = (play["start_frame"] - play["end_frame"].shift(1)).to_numpy() / fps
    block = np.cumsum(np.r_[0, gap[1:] >= float(cfg_sel.min_break_s)])
    block_of = dict(zip(play["segment_id"].astype(int), block))
    c = chosen.assign(block=chosen["segment_id"].map(block_of))
    n_blocks = int(block.max()) + 1

    def side_feats(side):
        g = c[c["side"] == side].groupby("block")[COLOUR_COLUMNS].median()
        return g.reindex(range(n_blocks)).to_numpy(np.float64)

    near, far = side_feats("near"), side_feats("far")
    both = c.groupby(["block", "side"]).size().unstack("side").reindex(range(n_blocks)).fillna(0).min(axis=1)
    w = np.sqrt(np.minimum(both.to_numpy(np.float64), fps * 60) / (fps * 60))  # a minute of both = a full say

    # The far end is lit differently and the far player is seen against the
    # boards: one colour shift for the match. Each player spends about half
    # of it at each end, so a first estimate is far minus near over
    # everything; the second pass takes it from the first pass's answer.
    # Fixed within a pass, not refitted per assignment, where it would bend
    # to suit a wrong one. (Nobody ever changing ends makes the two players
    # look alike, and then no change of ends wins on `switch_cost`: right.)
    switch = float(cfg_sel.switch_cost)
    states = []
    for n in range(min(int(cfg_sel.max_end_changes), n_blocks - 1) + 1):
        for at in itertools.combinations(range(1, n_blocks), n):
            state = np.zeros(n_blocks, int)
            for b in at:
                state[b:] = 1 - state[b:]
            states.append((n, state))

    def search(shift):
        scored = []
        for n, s in states:
            cost, A = _fit_appearance(near, far, w, s, shift)
            scored.append((cost + switch * n, s, A))
        return sorted(scored, key=lambda r: r[0])

    shift = np.nan_to_num(_wmean(far, w) - _wmean(near, w))
    _, state, A = search(shift)[0]
    shift = np.nan_to_num(_wmean(far - np.stack([A[1 - s] for s in state]), w))
    results = search(shift)
    best, state, A = results[0]
    margin = results[1][0] - best if len(results) > 1 else None

    near_of_seg = {int(s): int(state[b]) for s, b in block_of.items()}
    switches = []
    first = play.groupby(block)["segment_id"].first()
    for b in np.flatnonzero(np.diff(state)) + 1:
        sid = int(first[b])
        i = int(np.flatnonzero(play["segment_id"].to_numpy() == sid)[0])
        switches.append({"segment_id": sid, "frame": int(play["start_frame"].iloc[i]), "gap_s": round(float(gap[i]), 1)})
    report = {
        "switches": switches,
        "breaks": int(n_blocks - 1),
        "separation": round(_colour_distance(A[0], A[1]), 1),
        "margin": None if margin is None else round(float(margin), 1),
        "colours": {str(k): [round(float(v), 1) for v in A[k]] for k in (0, 1)},
        "far_shift": [round(float(v), 1) for v in shift],
    }
    return near_of_seg, report


def foreign_tracks(chosen: pd.DataFrame, near_of: dict, ident: dict, max_dist: float) -> pd.DataFrame:
    """Chosen tracks whose clothes match neither player: `(segment_id, track_id) -> n, dist`.

    With a player off court (at their bag in an interval), whoever else is
    on that half — a court cleaner, an official — is the only candidate and
    gets chosen. Their median colour sits far from the player expected
    there (`assign_identity`'s colours, plus the far-end shift). Real player
    tracks stayed within ~90 Lab units on the first four matches; cleaners
    and officials in black or blue were 105-195. Tracks with no colour
    (torso never seen) are kept.
    """
    if not ident.get("colours"):
        return pd.DataFrame(columns=["n", "dist"])
    A = {int(k): np.array(v) for k, v in ident["colours"].items()}
    shift = np.array(ident["far_shift"])
    near = chosen["segment_id"].map(near_of)
    pid = np.where(chosen["side"] == "near", near, 1 - near)
    g = chosen.assign(_pid=pid).groupby(["segment_id", "track_id"])
    stats = g.agg(n=("frame", "size"), pid=("_pid", "first"), side=("side", "first"),
                  **{c: (c, "median") for c in COLOUR_COLUMNS})
    stats = stats[stats["pid"].notna()]
    dist = [_colour_distance(r[COLOUR_COLUMNS].to_numpy(np.float64),
                             A[int(r["pid"])] + (shift if r["side"] == "far" else 0))
            if r[COLOUR_COLUMNS].notna().any() else 0.0
            for _, r in stats.iterrows()]
    stats["dist"] = dist
    return stats.loc[stats["dist"] > max_dist, ["n", "dist"]]


def _smoothed_tracks(sel: pd.DataFrame, cfg_sel: Config) -> Iterator[tuple[tuple, np.ndarray, int, np.ndarray]]:
    """Per (play span, player): row indices, first frame, and the position on
    every frame from there, median-filtered over `smooth_frames` (NaN gaps)."""
    med = int(cfg_sel.smooth_frames)
    for key, idx in sel.groupby(["segment_id", "player_id"]).indices.items():
        g = sel.iloc[idx]
        f0 = int(g["frame"].min())
        grid = g.set_index("frame")[["court_x", "court_y"]].reindex(np.arange(f0, int(g["frame"].max()) + 1))
        if med > 1:
            grid = grid.rolling(med, center=True, min_periods=med // 2 + 1).median()
        yield key, idx, f0, grid.to_numpy()


def _half_window(fps: float, cfg_sel: Config) -> int:
    return max(1, int(round(float(cfg_sel.speed_window_s) * fps / 2)))


def player_speeds(sel: pd.DataFrame, fps: float, cfg_sel: Config) -> np.ndarray:
    """Ground speed (m/s) of each row: centred difference over `speed_window_s`
    of positions median-filtered over `smooth_frames`, within one play span.

    Both are short and centred, so nothing is delayed; NaN where either end
    of the window is missing.
    """
    k = _half_window(fps, cfg_sel)
    speed = np.full(len(sel), np.nan)
    for _, idx, f0, pos in _smoothed_tracks(sel, cfg_sel):
        if len(pos) <= 2 * k:
            continue
        v = np.full(len(pos), np.nan)
        v[k:-k] = np.hypot(*(pos[2 * k:] - pos[:-2 * k]).T) / (2 * k / fps)
        speed[idx] = v[sel["frame"].to_numpy()[idx] - f0]
    return speed


def movement_violations(sel: pd.DataFrame, fps: float, cfg_sel: Config) -> pd.DataFrame:
    """Where a player's track cannot be one person moving: the phase 4 check.

    Two kinds, on positions median-filtered over `smooth_frames`:

    * `step` — moved more than `max_step_m` between consecutive frames. An
      identity swap, a hidden player's feet guessed on someone else, a
      camera bump: the feet land somewhere else at once. Real play peaked at
      1.2 m (a jump's ankles leaving the floor, read at the far end).
    * `sustained` — averaged more than `max_sustained_mps` over
      `sustained_window_s`. A wrong scale, a track drifting onto someone
      else. Real play peaked at 5.5 m/s over 1 s.

    The plan's single cap (4 m/s) does not separate errors from play: elite
    players run back to front at ~5 m/s, and a far-end jump reads 10-15 m/s
    for 0.2 s. Decided 2026-10-04; the share of frames over 4 m/s is still
    reported (`players.meta.json`).
    """
    max_step, vmax = float(cfg_sel.max_step_m), float(cfg_sel.max_sustained_mps)
    k = max(1, int(round(float(cfg_sel.sustained_window_s) * fps / 2)))
    cols = ["kind", "segment_id", "player_id", "start_frame", "end_frame", "value"]
    out = []
    for (sid, pid), _, f0, pos in _smoothed_tracks(sel, cfg_sel):
        step = np.hypot(*np.diff(pos, axis=0).T)
        for i in np.flatnonzero(step > max_step):
            out.append(["step", int(sid), int(pid), f0 + int(i), f0 + int(i) + 2, round(float(step[i]), 2)])
        if len(pos) <= 2 * k:
            continue
        v = np.full(len(pos), np.nan)
        v[k:-k] = np.hypot(*(pos[2 * k:] - pos[:-2 * k]).T) / (2 * k / fps)
        fast = np.flatnonzero(v > vmax)
        if len(fast) == 0:
            continue
        breaks = np.flatnonzero(np.diff(fast) > 1)
        for a, b in zip(fast[np.r_[0, breaks + 1]], fast[np.r_[breaks, len(fast) - 1]]):
            out.append(["sustained", int(sid), int(pid), f0 + int(a), f0 + int(b) + 1,
                        round(float(np.nanmax(v[a:b + 1])), 2)])
    return pd.DataFrame(out, columns=cols)


def _homography_digest(path: Path) -> str:
    segs = json.loads(path.read_text())["segments"]
    return hashlib.sha1(json.dumps([[s["segment_id"], s["image_to_court"]] for s in segs]).encode()).hexdigest()


def rally_mask(frames: np.ndarray, segs: pd.DataFrame) -> np.ndarray:
    """Which of `frames` fall inside a rally of `segments.csv`."""
    rally = segs[(segs["is_play"] == 1) & (segs["rally_id"] >= 0)]
    starts, ends = rally["start_frame"].to_numpy(), rally["end_frame"].to_numpy()
    i = np.searchsorted(starts, frames, side="right") - 1
    return (i >= 0) & (frames < ends[np.clip(i, 0, None)])


def select_players(cfg: Config, match_id: str, force: bool = False) -> pd.DataFrame:
    """The two players out of `poses.parquet`; write `players.parquet`.

    One row per player per frame they were found in: `frame, player_id,
    side, court_x, court_y, keypoints[17][3]` plus the box, feet and speed.
    Checks `movement_violations` inside rallies: each spot is blanked
    (`blank_flagged`), and more than `max_violations` in a match raises
    `PlayerCheckError`. Between points people walk on and off, and nothing
    downstream reads those frames.
    """
    out_dir = cache_dir(cfg, match_id)
    pose_file, hom_file = out_dir / "poses.parquet", out_dir / "homography.json"
    if not pose_file.exists():
        raise SystemExit(f"no {pose_file}; run `bda players` first")
    out_file, meta_file = out_dir / "players.parquet", out_dir / "players.meta.json"
    pose_meta = json.loads((out_dir / "poses.meta.json").read_text())
    cfg_sel = cfg.players
    seg_file = out_dir / "segments.csv"
    inputs = {"poses": pose_meta["inputs"], "homography": _homography_digest(hom_file),
              "rallies": hashlib.sha1(seg_file.read_bytes()).hexdigest(),
              "select": {k: v for k, v in cfg_sel.to_dict().items() if k != "detect"}}
    if out_file.exists() and meta_file.exists() and not force:
        if json.loads(meta_file.read_text()).get("inputs") == inputs:
            print(f"players: reusing {out_file}")
            return load_players(out_file)

    fps = float(pose_meta["fps"])
    views = pd.read_csv(out_dir / "view_segments.csv")
    homs = load_homographies(hom_file)
    poses, kp = load_poses(pose_file)
    height = json.loads(hom_file.read_text())["image_size"][1]
    poses["_row"] = np.arange(len(poses))
    placed = place_feet(poses, kp, homs, height, float(cfg_sel.ankle_conf))
    sel = choose_players(placed, cfg_sel)
    near_id, ident = assign_identity(sel, views, fps, cfg_sel)
    strangers = foreign_tracks(sel, near_id, ident, float(cfg_sel.max_track_colour))
    if len(strangers):  # someone else stood in for a player: choose again without them
        drop = placed.set_index(["segment_id", "track_id"]).index.isin(strangers.index)
        sel = choose_players(placed[~drop], cfg_sel)
        near_id, ident = assign_identity(sel, views, fps, cfg_sel)
    ident["dropped_tracks"] = [{"segment_id": int(s), "track_id": int(t), "frames": int(r.n),
                                "colour_distance": round(float(r.dist), 1)} for (s, t), r in strangers.iterrows()]
    near = sel["segment_id"].map(near_id)
    sel = sel[near.notna()].copy()
    sel["player_id"] = np.where(sel["side"] == "near", near[near.notna()], 1 - near[near.notna()]).astype(int)
    sel = sel.sort_values(["frame", "player_id"]).reset_index(drop=True)
    sel["speed"] = player_speeds(sel, fps, cfg_sel)

    in_rally = rally_mask(sel["frame"].to_numpy(), pd.read_csv(seg_file))
    bad = movement_violations(sel[in_rally], fps, cfg_sel)
    flagged = blank_flagged(sel, bad, fps, float(cfg_sel.blank_s))
    if flagged.any():
        sel["speed"] = player_speeds(sel, fps, cfg_sel)  # no speed across a blanked stretch

    table = _table({c: sel[c].to_numpy() for c in PLAYER_COLUMNS if c != "keypoints"},
                   kp[sel["_row"].to_numpy()], PLAYER_COLUMNS)
    tmp = out_file.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out_file)
    play = views[views["is_play"] == 1]
    play_frames = int((play["end_frame"] - play["start_frame"]).sum())
    found = sel.groupby("side")["frame"].nunique().reindex(list(SIDES), fill_value=0)
    rally = sel[in_rally]
    report = float(cfg_sel.report_speed_mps)
    meta = {
        "inputs": inputs,
        "found": {s: round(float(found[s]) / play_frames, 4) for s in SIDES},
        "foot_src_rally": {s: g["foot_src"].value_counts(normalize=True).round(4).to_dict()
                           for s, g in rally.groupby("side")},
        "identity": ident,
        "speed_mps_rally": {s: {"p50": round(float(g["speed"].median()), 2),
                                "p99": round(float(g["speed"].quantile(0.99)), 2),
                                f"over_{report:g}": round(float((g["speed"] > report).mean()), 4)}
                            for s, g in rally.groupby("side")},
        "violations": bad.to_dict("records"),
        "flagged_frames": int(flagged.sum()),
    }
    meta_file.write_text(json.dumps(meta, indent=1))
    over = max(v[f"over_{report:g}"] for v in meta["speed_mps_rally"].values()) if len(rally) else 0.0
    print(f"players: wrote {out_file} — near found in {meta['found']['near']:.1%} of play frames, "
          f"far {meta['found']['far']:.1%}; ends changed {len(ident['switches'])}x "
          f"(colour separation {ident['separation']}); up to {over:.1%} of rally frames over {report:g} m/s; "
          f"{len(bad)} movement violation(s) blanked ({int(flagged.sum())} rows)")
    if len(ident["switches"]) > int(cfg_sel.max_end_changes):
        raise PlayerCheckError(f"{match_id}: players changed ends {len(ident['switches'])} times "
                               f"(at most {cfg_sel.max_end_changes} in a match): identities are confused; "
                               f"see {meta_file}")
    if len(bad) > int(cfg_sel.max_violations):
        raise PlayerCheckError(f"{match_id}: {len(bad)} place(s) inside rallies where a player's track "
                               f"cannot be one person moving (step over {cfg_sel.max_step_m} m between frames, "
                               f"or over {cfg_sel.max_sustained_mps} m/s for {cfg_sel.sustained_window_s} s); "
                               f"more than {cfg_sel.max_violations} means something systematic — identity "
                               f"swaps, hidden feet, a bad homography, or rallies that are not rallies:\n"
                               f"{bad.head(10).to_string()}")
    return load_players(out_file)


def blank_flagged(sel: pd.DataFrame, bad: pd.DataFrame, fps: float, pad_s: float) -> np.ndarray:
    """Blank the court position around each movement violation, in place; return which rows.

    A flagged spot (a jump read as distance, a hidden foot guessed wrong,
    a corrupted frame) is a position nobody downstream should use: those
    rows keep their pose but get `foot_src = flagged` and no `court_x/y`,
    `pad_s` either side. The spots stay listed in `players.meta.json`.
    Decided 2026-10-04, after looking at all 23 on 32 matches.
    """
    hit = np.zeros(len(sel), bool)
    if bad.empty:
        return hit
    pad = int(round(pad_s * fps))
    f = sel["frame"].to_numpy()
    key = sel["segment_id"].to_numpy(), sel["player_id"].to_numpy()
    for v in bad.itertuples():
        hit |= ((key[0] == v.segment_id) & (key[1] == v.player_id)
                & (f >= v.start_frame - pad) & (f < v.end_frame + pad))
    sel.loc[hit, ["court_x", "court_y"]] = np.nan
    sel.loc[hit, "foot_src"] = "flagged"
    return hit


def load_players(path: str | Path) -> pd.DataFrame:
    """`players.parquet`; `keypoints_array(df)` gives the `(N, 17, 3)` keypoints."""
    return pq.read_table(path).to_pandas()


# --- Overlay, for checking by eye ----------------------------------------------------------

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
PLAYER_BGR = [(0, 200, 255), (255, 160, 0)]  # player 0 amber, player 1 blue


def _minimap(players_now: pd.DataFrame, trail: pd.DataFrame, scale: float) -> np.ndarray:
    """Top-down court, far end up, with each player's feet and last second of track."""
    from court import COURT_LENGTH, DOUBLES_WIDTH, painted_segments

    pad = 1.5
    w, h = int((DOUBLES_WIDTH + 2 * pad) * scale), int((COURT_LENGTH + 2 * pad) * scale)
    img = np.full((h, w, 3), (50, 90, 50), np.uint8)

    def uv(x, y):
        return int(round(w / 2 + x * scale)), int(round(h / 2 - y * scale))

    for _, a, b in painted_segments():
        cv2.line(img, uv(*a), uv(*b), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.line(img, uv(-DOUBLES_WIDTH / 2 - 0.3, 0), uv(DOUBLES_WIDTH / 2 + 0.3, 0), (200, 200, 255), 2)
    for pid, g in trail.groupby("player_id"):
        pts = [uv(x, y) for x, y in g[["court_x", "court_y"]].to_numpy() if np.isfinite(x)]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, PLAYER_BGR[int(pid)], 1, cv2.LINE_AA)
    for r in players_now.itertuples():
        if np.isfinite(r.court_x):
            cv2.circle(img, uv(r.court_x, r.court_y), 5, PLAYER_BGR[int(r.player_id)], -1, cv2.LINE_AA)
    return img


def draw_players(frame: np.ndarray, rows: pd.DataFrame, kp: np.ndarray, trail: pd.DataFrame) -> np.ndarray:
    """Boxes, skeletons, ids and speeds on the frame; the minimap top right."""
    out = frame.copy()
    t = max(1, frame.shape[0] // 540)
    for r, k in zip(rows.itertuples(), kp):
        c = PLAYER_BGR[int(r.player_id)]
        cv2.rectangle(out, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, t)
        for i, j in SKELETON:
            if k[i, 2] >= 0.3 and k[j, 2] >= 0.3:
                cv2.line(out, (int(k[i, 0]), int(k[i, 1])), (int(k[j, 0]), int(k[j, 1])), c, t, cv2.LINE_AA)
        cv2.drawMarker(out, (int(r.foot_x), int(r.foot_y)), (0, 0, 255), cv2.MARKER_CROSS, 12 * t, t)
        speed = "" if not np.isfinite(r.speed) else f" {r.speed:.1f} m/s"
        label = f"P{int(r.player_id)} {r.side}{speed}"
        cv2.putText(out, label, (int(r.x1), int(r.y1) - 6 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, c, t, cv2.LINE_AA)
    mm = _minimap(rows, trail, scale=14 * t)
    out[10 * t:10 * t + mm.shape[0], -mm.shape[1] - 10 * t:-10 * t] = mm
    return out


def write_players_overlay(cfg: Config, match_id: str, video_path: str | Path, out_path: str | Path,
                          start_frame: int, end_frame: int) -> Path:
    """Render `players.parquet` onto `[start_frame, end_frame)` of the video."""
    out_dir = cache_dir(cfg, match_id)
    players = load_players(out_dir / "players.parquet")
    players = players[(players.frame >= start_frame - 60) & (players.frame < end_frame)]
    kp = keypoints_array(players)
    info = probe_video(video_path)
    out_path = Path(out_path)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {out_path}")
    by_frame = players.groupby("frame").indices
    trail_n = int(round(info.fps))
    try:
        for idx, frame in iter_frames(video_path, start_frame, end_frame):
            rows = by_frame.get(idx)
            if rows is None:
                writer.write(frame)
                continue
            trail = players[(players.frame > idx - trail_n) & (players.frame <= idx)]
            writer.write(draw_players(frame, players.iloc[rows], kp[rows], trail))
    finally:
        writer.release()
    return out_path


# --- Smoothing, for readers --------------------------------------------------------------


def savgol_coeffs(window: int, order: int) -> np.ndarray:
    """Centred Savitzky-Golay smoothing weights (value at the centre)."""
    if window % 2 == 0 or window <= order:
        raise ValueError(f"window must be odd and > order, got {window}, {order}")
    m = window // 2
    A = np.vander(np.arange(-m, m + 1), order + 1, increasing=True)
    return np.linalg.pinv(A)[0]


def smooth_keypoints(kp: np.ndarray, frames: np.ndarray, window: int = 5, order: int = 2,
                     min_conf: float = 0.3, max_gap: int = 2) -> np.ndarray:
    """One person's keypoints, smoothed in time: `(N, 17, 2)` image x, y.

    `kp` is `(N, 17, 3)` for consecutive-ish `frames` (sorted, one track).
    A centred Savitzky-Golay fit: a local quadratic, so a wrist's speed peak
    keeps its frame and most of its height, unlike a moving average — the
    smoothing-window bug the plan warns about. Keypoints under `min_conf` are
    dropped; gaps up to `max_gap` frames are bridged linearly for the fit and
    left NaN after it. Where the window does not fit, the raw point is kept.
    """
    if len(kp) == 0:
        return np.zeros((0, kp.shape[1], 2))
    frames = np.asarray(frames)
    if np.any(np.diff(frames) <= 0):
        raise ValueError("frames must be strictly increasing (one track)")
    f0 = frames[0]
    n = int(frames[-1] - f0 + 1)
    grid = np.full((n, kp.shape[1], 2), np.nan)
    pts = kp[:, :, :2].astype(np.float64)
    pts[kp[:, :, 2] < min_conf] = np.nan
    grid[frames - f0] = pts
    raw = grid.copy()
    # Bridge short gaps, per series.
    flat = grid.reshape(n, -1)
    t = np.arange(n)
    for j in range(flat.shape[1]):
        ok = ~np.isnan(flat[:, j])
        if ok.sum() < 2:
            continue
        filled = np.interp(t, t[ok], flat[ok, j])
        last = np.maximum.accumulate(np.where(ok, t, -n))  # previous good frame
        nxt = np.minimum.accumulate(np.where(ok, t, 2 * n)[::-1])[::-1]  # next good frame
        bridge = ~ok & (last >= 0) & (nxt < n) & (nxt - last - 1 <= max_gap)
        flat[bridge, j] = filled[bridge]
    c = savgol_coeffs(window, order)
    m = window // 2
    out = raw.copy()
    if n >= window:
        win = np.lib.stride_tricks.sliding_window_view(grid, window, axis=0)  # (n - 2m, 17, 2, window)
        out[m:n - m] = win @ c
    out[np.isnan(out)] = raw[np.isnan(out)]  # window held a gap too long to bridge: keep the raw point
    out[np.isnan(raw)] = np.nan  # never invent a point
    return out[frames - f0]
