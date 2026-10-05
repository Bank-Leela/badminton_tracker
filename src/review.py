"""The phase 5 acceptance check: look at random shots against the video.

`write_shot_review` picks shots from a match's `shots.csv` and writes, per
shot, one image: five frames around the detected contact (the contact frame
outlined), the frame where the shot ends with its landing point drawn on the
court, and a top-down map of hitter, opponent and landing. Plus
`review.csv` with empty `contact_ok`, `landing_ok`, `note` columns to fill in.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import Config, cache_dir
from court import COURT_LENGTH, DOUBLES_WIDTH, load_homographies, painted_segments, project
from video import sample_frames


def _crop(img: np.ndarray, cx: float, cy: float, w: int = 480, h: int = 360) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 0, H - h))
    return img[y0:y0 + h, x0:x0 + w].copy()


def _map(points: dict, scale: float = 22.0) -> np.ndarray:
    pad = 1.5
    w, h = int((DOUBLES_WIDTH + 2 * pad) * scale), int((COURT_LENGTH + 2 * pad) * scale)
    img = np.full((h, w, 3), (50, 90, 50), np.uint8)
    uv = lambda x, y: (int(w / 2 + x * scale), int(h / 2 - y * scale))
    for _, a, b in painted_segments():
        cv2.line(img, uv(*a), uv(*b), (230, 230, 230), 1)
    cv2.line(img, uv(-3.4, 0), uv(3.4, 0), (200, 200, 255), 2)
    for label, (xy, col) in points.items():
        if xy is not None and np.isfinite(xy).all():
            cv2.circle(img, uv(*xy), 6, col, -1)
            cv2.putText(img, label, (uv(*xy)[0] + 8, uv(*xy)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return img


def write_shot_review(cfg: Config, match_id: str, video_path: str | Path, n: int = 20, seed: int = 0) -> Path:
    out_dir = cache_dir(cfg, match_id)
    review_dir = out_dir / "review"
    review_dir.mkdir(exist_ok=True)
    shots = pd.read_csv(out_dir / "shots.csv")
    contacts = pd.read_csv(out_dir / "contacts.csv")
    homs = {int(r.segment_id): r.H for r in load_homographies(out_dir / "homography.json").itertuples() if r.H is not None}
    from players import load_players

    players = load_players(out_dir / "players.parquet")
    seg_of = players.drop_duplicates("frame").set_index("frame")["segment_id"]
    pick = shots.sample(min(n, len(shots)), random_state=seed).sort_values("frame")
    rows = []
    for k, s in enumerate(pick.itertuples(), start=1):
        rally = contacts[contacts.rally_id == s.rally_id].sort_values("frame")
        e = rally[rally.frame == s.frame].iloc[0]
        nxt = rally[rally.frame > s.frame].head(1)
        frames = [int(s.frame) + d for d in (-4, -2, 0, 2, 4)]
        end_frame = int(nxt.frame.iloc[0]) if len(nxt) else int(s.frame) + 30
        imgs = sample_frames(video_path, sorted(set(frames + [end_frame])))
        by = dict(zip(sorted(set(frames + [end_frame])), imgs))
        tiles = []
        for f in frames:
            t = _crop(by[f], e.x, e.y)
            col = (0, 0, 255) if f == s.frame else (200, 200, 200)
            cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), col, 4 if f == s.frame else 1)
            cv2.putText(t, f"{f}{'  CONTACT' if f == s.frame else ''}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            tiles.append(cv2.resize(t, (320, 240)))
        top = np.hstack(tiles)
        # The end of the shot, landing point drawn on the court.
        end_img = by[end_frame].copy()
        seg = seg_of.get(int(s.frame))
        land_xy = None
        if np.isfinite(s.landing_depth) and seg is not None and int(seg) in homs:
            recv = "far" if s.hitter_side == "near" else "near"
            x = s.landing_lateral if recv == "near" else -s.landing_lateral
            y = -s.landing_depth if recv == "near" else s.landing_depth
            land_xy = np.array([x, y])
            p = project(np.linalg.inv(homs[int(seg)]), land_xy[None])[0]
            cv2.drawMarker(end_img, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3)
            cx, cy = p
        else:
            cx, cy = (float(nxt.x.iloc[0]), float(nxt.y.iloc[0])) if len(nxt) else (960, 540)
        kind = nxt.kind.iloc[0] if len(nxt) else "none"
        bottom_left = cv2.resize(_crop(end_img, cx, cy, 640, 360), (640, 360))
        cv2.putText(bottom_left, f"{end_frame}: next = {kind}; landing ({s.landing_src})", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        hitter = None if not np.isfinite(s.hitter_depth) else np.array(
            [s.hitter_lateral if s.hitter_side == "near" else -s.hitter_lateral,
             -s.hitter_depth if s.hitter_side == "near" else s.hitter_depth])
        mm = _map({"hitter": (hitter, (0, 200, 255)), "landing": (land_xy, (0, 0, 255))})
        mm = cv2.resize(mm, (int(mm.shape[1] * 360 / mm.shape[0]), 360))
        info = np.full((360, top.shape[1] - 640 - mm.shape[1], 3), 30, np.uint8)
        lines = [f"shot {k}: rally {s.rally_id} #{s.shot_index} {s.hitter_side} (player {s.hitter_id})",
                 f"contact frame {s.frame}  reach {s.contact_reach}  gap {s.contact_gap}",
                 f"flight {s.flight_time:.2f}s  speed {s.shuttle_speed * 3.6:.0f} km/h" if np.isfinite(s.shuttle_speed)
                 else f"flight {s.flight_time:.2f}s  speed -",
                 f"net clearance {s.net_clearance:.2f} m" if np.isfinite(s.net_clearance) else "net clearance -",
                 f"landing depth {s.landing_depth:.1f} lat {s.landing_lateral:.1f}" if np.isfinite(s.landing_depth) else "landing -",
                 f"from lines {s.dist_from_lines:.2f} m" if np.isfinite(s.dist_from_lines) else ""]
        for i, line in enumerate(lines):
            cv2.putText(info, line, (8, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        sheet = np.vstack([top, np.hstack([bottom_left, mm, info])])
        name = f"shot_{k:02d}_r{s.rally_id}_f{s.frame}.jpg"
        cv2.imwrite(str(review_dir / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append({"shot": k, "image": name, "rally_id": s.rally_id, "frame": s.frame, "hitter_side": s.hitter_side,
                     "next": kind, "contact_ok": "", "landing_ok": "", "note": ""})
    out = review_dir / "review.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"review: wrote {len(rows)} shot sheets and {out}")
    return out
