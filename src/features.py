"""Phase 5, part 4: the per-shot feature table, `shots.csv`.

One row per hit (`contacts.csv`), with the features agreed in
`docs/features.md`. Positions are court metres seen from that player's end:
`depth` = distance from the net, `lateral` = across the court as they face
the net (+ = their right). Nothing after this phase reads video.

The plan's invariants are checked on every value: shuttle speed under
500 km/h, every court position within the court plus 2 m. A value breaking
one is a measurement failure (a bad flight fit, a misplaced foot): it is
blanked and counted, and too many blanked in a match raises
`FeatureCheckError` — the same rule as phase 4's movement check.

The flight columns (`landing_*` ... `opponent_toward`) are measured up to the
next event — the reply or the floor — so they say whether the shot came
back. The `early_*` columns measure the same things from the shot's first
`features.early_frames` frames only (what the labelling clip shows), the
same way for every shot: phase 7's inputs (`early_flight_features`).
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from camera import load_camera
from config import Config, cache_dir
from contacts import clean_track
from court import COURT_LENGTH, DOUBLES_WIDTH, SINGLES_WIDTH, load_homographies, project
from flight import Anchor, fit_early_flight, fit_flight
from video import FrameClock, repeated_frames
from players import KEYPOINTS, keypoints_array, load_players, to_court

HALF_L, HALF_SW = COURT_LENGTH / 2, SINGLES_WIDTH / 2
NOSE = KEYPOINTS.index("nose")
L_SH, R_SH = KEYPOINTS.index("left_shoulder"), KEYPOINTS.index("right_shoulder")
L_HIP, R_HIP = KEYPOINTS.index("left_hip"), KEYPOINTS.index("right_hip")
L_ANK, R_ANK = KEYPOINTS.index("left_ankle"), KEYPOINTS.index("right_ankle")

SHOT_COLUMNS = [
    "match_id", "rally_id", "shot_index", "frame", "time_s", "hitter_id", "hitter_side", "is_serve",
    "time_since_prev_contact",
    "hitter_depth", "hitter_lateral", "hitter_speed", "contact_height", "hitter_lean", "stance_width",
    "hitter_recovery_time",
    "landing_depth", "landing_lateral", "landing_src", "ended", "dist_from_lines", "flight_time", "shot_length",
    "avg_speed", "cross_court", "net_clearance", "shuttle_speed",
    "opponent_id", "opponent_depth", "opponent_lateral", "opponent_dist", "opponent_speed", "opponent_toward",
    "fit_rms_px", "fit_n_obs", "contact_reach", "contact_gap",
]
# Phase 7's inputs: the same shot measured from its early flight only — what
# the labelling clip shows — flown on to the floor (`early_flight_features`).
EARLY_COLUMNS = [
    "early_landing_depth", "early_landing_lateral", "early_dist_from_lines", "early_flight_time",
    "early_shot_length", "early_avg_speed", "early_cross_court", "early_net_clearance", "early_shuttle_speed",
    "early_opponent_dist", "early_opponent_toward",
    "early_fit_rms_px", "early_fit_n_obs",
]
SHOT_COLUMNS += EARLY_COLUMNS

# Fewest track points in the early window for an early fit (seven unknowns;
# fewer leave the direction loose). Would be `features.early_min_obs`.
EARLY_MIN_OBS = 4
# The early fit's track is cleaned (`contacts.clean_track`, which judges a
# point by its neighbours) on the clip's own frames: from this many frames
# before the contact to the window's last — never with a point after it.
# Would be `features.early_clean_back_frames`.
EARLY_CLEAN_BACK = 12
# Share of a match's shots whose early-flight values may break an invariant
# before it counts as systematic (on the 32 matches: 3.6-17%, the far
# player's extrapolated landings mostly). Would be `features.max_early_blanked_frac`.
EARLY_MAX_BLANKED_FRAC = 0.30


class FeatureCheckError(AssertionError):
    """Too many shots broke the plan's invariants (raised explicitly, like `court.CourtCheckError`)."""


def own_frame(x: float, y: float, side: str) -> tuple[float, float]:
    """Court (x, y) as seen from `side`'s end: (depth from the net, lateral, + = their right).

    The near player faces away from the camera, so their right is +x; the
    far player faces it, so theirs is −x.
    """
    return (abs(y), x) if side == "near" else (abs(y), -x)


def net_height(x: float) -> float:
    """Net top height at court x: 1.55 m at the posts, 1.524 m at the centre."""
    return 1.524 + (1.55 - 1.524) * min(1.0, (x / 3.05) ** 2)


def dist_from_singles_lines(x: float, y: float) -> float:
    """Distance to the nearest singles boundary (sidelines, baseline), + inside, − out."""
    inside = min(HALF_SW - abs(x), HALF_L - abs(y))
    if inside >= 0:
        return inside
    dx, dy = max(0.0, abs(x) - HALF_SW), max(0.0, abs(y) - HALF_L)
    return -float(np.hypot(dx, dy))


def dist_from_receiver_court(x: float, depth: float) -> float:
    """`dist_from_singles_lines` for a landing at signed `depth` past the net (− = short, on the hitter's own half).

    Past the net it is the same (the sidelines and the baseline); short of it,
    minus the distance to the receiver's court.
    """
    if depth >= 0:
        return dist_from_singles_lines(x, depth)
    return -float(np.hypot(max(0.0, abs(x) - HALF_SW), depth))


def early_window(frame: int, next_frame: int | None, early_frames: int, margin: int) -> int:
    """Last frame of a shot's early window: `early_frames` after the contact, as the labelling clip shows.

    In a fast exchange the clip stops `margin` frames short of the next event
    (`labeler.clip_range`), so the window does too — the reply's own flight is
    never fitted as this shot. Only the next event's frame matters, never
    what it was (a reply or the landing).
    """
    last = frame + early_frames
    if next_frame is not None:
        last = min(last, max(frame, next_frame - margin - 1))
    return last


def clip_track(f0: int, last: int, tf: np.ndarray, tx: np.ndarray, ty: np.ndarray, top_px: float,
               jump_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shuttle track the labelling clip shows around a contact at `f0`, cleaned on its own.

    `tf, tx, ty`: the match's visible shuttle points (frame-sorted). Only
    frames `f0 - EARLY_CLEAN_BACK` .. `last` go in, so the one-frame
    false-detection rule (`clean_track`) never looks past the window: the
    track after it — the reply, or the shuttle on the floor — can't drop or
    keep a point inside it.
    """
    i0, i1 = np.searchsorted(tf, f0 - EARLY_CLEAN_BACK), np.searchsorted(tf, last, side="right")
    return clean_track(tf[i0:i1], tx[i0:i1], ty[i0:i1], top_px, jump_px)


def start_pixel(f0: int, cf: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray | None:
    """The early fit's start pixel, from what the clip shows: the shuttle at the contact frame, else the frame
    before; None when it is seen at neither (the fit then places the hit from the feet and the launch moment).

    Not `contacts.csv`'s pixel: that is where the incoming and the outgoing
    curves meet, and the outgoing one is fitted over the whole flight, up to
    the reply or the landing.
    """
    for f in (f0, f0 - 1):
        i = np.flatnonzero(cf == f)
        if len(i):
            return np.array([cx[i[0]], cy[i[0]]], float)
    return None


def early_flight_features(contact: dict, next_frame: int | None, tf: np.ndarray, tx: np.ndarray, ty: np.ndarray,
                          clock: FrameClock, cam, hitter_xy, opp_xy, opp_v, receiver_side: str, cfg: Config,
                          scale: float) -> tuple[dict, dict]:
    """A shot's `early_*` columns, and which invariants they broke (`{"unfit", "speed", "time", "position"}` flags).

    Everything comes from what the labelling clip shows — the contact's
    frame and side, the hitter's feet, the shuttle track up to the window's
    end (`tf, tx, ty`: the match's visible points, uncleaned; `clip_track`) —
    and the receiver's position and velocity at the contact. The next event
    enters only through its frame, to end the window in a fast exchange;
    whether the shot was returned, and where or when, never does. Units and
    frames as the namesakes: metres, seconds, m/s; the landing point in the
    receiver's own frame — `early_landing_depth` signed, negative = it comes
    down short of the net, on the hitter's side (`early_net_clearance` is
    then NaN: it never gets there).

    A fit breaking the speed invariant — launched at `max_shuttle_kmh` or
    more, or stopped at the solver's cap (`flight.V_CAP_MPS`), or averaging
    that much to its landing — or reaching the floor at or before the
    contact frame is broken, not a fast shot: none of its values are kept
    (only `early_fit_rms_px`, `early_fit_n_obs`). A landing outside the court
    + `court_margin_m` blanks the landing's values only.
    """
    out, broke = {}, {"unfit": False, "speed": False, "time": False, "position": False}
    if cam is None or hitter_xy is None:
        return out, broke
    cfg_f = cfg.features
    f0 = int(contact["frame"])
    last = early_window(f0, next_frame, int(cfg_f.early_frames), int(cfg.labeler.next_event_margin))
    cf, cx, cy = clip_track(f0, last, tf, tx, ty, float(cfg.contacts.top_px) * scale,
                            float(cfg.contacts.jump_px) * scale)
    sel = (cf > f0) & (cf <= last)
    if sel.sum() < EARLY_MIN_OBS:
        return out, broke
    ts, uv = clock.seconds(f0, cf[sel]), np.c_[cx[sel], cy[sel]]
    fit = fit_early_flight(ts, uv, Anchor(0.0, start_pixel(f0, cf, cx, cy), hitter_xy), cam, cfg.flight,
                           contact["side"])
    out["early_fit_rms_px"], out["early_fit_n_obs"] = fit["rms_px"], fit["n_obs"]
    if not np.isfinite(fit["rms_px"]) or fit["rms_px"] > float(cfg_f.max_fit_rms_px) * scale:
        broke["unfit"] = True
        return out, broke
    max_mps = float(cfg_f.max_shuttle_kmh) / 3.6
    if fit["capped"] or fit["speed_mps"] >= max_mps:
        broke["speed"] = True
        return out, broke
    land, t_land = fit["land"], fit["t_land"]
    landed = land is not None and np.isfinite(land).all() and t_land is not None
    if landed:
        d = land - fit["P0"][:2]
        if t_land <= 0:  # on the floor at the contact frame, with the clip still to come: an underground fit
            broke["time"] = True
            return out, broke
        if float(np.linalg.norm(d)) / t_land >= max_mps:
            broke["speed"] = True
            return out, broke
    margin = float(cfg_f.court_margin_m)
    if fit["early_obs"] >= int(cfg.flight.min_early_obs):
        out["early_shuttle_speed"] = fit["speed_mps"]
    if fit["net_z"] is not None and abs(fit["net_x"]) <= DOUBLES_WIDTH / 2 + margin:
        out["early_net_clearance"] = fit["net_z"] - net_height(fit["net_x"])
    if not landed:
        return out, broke
    if not (abs(land[0]) <= DOUBLES_WIDTH / 2 + margin and abs(land[1]) <= HALF_L + margin):
        broke["position"] = True
        return out, broke
    depth = float(land[1]) if receiver_side == "far" else -float(land[1])
    lateral = own_frame(land[0], land[1], receiver_side)[1]
    out.update({"early_landing_depth": depth, "early_landing_lateral": lateral,
                "early_dist_from_lines": dist_from_receiver_court(lateral, depth),
                "early_flight_time": t_land, "early_shot_length": float(np.linalg.norm(d)),
                "early_cross_court": float(np.degrees(np.arctan2(abs(d[0]), abs(d[1])))),
                "early_avg_speed": float(np.linalg.norm(d)) / t_land})
    if opp_xy is not None:
        to_land = land - opp_xy
        out["early_opponent_dist"] = float(np.linalg.norm(to_land))
        if opp_v is not None and out["early_opponent_dist"] > 1e-6:
            out["early_opponent_toward"] = float(opp_v @ to_land / out["early_opponent_dist"])
    return out, broke


class Track:
    """One player's per-frame state (by side), looked up by frame."""

    def __init__(self, players: pd.DataFrame, kp: np.ndarray, homs: dict, fps: float):
        self.fps = fps
        self.rows = {}
        for (side), g in players.groupby("side"):
            g = g.sort_values("frame")
            self.rows[side] = (g["frame"].to_numpy(), g.index.to_numpy())
        self.p, self.kp, self.homs = players, kp, homs

    def at(self, side: str, frame: float, max_lag: int = 3) -> int | None:
        """Row index of `side`'s player nearest `frame` within `max_lag`."""
        if side not in self.rows:
            return None
        f, idx = self.rows[side]
        i = int(np.clip(np.searchsorted(f, frame), 0, len(f) - 1))
        c = min((c for c in (i - 1, i) if 0 <= c < len(f)), key=lambda c: abs(f[c] - frame))
        return int(idx[c]) if abs(f[c] - frame) <= max_lag else None

    def velocity(self, side: str, frame: float, half_s: float = 0.1) -> np.ndarray | None:
        """Court velocity (m/s) from positions `half_s` either side."""
        k = int(round(half_s * self.fps))
        a, b = self.at(side, frame - k, 2), self.at(side, frame + k, 2)
        if a is None or b is None:
            return None
        pa = self.p.loc[a, ["court_x", "court_y"]].to_numpy(float)
        pb = self.p.loc[b, ["court_x", "court_y"]].to_numpy(float)
        dt = (self.p.at[b, "frame"] - self.p.at[a, "frame"]) / self.fps
        if dt <= 0 or not np.isfinite(pa).all() or not np.isfinite(pb).all():
            return None
        return (pb - pa) / dt

    def pose_features(self, row: int, shuttle_y: float) -> dict:
        """Contact height (shuttle against ankles..nose, image), torso lean, stance width."""
        k = self.kp[row]
        out = {"contact_height": np.nan, "hitter_lean": np.nan, "stance_width": np.nan}
        ank_y = k[[L_ANK, R_ANK], 1].mean() if (k[[L_ANK, R_ANK], 2] >= 0.5).all() else self.p.at[row, "y2"]
        if k[NOSE, 2] >= 0.5 and ank_y - k[NOSE, 1] > 5:
            out["contact_height"] = float((ank_y - shuttle_y) / (ank_y - k[NOSE, 1]))
        if (k[[L_SH, R_SH, L_HIP, R_HIP], 2] >= 0.5).all():
            sh, hip = k[[L_SH, R_SH], :2].mean(0), k[[L_HIP, R_HIP], :2].mean(0)
            v = sh - hip
            out["hitter_lean"] = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
        H = self.homs.get(int(self.p.at[row, "segment_id"]))
        if H is not None and (k[[L_ANK, R_ANK], 2] >= 0.5).all():
            a = to_court(H, k[[L_ANK, R_ANK], :2].astype(float))
            if np.isfinite(a).all():
                out["stance_width"] = float(np.linalg.norm(a[0] - a[1]))
        return out


def player_bases(contacts: pd.DataFrame, track: Track) -> dict:
    """Each player's base: their median (depth, lateral) when the opponent hits."""
    pts = {}
    for c in contacts.itertuples():
        other = "far" if c.side == "near" else "near"
        r = track.at(other, c.frame)
        if r is None:
            continue
        x, y = track.p.at[r, "court_x"], track.p.at[r, "court_y"]
        if np.isfinite(x) and np.isfinite(y):
            pts.setdefault(int(track.p.at[r, "player_id"]), []).append(own_frame(x, y, other))
    return {pid: np.median(np.array(v), axis=0) for pid, v in pts.items()}


def recovery_time(track: Track, side: str, pid: int, start: int, stop: int, base, radius: float) -> float:
    """Seconds from `start` until `side`'s player is within `radius` of `base`, before `stop`; NaN if not."""
    if base is None:
        return np.nan
    f, idx = track.rows.get(side, (np.array([]), np.array([])))
    sel = idx[(f > start) & (f < stop)]
    if len(sel) == 0:
        return np.nan
    g = track.p.loc[sel]
    g = g[g["player_id"] == pid]
    for row in g.itertuples():
        if np.isfinite(row.court_x) and np.isfinite(row.court_y):
            d, l = own_frame(row.court_x, row.court_y, side)
            if np.hypot(d - base[0], l - base[1]) <= radius:
                return (row.frame - start) / track.fps
    return np.nan


def frame_clock(out_dir, contacts: pd.DataFrame, fps: float) -> FrameClock:
    """The match's `FrameClock`; repeated frames over the rallies found once and cached in `repeats.npz`."""
    spans = contacts.groupby("rally_id")["frame"].agg(["min", "max"])
    spans = [(int(a) - 5, int(b) + 30) for a, b in zip(spans["min"], spans["max"])]
    key = np.array([[a, b] for a, b in spans], dtype=np.int64)
    cache = out_dir / "repeats.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["spans"], key):
            return FrameClock(fps, z["repeats"], int(z["n_frames"]))
    video = json.loads((out_dir / "poses.meta.json").read_text())["video"]
    repeats = repeated_frames(video, spans)
    n_frames = int(sum(b - a for a, b in spans))
    np.savez(cache, spans=key, repeats=repeats, n_frames=n_frames)
    return FrameClock(fps, repeats, n_frames)


def shot_features(cfg: Config, match_id: str, force: bool = False) -> pd.DataFrame:
    """Build `shots.csv` for a match from contacts, players, shuttle track and camera."""
    out_dir = cache_dir(cfg, match_id)
    out_file, meta_file = out_dir / "shots.csv", out_dir / "shots.meta.json"
    for name in ("contacts.csv", "players.parquet", "shuttle.csv", "camera.json", "homography.json"):
        if not (out_dir / name).exists():
            raise SystemExit(f"no {out_dir / name}; run the earlier stages first")
    cfg_f = cfg.features
    started = time.perf_counter()
    fps = float(json.loads((out_dir / "poses.meta.json").read_text())["fps"])
    height = json.loads((out_dir / "homography.json").read_text())["image_size"][1]
    scale = height / 1080
    homs = {int(r.segment_id): r.H for r in load_homographies(out_dir / "homography.json").itertuples() if r.H is not None}
    cam = load_camera(out_dir / "camera.json")
    contacts = pd.read_csv(out_dir / "contacts.csv")
    players = load_players(out_dir / "players.parquet")
    kp = keypoints_array(players)
    track = Track(players, kp, homs, fps)
    shuttle = pd.read_csv(out_dir / "shuttle.csv")
    shuttle = shuttle[shuttle["visible"] == 1]  # in frame order, as phase 2 writes it
    vf, vx, vy = shuttle["frame"].to_numpy(float), shuttle["x"].to_numpy(float), shuttle["y"].to_numpy(float)
    sf, sx, sy = clean_track(vf, vx, vy, float(cfg.contacts.top_px) * scale, float(cfg.contacts.jump_px) * scale)
    clock = frame_clock(out_dir, contacts, fps)
    seg_of = players.drop_duplicates("frame").set_index("frame")["segment_id"]
    hits = contacts[contacts["kind"] != "landing"]
    bases = player_bases(hits, track)
    max_kmh, margin = float(cfg_f.max_shuttle_kmh), float(cfg_f.court_margin_m)
    rows, unfit, long_flights = [], 0, 0
    blanked = {"speed": 0, "position": 0, "early_speed": 0, "early_time": 0, "early_position": 0}
    early_unfit = 0

    def in_bounds(x, y):
        return abs(x) <= DOUBLES_WIDTH / 2 + margin and abs(y) <= HALF_L + margin

    for rid, rally in contacts.groupby("rally_id"):
        ev = rally.sort_values("frame").to_dict("records")
        rally_hits = [e for e in ev if e["kind"] != "landing"]
        for k, e in enumerate(rally_hits):
            side, other = e["side"], ("far" if e["side"] == "near" else "near")
            nxt = next((n for n in ev if n["frame"] > e["frame"]), None)
            next_frame = None if nxt is None else int(nxt["frame"])  # for the early window only: when, not what
            r_h, r_o = track.at(side, e["frame"]), track.at(other, e["frame"])
            row = {c: np.nan for c in SHOT_COLUMNS}
            row.update({"match_id": match_id, "rally_id": int(rid), "shot_index": int(e["shot_index"]),
                        "frame": int(e["frame"]), "time_s": e["frame"] / fps, "hitter_id": e["player_id"],
                        "hitter_side": side, "is_serve": int(e["kind"] == "serve"),
                        "time_since_prev_contact": float(clock.seconds(rally_hits[k - 1]["frame"], e["frame"])) if k else np.nan,
                        "contact_reach": e["reach"], "contact_gap": e["gap"], "ended": int(nxt is None or nxt["kind"] == "landing")})
            hitter_xy = opp_xy = None
            if r_h is not None:
                x, y = players.at[r_h, "court_x"], players.at[r_h, "court_y"]
                if np.isfinite(x) and np.isfinite(y):
                    if in_bounds(x, y):
                        hitter_xy = np.array([x, y])
                        row["hitter_depth"], row["hitter_lateral"] = own_frame(x, y, side)
                    else:
                        blanked["position"] += 1
                row["hitter_speed"] = players.at[r_h, "speed"]
                row.update(track.pose_features(r_h, e["y"]))
            if r_o is not None:
                x, y = players.at[r_o, "court_x"], players.at[r_o, "court_y"]
                row["opponent_id"] = players.at[r_o, "player_id"]
                if np.isfinite(x) and np.isfinite(y):
                    if in_bounds(x, y):
                        opp_xy = np.array([x, y])
                        row["opponent_depth"], row["opponent_lateral"] = own_frame(x, y, other)
                    else:
                        blanked["position"] += 1
                row["opponent_speed"] = players.at[r_o, "speed"]

            # The flight: hit to next event.
            seg = seg_of.get(e["frame"])
            pose = cam["poses"].get(int(seg)) if seg is not None and not pd.isna(seg) else None
            fit = None
            # A flight longer than any badminton shot holds a hit we missed:
            # nothing measured across it would be one shot's.
            if nxt is not None and float(clock.seconds(e["frame"], nxt["frame"])) > float(cfg_f.max_flight_s):
                nxt = None
                long_flights += 1
                row["ended"] = np.nan
            if nxt is not None and pose is not None and hitter_xy is not None:
                T = float(clock.seconds(e["frame"], nxt["frame"]))
                sel = (sf > e["frame"]) & (sf < nxt["frame"])
                ts, uv = clock.seconds(e["frame"], sf[sel]), np.c_[sx[sel], sy[sel]]
                if nxt["kind"] == "landing":
                    end = Anchor(T, np.array([nxt["x"], nxt["y"]]), None)
                else:
                    r_n = track.at(nxt["side"], nxt["frame"])
                    feet = players.loc[r_n, ["court_x", "court_y"]].to_numpy(float) if r_n is not None else None
                    end = Anchor(T, np.array([nxt["x"], nxt["y"]]),
                                 feet if feet is not None and np.isfinite(feet).all() else opp_xy)
                if end.feet is not None or nxt["kind"] == "landing":
                    fit = fit_flight(ts, uv, Anchor(0.0, np.array([e["x"], e["y"]]), hitter_xy), end,
                                     (cam["K"], *pose), cfg.flight, over_net=nxt["kind"] != "landing")
                row["flight_time"] = T
            if fit is not None:
                row["fit_rms_px"], row["fit_n_obs"] = fit["rms_px"], fit["n_obs"]
                if not np.isfinite(fit["rms_px"]) or fit["rms_px"] > float(cfg_f.max_fit_rms_px) * scale:
                    unfit += 1
                    fit = None
            if fit is not None:
                kmh = fit["speed_mps"] * 3.6
                if fit["early_obs"] >= int(cfg.flight.min_early_obs):
                    if kmh < max_kmh:
                        row["shuttle_speed"] = fit["speed_mps"]
                    else:
                        blanked["speed"] += 1
                if fit["net_seen"] and fit["net_z"] is not None and abs(fit["net_x"]) <= DOUBLES_WIDTH / 2 + margin:
                    row["net_clearance"] = fit["net_z"] - net_height(fit["net_x"])

            # Landing: the floor point, or where the shuttle met the receiver.
            land = None
            if nxt is not None and nxt["kind"] == "landing" and seg is not None and int(seg) in homs:
                land, row["landing_src"] = project(homs[int(seg)], np.array([[nxt["x"], nxt["y"]]]))[0], "floor"
            elif fit is not None and nxt is not None:
                land, row["landing_src"] = fit["end"][:2], "shuttle3d"
            elif nxt is not None and nxt["kind"] != "landing":
                r_n = track.at(nxt["side"], nxt["frame"])
                if r_n is not None:
                    land = players.loc[r_n, ["court_x", "court_y"]].to_numpy(float)
                    row["landing_src"] = "receiver"
            if land is not None and np.isfinite(land).all():
                if in_bounds(*land):
                    row["landing_depth"], row["landing_lateral"] = own_frame(land[0], land[1], other)
                    row["dist_from_lines"] = dist_from_singles_lines(*land)
                    start_xy = fit["P0"][:2] if fit is not None else hitter_xy
                    if start_xy is not None:
                        d = land - start_xy
                        row["shot_length"] = float(np.linalg.norm(d))
                        row["cross_court"] = float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]))))
                        if np.isfinite(row["flight_time"]) and row["flight_time"] > 0:
                            row["avg_speed"] = row["shot_length"] / row["flight_time"]
                    if opp_xy is not None:
                        to_land = land - opp_xy
                        row["opponent_dist"] = float(np.linalg.norm(to_land))
                        v = track.velocity(other, e["frame"])
                        if v is not None and row["opponent_dist"] > 1e-6:
                            row["opponent_toward"] = float(v @ to_land / row["opponent_dist"])
                else:
                    blanked["position"] += 1
            # The early flight (phase 7's inputs): what the labelling clip
            # shows, the same for every shot whether it came back or not.
            early, broke = early_flight_features(e, next_frame, vf, vx, vy, clock,
                                                 (cam["K"], *pose) if pose is not None else None, hitter_xy, opp_xy,
                                                 track.velocity(other, e["frame"]) if opp_xy is not None else None,
                                                 other, cfg, scale)
            row.update(early)
            early_unfit += broke["unfit"]
            blanked["early_speed"] += broke["speed"]
            blanked["early_time"] += broke["time"]
            blanked["early_position"] += broke["position"]
            # Recovery: back near their base before their next hit (or the rally's end).
            nxt_own = next((h["frame"] for h in rally_hits[k + 1:] if h["side"] == side), rally["frame"].max() + int(fps))
            row["hitter_recovery_time"] = recovery_time(track, side, e["player_id"], e["frame"], nxt_own,
                                                        bases.get(int(e["player_id"])) if not pd.isna(e["player_id"]) else None,
                                                        float(cfg_f.recovery_radius_m))
            rows.append(row)

    df = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    df.to_csv(out_file, index=False)
    n = len(df)
    meta = {"shots": n, "blanked": blanked, "unfit": unfit, "long_flights": long_flights, "early_unfit": early_unfit,
            "with_early_landing": int(df["early_landing_depth"].notna().sum()),
            "with_early_speed": int(df["early_shuttle_speed"].notna().sum()),
            "with_early_net_clearance": int(df["early_net_clearance"].notna().sum()),
            "seconds": round(time.perf_counter() - started, 1),
            "with_speed": int(df["shuttle_speed"].notna().sum()), "with_net_clearance": int(df["net_clearance"].notna().sum()),
            "bases": {str(k): [round(float(v), 2) for v in b] for k, b in bases.items()}}
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"shots: wrote {out_file} — {n} shots; speed on {meta['with_speed']}, net clearance on "
          f"{meta['with_net_clearance']}; {unfit} flights not fitted, {long_flights} over {cfg_f.max_flight_s} s (a missed hit); "
          f"early flight: landing on {meta['with_early_landing']}, speed on {meta['with_early_speed']}, net clearance on "
          f"{meta['with_early_net_clearance']}, {early_unfit} not fitted; blanked {blanked} in {meta['seconds']}s")
    bad = blanked["speed"] + blanked["position"]
    if n and bad / n > float(cfg_f.max_blanked_frac):
        raise FeatureCheckError(f"{match_id}: {bad} of {n} shot values broke an invariant or failed to fit "
                                f"({blanked}) — over {cfg_f.max_blanked_frac:.0%} is systematic")
    # The early flight has its own rule: extrapolated from a third of a
    # second, its landing leaves the court + 2 m on 2-13% of a match's shots
    # (the far player's mostly, where one camera barely pins depth) and the
    # fit breaks (500 km/h, the solver's cap) on 1-5% — a known limit, not a
    # broken match — so the 15% above stays about the measured flights, as it was.
    bad_early = blanked["early_speed"] + blanked["early_time"] + blanked["early_position"]
    if n and bad_early / n > EARLY_MAX_BLANKED_FRAC:
        raise FeatureCheckError(f"{match_id}: {bad_early} of {n} early-flight values broke an invariant "
                                f"({blanked}) — over {EARLY_MAX_BLANKED_FRAC:.0%} is systematic")
    return df
