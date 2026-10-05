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
from flight import Anchor, fit_flight
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
    shuttle = shuttle[shuttle["visible"] == 1]
    sf, sx, sy = clean_track(shuttle["frame"].to_numpy(float), shuttle["x"].to_numpy(), shuttle["y"].to_numpy(),
                             float(cfg.contacts.top_px) * scale, float(cfg.contacts.jump_px) * scale)
    clock = frame_clock(out_dir, contacts, fps)
    seg_of = players.drop_duplicates("frame").set_index("frame")["segment_id"]
    hits = contacts[contacts["kind"] != "landing"]
    bases = player_bases(hits, track)
    max_kmh, margin = float(cfg_f.max_shuttle_kmh), float(cfg_f.court_margin_m)
    rows, blanked, unfit, long_flights = [], {"speed": 0, "position": 0}, 0, 0

    def in_bounds(x, y):
        return abs(x) <= DOUBLES_WIDTH / 2 + margin and abs(y) <= HALF_L + margin

    for rid, rally in contacts.groupby("rally_id"):
        ev = rally.sort_values("frame").to_dict("records")
        rally_hits = [e for e in ev if e["kind"] != "landing"]
        for k, e in enumerate(rally_hits):
            side, other = e["side"], ("far" if e["side"] == "near" else "near")
            nxt = next((n for n in ev if n["frame"] > e["frame"]), None)
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
            # Recovery: back near their base before their next hit (or the rally's end).
            nxt_own = next((h["frame"] for h in rally_hits[k + 1:] if h["side"] == side), rally["frame"].max() + int(fps))
            row["hitter_recovery_time"] = recovery_time(track, side, e["player_id"], e["frame"], nxt_own,
                                                        bases.get(int(e["player_id"])) if not pd.isna(e["player_id"]) else None,
                                                        float(cfg_f.recovery_radius_m))
            rows.append(row)

    df = pd.DataFrame(rows, columns=SHOT_COLUMNS)
    df.to_csv(out_file, index=False)
    n = len(df)
    meta = {"shots": n, "blanked": blanked, "unfit": unfit, "long_flights": long_flights, "seconds": round(time.perf_counter() - started, 1),
            "with_speed": int(df["shuttle_speed"].notna().sum()), "with_net_clearance": int(df["net_clearance"].notna().sum()),
            "bases": {str(k): [round(float(v), 2) for v in b] for k, b in bases.items()}}
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"shots: wrote {out_file} — {n} shots; speed on {meta['with_speed']}, net clearance on "
          f"{meta['with_net_clearance']}; {unfit} flights not fitted, {long_flights} over {cfg_f.max_flight_s} s (a missed hit); blanked {blanked} in {meta['seconds']}s")
    bad = sum(blanked.values())
    if n and bad / n > float(cfg_f.max_blanked_frac):
        raise FeatureCheckError(f"{match_id}: {bad} of {n} shot values broke an invariant or failed to fit "
                                f"({blanked}) — over {cfg_f.max_blanked_frac:.0%} is systematic")
    return df
