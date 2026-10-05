"""Phase 5, part 1: hit moments.

A hit is found from two signals, fused per rally:

* **The shuttle's image track breaks.** Between hits the shuttle flies a smooth
  curve; a racket bends it sharply. Each rally's track is cut into the fewest
  smooth pieces (a cubic in time for x and for y) that fit it — an optimal
  partition, `shuttle_pieces` — so a break needs real evidence, not one noisy
  point. Where the shuttle is lost around a hit (behind a player, blurred),
  the hit is where the two neighbouring pieces' curves come closest.
* **A player is in reach.** At the break the shuttle must be within racket
  reach of one player's wrist in the image (`reach`, in body heights).

Hits then alternate near/far through the rally (singles), so the pieces are
labelled jointly (`label_rally`) rather than one break at a time. A shuttle
at rest before the first hit is the serve being held; at rest after the last,
it has landed — that point is on the floor, so the homography maps it exactly.

Output `contacts.csv`: one row per hit, plus one `landing` row per rally when
the shuttle is seen coming to rest.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config, cache_dir
from players import KEYPOINTS, keypoints_array, load_players

L_WRIST, R_WRIST = KEYPOINTS.index("left_wrist"), KEYPOINTS.index("right_wrist")
DEGREE = 3
CONTACT_COLUMNS = ["rally_id", "shot_index", "kind", "frame", "player_id", "side", "x", "y", "reach",
                   "gap", "miss_px"]


class ContactCheckError(AssertionError):
    """A rally's hits broke one of the plan's invariants (raised explicitly,
    like `court.CourtCheckError`)."""


# --- The shuttle track ---------------------------------------------------------------------


def clean_track(t: np.ndarray, x: np.ndarray, y: np.ndarray, top_px: float, jump_px: float):
    """Visible shuttle points of one rally, minus the unusable ones.

    Points within `top_px` of the top edge are the shuttle leaving the frame
    (TrackNet pins it to the edge). A point far from both neighbours while
    they are close to each other is a one-frame false detection.
    """
    ok = y > top_px
    t, x, y = t[ok], x[ok], y[ok]
    if len(t) < 3:
        return t, x, y
    keep = np.ones(len(t), bool)
    dt_prev = np.maximum(1, t[1:-1] - t[:-2])
    dt_next = np.maximum(1, t[2:] - t[1:-1])
    d_prev = np.hypot(x[1:-1] - x[:-2], y[1:-1] - y[:-2]) / dt_prev
    d_next = np.hypot(x[2:] - x[1:-1], y[2:] - y[1:-1]) / dt_next
    d_skip = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2]) / (dt_prev + dt_next)
    keep[1:-1] = ~((d_prev > jump_px) & (d_next > jump_px) & (d_skip < jump_px / 2))
    return t[keep], x[keep], y[keep]


@dataclass
class Piece:
    """A stretch of track fit by one smooth curve: rows `[i, j)` of the cleaned track."""

    i: int
    j: int
    t0: float  # first and last frame
    t1: float
    cx: np.ndarray  # polynomial coefficients (increasing powers of seconds from t0)
    cy: np.ndarray
    fps: float

    def at(self, t) -> np.ndarray:
        s = (np.asarray(t, float) - self.t0) / self.fps
        V = np.vander(np.atleast_1d(s), len(self.cx), increasing=True)
        return np.c_[V @ self.cx, V @ self.cy]

    def speed(self) -> float:
        """Mean image speed over the piece, px per frame."""
        ts = np.linspace(self.t0, self.t1, max(2, int(self.t1 - self.t0) + 1))
        p = self.at(ts)
        return float(np.hypot(*np.diff(p, axis=0).T).sum() / max(1.0, self.t1 - self.t0))


def _fit(tt: np.ndarray, v: np.ndarray, deg: int) -> np.ndarray:
    V = np.vander(tt, deg + 1, increasing=True)
    return np.linalg.lstsq(V, v, rcond=None)[0]


def shuttle_pieces(t: np.ndarray, x: np.ndarray, y: np.ndarray, fps: float, penalty: float,
                   min_points: int, max_points: int) -> list[Piece]:
    """Optimal partition of a track into smooth pieces.

    Minimises the summed squared residual of a cubic-in-time fit (x and y)
    over every piece plus `penalty` (px²) per piece. Pieces have at least
    `min_points` points (fewer only at the ends, where a track can stop) and
    at most `max_points`. Least squares over any span comes from cumulative
    moment sums, so the partition costs O(n · max_points) small solves.
    """
    n = len(t)
    if n == 0:
        return []
    s = (t - t[0]) / fps
    V = np.vander(s, DEGREE + 1, increasing=True)
    A = np.cumsum(np.einsum("ni,nj->nij", V, V), axis=0)
    bx, by = np.cumsum(V * x[:, None], axis=0), np.cumsum(V * y[:, None], axis=0)
    xx, yy = np.cumsum(x ** 2), np.cumsum(y ** 2)
    ridge = 1e-9 * np.eye(DEGREE + 1)

    def span(c, i, j):
        return c[j - 1] - (c[i - 1] if i > 0 else 0)

    def sse(i, j):
        if j - i <= DEGREE + 1:
            return 0.0  # interpolated exactly
        M = span(A, i, j) + ridge
        cx, cy = span(bx, i, j), span(by, i, j)
        sx = span(xx, i, j) - cx @ np.linalg.solve(M, cx)
        sy = span(yy, i, j) - cy @ np.linalg.solve(M, cy)
        return max(sx, 0.0) + max(sy, 0.0)

    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(n + 1, int)
    for j in range(1, n + 1):
        for i in range(max(0, j - max_points), j):
            if j - i < min_points and i != 0 and j != n:
                continue
            if not np.isfinite(best[i]):
                continue
            c = best[i] + penalty + sse(i, j)
            if c < best[j]:
                best[j], prev[j] = c, i
    bounds, j = [], n
    while j > 0:
        bounds.append((prev[j], j))
        j = prev[j]
    pieces = []
    for i, j in reversed(bounds):
        deg = min(DEGREE, j - i - 1)
        tt = (t[i:j] - t[i]) / fps
        cx, cy = _fit(tt, x[i:j], deg), _fit(tt, y[i:j], deg)
        pieces.append(Piece(i, j, float(t[i]), float(t[j - 1]), cx, cy, fps))
    return pieces


def meeting_point(a: Piece, b: Piece) -> tuple[float, np.ndarray, float]:
    """Where piece `a` hands over to piece `b`: frame, image point, and how far apart the curves are there.

    Without a gap it is the boundary. Across a gap (shuttle unseen around
    the hit), each curve is extrapolated into it and the hit put where they
    come closest; the miss distance says how much to trust it.
    """
    lo, hi = a.t1, b.t0
    ts = np.arange(np.floor(lo), np.ceil(hi) + 1) if hi - lo > 1 else np.array([lo, hi])
    pa, pb = a.at(ts), b.at(ts)
    d = np.hypot(*(pa - pb).T)
    k = int(np.argmin(d))
    return float(ts[k]), (pa[k] + pb[k]) / 2, float(d[k])


# --- Players at a moment ------------------------------------------------------------------


class PlayerFrames:
    """Each player's wrists, body height and court position, looked up by frame
    (the nearest row within `max_lag` frames)."""

    def __init__(self, players: pd.DataFrame, kp: np.ndarray, max_lag: int = 3):
        self.by_side = {}
        for side, g in players.groupby("side"):
            order = np.argsort(g["frame"].to_numpy())
            g = g.iloc[order]
            k = kp[g.index.to_numpy()]
            self.by_side[side] = {
                "frame": g["frame"].to_numpy(), "pid": g["player_id"].to_numpy(),
                "wrists": k[:, [L_WRIST, R_WRIST]], "height": (g["y2"] - g["y1"]).to_numpy(),
                "court": g[["court_x", "court_y"]].to_numpy(np.float64), "speed": g["speed"].to_numpy(np.float64),
            }
        self.max_lag = max_lag

    def _row(self, side: str, frame: float) -> int | None:
        if side not in self.by_side:
            return None
        f = self.by_side[side]["frame"]
        i = int(np.clip(np.searchsorted(f, frame), 0, len(f) - 1))
        c = min((c for c in (i - 1, i) if 0 <= c < len(f)), key=lambda c: abs(f[c] - frame))
        return c if abs(f[c] - frame) <= self.max_lag else None

    def at(self, side: str, frame: float) -> dict | None:
        """`pid, court (x, y), speed` of the side's player at a frame."""
        c = self._row(side, frame)
        if c is None:
            return None
        d = self.by_side[side]
        return {"pid": int(d["pid"][c]), "court": d["court"][c], "speed": float(d["speed"][c])}

    def reach(self, side: str, frame: float, point: np.ndarray) -> tuple[float, int | None]:
        """Distance from `point` to the side's nearer wrist, in body heights; and the player id."""
        c = self._row(side, frame)
        if c is None:
            return np.inf, None
        d = self.by_side[side]
        pid, h, w = d["pid"], d["height"], d["wrists"][c]
        ok = w[:, 2] >= 0.3
        if not ok.any():
            return np.inf, int(pid[c])
        d = np.hypot(w[ok, 0] - point[0], w[ok, 1] - point[1]).min()
        return float(d / max(h[c], 1.0)), int(pid[c])


# --- One rally ----------------------------------------------------------------------------


def piece_state(p: Piece, who: PlayerFrames, cfg_c: Config, scale: float = 1.0) -> str:
    """`rest` (slower than `rest_px_per_frame`: rolling, on the floor), `held`
    (within `hold_reach` body heights of a wrist for `hold_frac` of the piece:
    a server walking to the line with it), or `flight`.

    Speed alone cannot tell a held shuttle from a flight: a server walking
    with it moved it 7 px/frame on World Champs 2025, faster than the slowest
    2% of real flights. But a flying shuttle leaves the hitter's hand at once
    (at contact it is a racket's length away, ~0.3-0.5 body heights).
    """
    speed = p.speed()
    if speed < float(cfg_c.rest_px_per_frame) * scale:
        return "rest"
    if speed >= float(cfg_c.hold_px_per_frame) * scale:
        return "flight"
    ts = np.arange(p.t0, p.t1 + 1, 2)
    pts = p.at(ts)
    hold = float(cfg_c.hold_reach)
    for side in ("near", "far"):  # one player, nearly throughout: a flight only nears the receiver at its end
        in_hand = [who.reach(side, t, pt)[0] <= hold for t, pt in zip(ts, pts)]
        if np.mean(in_hand) >= float(cfg_c.hold_frac):
            return "held"
    return "flight"


def serve_position(who: PlayerFrames, side: str, frame: float, fps: float, cfg_c: Config) -> bool:
    """Both players placed for a serve by `side` just before `frame`.

    In the service courts (`service_depth_m` from the net), standing nearly
    still, and diagonally opposite: as the camera sees it the near player's
    right is +x and the far player's right is −x, so server and receiver
    have opposite-signed x — unless one stands on the centre line.
    """
    before = frame - float(cfg_c.serve_still_s) * fps
    s, r = who.at(side, before), who.at("far" if side == "near" else "near", before)
    if s is None or r is None:
        return False
    lo, hi = cfg_c.service_depth_m
    (xs, ys), (xr, yr) = s["court"], r["court"]
    if not all(np.isfinite([xs, ys, xr, yr])) or not (lo <= abs(ys) <= hi and lo <= abs(yr) <= hi):
        return False
    still = float(cfg_c.serve_still_mps)
    if not (s["speed"] <= still or not np.isfinite(s["speed"])) or not (r["speed"] <= still or not np.isfinite(r["speed"])):
        return False
    centre = float(cfg_c.centre_line_m)
    return xs * xr < 0 or min(abs(xs), abs(xr)) < centre


def held_before(pieces: list[Piece], who: PlayerFrames, side: str, frame: float, fps: float, cfg_c: Config) -> bool:
    """The shuttle was seen at `side`'s hand in the moment before `frame`.

    A server holds the shuttle until the stroke; a tap back to the server
    comes off the floor or out of a flight. Checked on the fitted track over
    `serve_hold_s` before the stroke (its last frames left out: the swing).
    """
    fs = np.arange(frame - float(cfg_c.serve_hold_s) * fps, frame - 2)
    seen = []
    for f in fs:
        p = next((p for p in pieces if p.t0 <= f <= p.t1), None)
        if p is not None:
            seen.append(who.reach(side, f, p.at(f)[0])[0])
    if len(seen) < 0.7 * len(fs):
        return False
    return float(np.median(seen)) <= 1.5 * float(cfg_c.hold_reach)


def label_rally(pieces: list[Piece], who: PlayerFrames, cfg_c: Config, fps: float, scale: float = 1.0) -> list[dict]:
    """The rally's hits (and landing) from its pieces.

    Each piece is a flight or not (`piece_state`: held in a hand, or slower
    than `rest_px_per_frame` — rolling, on the floor). Every boundary into a
    flight is a candidate hit by whichever player is in reach.

    The serve is the first candidate made from a serving position
    (`serve_position`) with the shuttle in the server's hand just before
    (`held_before`); before it come the shuttle being tapped back to the
    server and carried to the line, which are not the rally. A flight that
    ends at rest after the serve is the landing, and ends the rally. Between
    them hits must alternate sides: a Viterbi pass keeps the alternating
    sequence with the most hits, ties going to the closer reach. A boundary
    nobody reaches is not a hit (a break in one flight — the shuttle lost, or
    a fit glitch). With no serving position found, the first candidate is the
    serve.
    """
    reach_max = float(cfg_c.reach)
    states = [piece_state(p, who, cfg_c, scale) for p in pieces]
    flying = [s == "flight" for s in states]
    if not any(flying):
        return []

    cand = []  # (frame, image point, miss px, gap frames)
    if flying[0]:
        cand.append((pieces[0].t0, pieces[0].at(pieces[0].t0)[0], 0.0, 0))
    for k in range(len(pieces) - 1):
        if flying[k + 1]:
            t, pt, miss = meeting_point(pieces[k], pieces[k + 1])
            cand.append((t, pt, miss, pieces[k + 1].t0 - pieces[k].t1))
    landings = [(pieces[k + 1].t0, pieces[k + 1].at((pieces[k + 1].t0 + pieces[k + 1].t1) / 2)[0])
                for k in range(len(pieces) - 1) if flying[k] and states[k + 1] == "rest"]

    scored = []
    for t, pt, miss, gap in cand:
        opts = {}
        for side in ("near", "far"):
            d, pid = who.reach(side, t, pt)
            if d <= reach_max:
                opts[side] = (d, pid)
        scored.append((t, pt, miss, gap, opts))

    # The serve, then everything up to the landing.
    serve = next(((i, side) for i, (t, *_, opts) in enumerate(scored)
                  for side in sorted(opts, key=lambda s: opts[s][0])
                  if serve_position(who, side, t, fps, cfg_c) and held_before(pieces, who, side, t, fps, cfg_c)),
                 None)
    start = serve[0] if serve else next((i for i, c in enumerate(scored) if c[4]), None)
    if start is None:
        return []
    if serve:
        scored[start] = (*scored[start][:4], {serve[1]: scored[start][4][serve[1]]})
    t_serve = scored[start][0]
    landing = next((l for l in landings if l[0] > t_serve), None)
    scored = [c for c in scored[start:] if landing is None or c[0] < landing[0]]

    # Viterbi over candidates: state = side of the last accepted hit; the serve is always taken.
    min_gap = float(cfg_c.min_gap_s) * fps
    t0, pt0, _, _, opts0 = scored[0]
    best = {side: (1, -d, [(0, side, d, pid)]) for side, (d, pid) in opts0.items()}
    for idx, (t, pt, miss, gap, opts) in enumerate(scored[1:], start=1):
        new = dict(best)
        for last, (n, sr, chosen) in best.items():
            for side, (d, pid) in opts.items():
                if side == last:
                    continue
                if chosen and t - scored[chosen[-1][0]][0] < min_gap:
                    continue
                cand_val = (n + 1, sr - d, chosen + [(idx, side, d, pid)])
                if side not in new or cand_val[:2] > new[side][:2]:
                    new[side] = cand_val
        best = new
    if not best:
        return []
    n, _, chosen = max(best.values(), key=lambda v: v[:2])
    events = []
    for shot, (idx, side, d, pid) in enumerate(chosen, start=1):
        t, pt, miss, gap, _ = scored[idx]
        kind = ("serve" if serve else "first") if shot == 1 else "hit"
        events.append({"shot_index": shot, "kind": kind, "frame": int(round(t)),
                       "player_id": pid, "side": side, "x": float(pt[0]), "y": float(pt[1]),
                       "reach": round(d, 3), "gap": int(gap), "miss_px": round(miss, 1)})
    if landing is not None and events and landing[0] > events[-1]["frame"]:
        t, rest_pt = landing
        events.append({"shot_index": None, "kind": "landing", "frame": int(round(t)), "player_id": None,
                       "side": None, "x": float(rest_pt[0]), "y": float(rest_pt[1]), "reach": None,
                       "gap": None, "miss_px": None})
    return events


def check_rally(events: list[dict], fps: float, cfg_c: Config) -> None:
    """The plan's invariants for one rally's hits."""
    hits = [e for e in events if e["kind"] != "landing"]
    frames = [e["frame"] for e in hits]
    if any(b <= a for a, b in zip(frames, frames[1:])):
        raise ContactCheckError(f"contact frames not strictly increasing: {frames}")
    if any((b - a) / fps < 0.1 for a, b in zip(frames, frames[1:])):
        raise ContactCheckError(f"two contacts within 100 ms: {frames}")
    if not 1 <= len(hits) <= int(cfg_c.max_shots):
        raise ContactCheckError(f"{len(hits)} shots in a rally (must be 1-{cfg_c.max_shots})")


# --- Driver -------------------------------------------------------------------------------


def rally_spans(segs: pd.DataFrame) -> pd.DataFrame:
    """`rally_id, start_frame, end_frame` from `segments.csv` (a rally may span several rows)."""
    r = segs[(segs["is_play"] == 1) & (segs["rally_id"] >= 0)]
    return r.groupby("rally_id").agg(start_frame=("start_frame", "min"), end_frame=("end_frame", "max")).reset_index()


def detect_contacts(cfg: Config, match_id: str, force: bool = False) -> pd.DataFrame:
    """Hits and landings for every rally of a match; writes `contacts.csv`."""
    out_dir = cache_dir(cfg, match_id)
    out_file = out_dir / "contacts.csv"
    meta_file = out_dir / "contacts.meta.json"
    needed = [out_dir / n for n in ("shuttle.csv", "segments.csv", "players.parquet", "poses.meta.json")]
    for f in needed:
        if not f.exists():
            raise SystemExit(f"no {f}; run the earlier phases first")
    cfg_c = cfg.contacts
    import hashlib

    inputs = {"contacts": cfg_c.to_dict(),
              **{f.name: hashlib.sha1(f.read_bytes()).hexdigest() for f in needed[:2]},
              "players": json.loads((out_dir / "players.meta.json").read_text())["inputs"]}
    if out_file.exists() and meta_file.exists() and not force:
        if json.loads(meta_file.read_text()).get("inputs") == inputs:
            print(f"contacts: reusing {out_file}")
            return pd.read_csv(out_file)

    started = time.perf_counter()
    fps = float(json.loads((out_dir / "poses.meta.json").read_text())["fps"])
    height = json.loads((out_dir / "homography.json").read_text())["image_size"][1]
    scale = height / 1080
    shuttle = pd.read_csv(out_dir / "shuttle.csv")
    shuttle = shuttle[shuttle["visible"] == 1]
    players = load_players(out_dir / "players.parquet")
    who = PlayerFrames(players, keypoints_array(players))
    rows, problems = [], []
    for r in rally_spans(pd.read_csv(out_dir / "segments.csv")).itertuples():
        sh = shuttle[(shuttle["frame"] >= r.start_frame) & (shuttle["frame"] < r.end_frame)]
        t, x, y = clean_track(sh["frame"].to_numpy(float), sh["x"].to_numpy(), sh["y"].to_numpy(),
                              float(cfg_c.top_px) * scale, float(cfg_c.jump_px) * scale)
        if len(t) < 4:
            problems.append({"rally_id": int(r.rally_id), "problem": "no shuttle track"})
            continue
        pieces = shuttle_pieces(t, x, y, fps, float(cfg_c.break_penalty_px2) * scale ** 2,
                                int(cfg_c.min_points), int(round(float(cfg_c.max_piece_s) * fps)))
        events = label_rally(pieces, who, cfg_c, fps, scale)
        try:
            check_rally(events, fps, cfg_c)
        except ContactCheckError as exc:
            problems.append({"rally_id": int(r.rally_id), "problem": str(exc)})
            continue
        rows += [{"rally_id": int(r.rally_id), **e} for e in events]
    df = pd.DataFrame(rows, columns=CONTACT_COLUMNS)
    df.to_csv(out_file, index=False)
    hits = df[df["kind"] != "landing"]
    meta = {"inputs": inputs, "rallies": int(df["rally_id"].nunique()), "hits": len(hits),
            "landings": int((df["kind"] == "landing").sum()), "problems": problems,
            "seconds": round(time.perf_counter() - started, 1)}
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"contacts: wrote {out_file} — {len(hits)} hits in {meta['rallies']} rallies "
          f"({len(hits) / max(1, meta['rallies']):.1f} a rally), {meta['landings']} landings seen; "
          f"{len(problems)} rallies skipped")
    return df
