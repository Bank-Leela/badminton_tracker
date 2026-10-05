import numpy as np
import pandas as pd
import pytest

from config import load_config
from contacts import (
    ContactCheckError,
    PlayerFrames,
    check_rally,
    clean_track,
    label_rally,
    meeting_point,
    shuttle_pieces,
)
from players import KEYPOINTS

CFG = load_config()
C = CFG.contacts
FPS = 30.0
L_WRIST, R_WRIST = KEYPOINTS.index("left_wrist"), KEYPOINTS.index("right_wrist")


def arc(t, p0, v0, a):
    """A smooth image-plane flight from p0 at frame t[0]."""
    s = (t - t[0]) / FPS
    return np.c_[p0[0] + v0[0] * s + 0.5 * a[0] * s ** 2, p0[1] + v0[1] * s + 0.5 * a[1] * s ** 2]


def flights(segments, noise=1.5, seed=0):
    """Concatenate flights `(n_frames, start point, velocity px/s, accel px/s²)` end to end.

    Each flight starts where the last one ended, as a hit does. Returns
    frames, points and the hit frames (the joins).
    """
    rng = np.random.default_rng(seed)
    t0, p, ts, pts, joins = 1000, None, [], [], []
    for n, start, v, a in segments:
        t = np.arange(t0, t0 + n, dtype=float)
        q = arc(t, start if p is None else p, v, a)
        ts.append(t)
        pts.append(q)
        if p is not None:
            joins.append(t0)
        p = arc(np.array([t0, t0 + n]), start if p is None else p, v, a)[1]
        t0 += n
    t, q = np.concatenate(ts), np.concatenate(pts)
    return t, q + rng.normal(0, noise, q.shape), joins


def test_clean_track_drops_the_top_edge_and_lone_jumps():
    t = np.arange(10, dtype=float)
    x = 500 + 5 * t
    y = 400 + 3 * t
    y[3] = 5  # pinned to the top edge
    x[6] += 300  # a lone false detection
    tc, xc, yc = clean_track(t, x, y, top_px=30, jump_px=60)
    assert 3 not in tc and 6 not in tc and len(tc) == 8


def test_pieces_break_where_the_racket_bends_the_track():
    t, q, joins = flights([(25, (900, 600), (300, -900), (0, 1200)),
                           (30, None, (-200, 500), (0, 600)),
                           (20, None, (250, -700), (0, 1500))])
    pieces = shuttle_pieces(t, q[:, 0], q[:, 1], FPS, C.break_penalty_px2, C.min_points, 120)
    assert len(pieces) == 3
    assert [p.t0 for p in pieces[1:]] == pytest.approx(joins, abs=1)


def test_one_smooth_flight_is_one_piece():
    t, q, _ = flights([(45, (800, 650), (100, -1200), (0, 900))])
    assert len(shuttle_pieces(t, q[:, 0], q[:, 1], FPS, C.break_penalty_px2, C.min_points, 120)) == 1


def test_meeting_point_across_a_gap():
    t, q, joins = flights([(25, (900, 600), (300, -900), (0, 1200)), (25, None, (-200, 500), (0, 600))], noise=0.5)
    hidden = (t < joins[0] - 2) | (t > joins[0] + 3)  # the shuttle unseen around the hit
    pieces = shuttle_pieces(t[hidden], q[hidden, 0], q[hidden, 1], FPS, C.break_penalty_px2, C.min_points, 120)
    frame, point, miss = meeting_point(pieces[0], pieces[1])
    assert frame == pytest.approx(joins[0], abs=1)
    true = q[t == joins[0]][0]
    assert np.hypot(*(point - true)) < 15 and miss < 20


# --- A whole rally with stand-in players -------------------------------------------------------


def players_frame(frames, near_wrist, far_wrist, near_court=(0.5, -3.0), far_court=(-0.5, 3.5), speed=0.3):
    """Two players over `frames`; wrists given per frame as callables -> (x, y)."""
    rows, kps = [], []
    for f in frames:
        for side, pid, wrist, court, h in (("near", 0, near_wrist, near_court, 300), ("far", 1, far_wrist, far_court, 180)):
            wx, wy = wrist(f)
            kp = np.zeros((17, 3))
            kp[:, 2] = 0.9
            kp[L_WRIST] = (wx, wy, 0.9)
            kp[R_WRIST] = (wx + 400, wy + 400, 0.9)  # the other hand is well away
            rows.append({"frame": f, "side": side, "player_id": pid, "x1": wx - 60, "y1": wy - h / 2,
                         "x2": wx + 60, "y2": wy + h / 2, "court_x": court[0], "court_y": court[1], "speed": speed})
            kps.append(kp)
    return pd.DataFrame(rows), np.array(kps)


def test_a_rally_serve_hits_and_landing():
    """Held by the far server, served, five alternating hits, landing on the floor."""
    hold = np.arange(1000, 1015, dtype=float)
    held = np.c_[700 + 0.3 * (hold - 1000), np.full(len(hold), 400.0)]  # in the far player's hand
    t, q, joins = flights([(20, (705, 400), (150, 600), (0, 300)),   # serve towards the near player
                           (22, None, (-100, -900), (0, 900)),          # near -> far
                           (20, None, (200, 700), (0, 300)),            # far -> near
                           (24, None, (-150, -800), (0, 800)),
                           (20, None, (100, 750), (0, 200)),
                           (18, None, (120, 500), (0, 900))], noise=1.0)
    t = t + 15  # after the hold
    joins = [j + 15 for j in joins]
    rest_t = np.arange(t[-1] + 1, t[-1] + 25)
    rest = np.tile(q[-1], (len(rest_t), 1))
    T = np.r_[hold, t, rest_t]
    Q = np.r_[held, q, rest]
    hitters = ["far"] + ["near", "far"] * 3
    hit_frames = [1015] + joins

    def wrist_for(side):
        def w(f):  # each player's wrist at the shuttle when they hit it, else far away
            k = np.searchsorted(hit_frames, f, side="right") - 1
            if side == "far" and f < 1015:
                return tuple(Q[np.searchsorted(T, f)])  # holding it
            if k >= 0 and hitters[k] == side and abs(f - hit_frames[k]) <= 2:
                i = np.searchsorted(T, hit_frames[k])
                return (Q[i, 0] + 20, Q[i, 1] + 20)
            return (300.0, 900.0) if side == "near" else (1500.0, 300.0)
        return w

    pl, kp = players_frame(np.arange(950, int(T[-1]) + 1), wrist_for("near"), wrist_for("far"))
    who = PlayerFrames(pl, kp)
    pieces = shuttle_pieces(T, Q[:, 0], Q[:, 1], FPS, C.break_penalty_px2, C.min_points, 120)
    events = label_rally(pieces, who, C, FPS)
    hits = [e for e in events if e["kind"] != "landing"]
    assert [e["side"] for e in hits] == hitters[:len(hits)]
    assert len(hits) == 6
    assert [e["frame"] for e in hits] == pytest.approx(hit_frames[:6], abs=2)
    assert hits[0]["kind"] == "serve"
    assert events[-1]["kind"] == "landing" and events[-1]["frame"] == pytest.approx(rest_t[0], abs=2)
    check_rally(events, FPS, C)


def test_invariants():
    hit = lambda f: {"kind": "hit", "frame": f}
    with pytest.raises(ContactCheckError, match="increasing"):
        check_rally([hit(10), hit(40), hit(30)], FPS, C)
    with pytest.raises(ContactCheckError, match="100 ms"):
        check_rally([hit(10), hit(12)], FPS, C)
    with pytest.raises(ContactCheckError, match="shots"):
        check_rally([], FPS, C)
    check_rally([hit(10), hit(40)], FPS, C)
