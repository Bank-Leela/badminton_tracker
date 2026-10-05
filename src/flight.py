"""Phase 5, part 3: each shot's 3D flight, from one camera.

Between a hit and the next hit (or the landing) the shuttle flies under
gravity and quadratic drag — a feather shuttle's terminal velocity is ~6.8
m/s — so its path is set by six numbers: where it left the racket and how
fast. Those are fitted so that the path, projected through the calibrated
camera (`camera.py`), runs through the shuttle's image track, with the ends
pinned the way badminton pins them: it leaves within reach of the hitter at
the contact pixel, and arrives within reach of the receiver at the next
contact pixel — or on the floor at the landing pixel. One camera cannot see
depth; gravity, drag and those two anchors supply it.

From the fitted path: the speed off the racket (`speed_mps`), the height as
it crosses the net plane (`net_z`, so `net_clearance`), its apex.

The solver is a small damped Gauss-Newton (Levenberg-Marquardt) on numeric
Jacobians: six unknowns, a few hundred residuals a shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


def simulate(P0: np.ndarray, V0: np.ndarray, ts: np.ndarray, v_term: float, dt: float = 1 / 120) -> np.ndarray:
    """Positions at times `ts` (s, from 0) for a batch of launches: `P0, V0` `(B, 3)` -> `(B, len(ts), 3)`.

    dv/dt = −g ẑ − (g / v_term²)|v| v, integrated with RK4.
    """
    P0, V0 = np.atleast_2d(P0).astype(float), np.atleast_2d(V0).astype(float)
    k = G / v_term ** 2
    gvec = np.array([0.0, 0.0, -G])

    def acc(v):
        return gvec - k * np.linalg.norm(v, axis=1, keepdims=True) * v

    t_end = float(np.max(ts)) if len(ts) else 0.0
    n = int(np.ceil(t_end / dt)) + 1
    grid = np.arange(n + 1) * dt
    traj = np.empty((len(P0), n + 1, 3))
    p, v = P0.copy(), V0.copy()
    traj[:, 0] = p
    for i in range(n):
        a1 = acc(v)
        v2 = v + 0.5 * dt * a1
        a2 = acc(v2)
        v3 = v + 0.5 * dt * a2
        a3 = acc(v3)
        v4 = v + dt * a3
        a4 = acc(v4)
        p = p + dt * (v + 2 * v2 + 2 * v3 + v4) / 6
        v = v + dt * (a1 + 2 * a2 + 2 * a3 + a4) / 6
        traj[:, i + 1] = p
    # Linear interpolation onto the requested times.
    j = np.clip((np.asarray(ts) / dt).astype(int), 0, n - 1)
    w = (np.asarray(ts) / dt - j)[None, :, None]
    return traj[:, j] * (1 - w) + traj[:, j + 1] * w


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """`(..., 3)` court points to `(..., 2)` pixels."""
    p = X @ R.T + t
    p = p @ K.T
    return p[..., :2] / p[..., 2:3]


def back_project(K: np.ndarray, R: np.ndarray, t: np.ndarray, uv) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre and unit ray direction (court frame) through pixel `uv`."""
    d = R.T @ np.linalg.solve(K, np.r_[uv, 1.0])
    return -R.T @ t, d / np.linalg.norm(d)


def ray_at_y(K, R, t, uv, y: float) -> np.ndarray:
    """The point on pixel `uv`'s ray at court depth `y`."""
    c, d = back_project(K, R, t, uv)
    s = (y - c[1]) / d[1] if abs(d[1]) > 1e-9 else 0.0
    return c + s * d


def ray_at_z(K, R, t, uv, z: float = 0.0) -> np.ndarray:
    """The point on pixel `uv`'s ray at height `z` (the floor for 0)."""
    c, d = back_project(K, R, t, uv)
    s = (z - c[2]) / d[2] if abs(d[2]) > 1e-9 else 0.0
    return c + s * d


@dataclass
class Anchor:
    """One end of a flight: when, which pixel, and where in the world it must be.

    `feet` is the player's court (x, y) for a hit, or None for a landing
    (the shuttle on the floor). For a hit the shuttle is pulled towards the
    player — horizontally, a Gaussian of `spread` metres around the feet —
    and kept within `reach` (a full-stretch lunge) and between `z_lo` and
    `z_hi`. A pull, not only a fence: one camera leaves a flat drive's depth
    loose by a metre, and "the hit happens at the body" is what settles it.
    """

    s: float  # seconds from the start of the flight
    uv: np.ndarray
    feet: np.ndarray | None = None
    spread: float = 0.6
    reach: float = 1.4
    z_lo: float = 0.05
    z_hi: float = 3.6


NET_SAMPLES = 24


def _net_shortfall(X: np.ndarray) -> np.ndarray:
    """How far below the net tape each path `(B, n, 3)` crosses the net plane (0 if over, or if it never crosses)."""
    y, z, x = X[..., 1], X[..., 2], X[..., 0]
    out = np.zeros(len(X))
    for b in range(len(X)):
        c = np.flatnonzero(np.sign(y[b, :-1]) != np.sign(y[b, 1:]))
        if len(c):
            i = c[0]
            a = y[b, i] / (y[b, i] - y[b, i + 1])
            zc, xc = z[b, i] + a * (z[b, i + 1] - z[b, i]), x[b, i] + a * (x[b, i + 1] - x[b, i])
            top = 1.524 + (1.55 - 1.524) * min(1.0, (xc / 3.05) ** 2)
            out[b] = max(0.0, top - zc)
    return out


def _residuals(theta, ts, uv, anchors, cam, cfg_f, sigma_px, over_net=False):
    """Pixel residuals of the track and the anchors, plus soft bounds, for a batch of parameter vectors `(B, 6)`.

    `over_net`: the opponent hit it next, so it went over the net — a path
    crossing below the tape is penalised.
    """
    K, R, t = cam
    dense = np.linspace(0, anchors[-1].s, NET_SAMPLES) if over_net else np.zeros(0)
    s_all = np.r_[ts, [a.s for a in anchors], dense]
    X_all = simulate(theta[:, :3], theta[:, 3:], s_all, float(cfg_f.v_term_mps))
    X = X_all[:, :len(ts) + len(anchors)]
    px = project(K, R, t, X)
    obs = np.r_[uv, [a.uv for a in anchors]]
    w = np.r_[np.ones(len(ts)), np.full(len(anchors), float(cfg_f.anchor_weight))]
    r = [((px - obs[None]) * w[None, :, None] / sigma_px).reshape(len(theta), -1)]
    for k, a in enumerate(anchors):
        P = X[:, len(ts) + k]
        if a.feet is None:  # landing: on the floor
            r.append(((P[:, 2] - 0.0) / 0.05)[:, None])
        else:
            d = np.linalg.norm(P[:, :2] - a.feet[None], axis=1)
            r.append((d / a.spread)[:, None])
            r.append((np.maximum(0, d - a.reach) / 0.1)[:, None])
            r.append((np.maximum(0, a.z_lo - P[:, 2]) / 0.05)[:, None])
            r.append((np.maximum(0, P[:, 2] - a.z_hi) / 0.05)[:, None])
    if over_net:
        r.append((_net_shortfall(X_all[:, len(ts) + len(anchors):]) / 0.03)[:, None])
    return np.concatenate(r, axis=1)


def fit_flight(ts: np.ndarray, uv: np.ndarray, start: Anchor, end: Anchor, cam, cfg_f, sigma_px: float = 3.0,
               over_net: bool = False,
               iters: int = 40) -> dict:
    """Fit `P0, V0` (court m, m/s at the hit) to a shot's image track; see the module docstring.

    `ts` seconds from the hit, `uv` the shuttle pixels between the anchors.
    """
    K, R, t = cam
    anchors = [start, end]
    # Start: on the contact pixel's ray at the hitter's depth. End: on the
    # next contact's ray at the receiver's depth, or on the floor.
    P0 = ray_at_y(K, R, t, start.uv, start.feet[1]) if start.feet is not None else ray_at_z(K, R, t, start.uv)
    P1 = ray_at_y(K, R, t, end.uv, end.feet[1]) if end.feet is not None else ray_at_z(K, R, t, end.uv)
    T = max(end.s, 1e-3)
    V0 = (P1 - P0) / T
    V0[2] += 0.5 * G * T  # ballistic first guess; drag is learned from the track
    theta = np.r_[P0, V0][None]
    lam = 1e-2
    r = _residuals(theta, ts, uv, anchors, cam, cfg_f, sigma_px, over_net)[0]
    cost = r @ r
    eps = np.r_[np.full(3, 1e-3), np.full(3, 1e-3)]
    for _ in range(iters):
        batch = theta + np.vstack([np.zeros(6), np.diag(eps)])
        rr = _residuals(batch, ts, uv, anchors, cam, cfg_f, sigma_px, over_net)
        J = ((rr[1:] - rr[0]) / eps[:, None]).T
        g = J.T @ rr[0]
        A = J.T @ J
        improved = False
        for _ in range(8):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            cand = theta + step[None]
            v = np.linalg.norm(cand[0, 3:])
            if v > 160.0:  # faster than any shuttle: a step into nonsense (and float overflow)
                cand[0, 3:] *= 160.0 / v
            rc = _residuals(cand, ts, uv, anchors, cam, cfg_f, sigma_px, over_net)[0]
            if rc @ rc < cost:
                theta, cost, lam, improved = cand, rc @ rc, lam / 3, True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-5:
            break
    P0, V0 = theta[0, :3], theta[0, 3:]
    s_dense = np.linspace(0, end.s, max(2, int(end.s * 240) + 1))
    X = simulate(P0, V0, s_dense, float(cfg_f.v_term_mps))[0]
    px = project(K, R, t, simulate(P0, V0, ts, float(cfg_f.v_term_mps))[0]) if len(ts) else np.zeros((0, 2))
    rms = float(np.sqrt(np.mean(np.sum((px - uv) ** 2, axis=1)))) if len(ts) else np.nan
    # Net plane crossing (y = 0), if the flight crosses it.
    y = X[:, 1]
    cross = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))
    net, s_net = None, None
    if len(cross):
        i = cross[0]
        a = y[i] / (y[i] - y[i + 1])
        net = X[i] + a * (X[i + 1] - X[i])
        s_net = float(s_dense[i] + a * (s_dense[i + 1] - s_dense[i]))
    # What the track actually saw: the speed off the racket is only pinned
    # by points soon after the hit (drag takes most of it within ~0.2 s);
    # the net height by points either side of the crossing.
    ts = np.asarray(ts)
    early = int((ts <= float(cfg_f.early_s)).sum())
    w = float(cfg_f.net_window_s)
    net_seen = s_net is not None and ((ts >= s_net - w) & (ts < s_net)).any() and ((ts > s_net) & (ts <= s_net + w)).any()
    return {"P0": P0, "V0": V0, "speed_mps": float(np.linalg.norm(V0)), "rms_px": rms, "n_obs": int(len(ts)),
            "net_x": None if net is None else float(net[0]), "net_z": None if net is None else float(net[2]),
            "s_net": s_net, "early_obs": early, "net_seen": bool(net_seen),
            "apex_z": float(X[:, 2].max()), "end": X[-1], "path": X}
