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

The end anchor and the net constraint say how the flight ended, so phase 7
can't use what they shape (it would learn the outcome from them).
`fit_early_flight` is the second fit, for phase 7: the start anchor and the
first frames of track only — what the labelling clip shows — flown on to
the floor. See `docs/features.md`, "Early-flight features".

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
    t_end = float(np.max(ts)) if len(ts) else 0.0
    traj, n = _integrate(P0, V0, t_end, v_term, dt)
    # Linear interpolation onto the requested times.
    j = np.clip((np.asarray(ts) / dt).astype(int), 0, n - 1)
    w = (np.asarray(ts) / dt - j)[None, :, None]
    return traj[:, j] * (1 - w) + traj[:, j + 1] * w


def simulate_rows(P0: np.ndarray, V0: np.ndarray, T: np.ndarray, v_term: float, dt: float = 1 / 120) -> np.ndarray:
    """`simulate` with each launch at its own times: `T` `(B, n)` -> `(B, n, 3)` (times before 0 read as 0)."""
    P0, V0 = np.atleast_2d(P0).astype(float), np.atleast_2d(V0).astype(float)
    T = np.clip(np.asarray(T, float), 0.0, None)
    traj, n = _integrate(P0, V0, float(T.max()) if T.size else 0.0, v_term, dt)
    j = np.clip((T / dt).astype(int), 0, n - 1)
    w = (T / dt - j)[..., None]
    b = np.arange(len(P0))[:, None]
    return traj[b, j] * (1 - w) + traj[b, j + 1] * w


def _integrate(P0: np.ndarray, V0: np.ndarray, t_end: float, v_term: float, dt: float) -> tuple[np.ndarray, int]:
    """The paths `(B, n + 1, 3)` on the grid 0, dt, ... past `t_end` (RK4), and `n`."""
    k = G / v_term ** 2
    gvec = np.array([0.0, 0.0, -G])

    def acc(v):
        # |v| as np.linalg.norm(v, axis=1) computes it (bit for bit), without its per-call overhead.
        return gvec - k * np.sqrt(np.add.reduce(v * v, axis=1, keepdims=True)) * v

    n = int(np.ceil(t_end / dt)) + 1
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
    return traj, n


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


EPS = np.r_[np.full(3, 1e-3), np.full(3, 1e-3)]  # numeric-Jacobian steps for (P0, V0)
# The solver never steps to a launch faster than this (m/s, 576 km/h — faster
# than any shuttle; past it lie nonsense and float overflow). A fit that ends
# here was stopped, not fitted: a broken fit, whatever the speed invariant.
V_CAP_MPS = 160.0


def at_speed_cap(V0: np.ndarray) -> bool:
    """The launch velocity sits on `_solve`'s speed cap (the fit was stopped there, not fitted)."""
    return float(np.linalg.norm(V0)) >= V_CAP_MPS * (1 - 1e-6)


def _solve(theta: np.ndarray, resid, iters: int, eps: np.ndarray = EPS) -> tuple[np.ndarray, float]:
    """Damped Gauss-Newton (Levenberg-Marquardt) from `theta` `(1, p)` on the batched residual function `resid`.

    `theta[:, :3]` and `[:, 3:6]` are `P0, V0`; `eps` the Jacobian steps, one per parameter.
    """
    lam = 1e-2
    r = resid(theta)[0]
    cost = r @ r
    for _ in range(iters):
        batch = theta + np.vstack([np.zeros(len(eps)), np.diag(eps)])
        rr = resid(batch)
        J = ((rr[1:] - rr[0]) / eps[:, None]).T
        g = J.T @ rr[0]
        A = J.T @ J
        improved = False
        for _ in range(8):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            cand = theta + step[None]
            v = np.linalg.norm(cand[0, 3:6])
            if v > V_CAP_MPS:  # faster than any shuttle: a step into nonsense (and float overflow)
                cand[0, 3:6] *= V_CAP_MPS / v
            rc = resid(cand)[0]
            if rc @ rc < cost:
                theta, cost, lam, improved = cand, rc @ rc, lam / 3, True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-5:
            break
    return theta, float(cost)


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
    theta, _ = _solve(np.r_[P0, V0][None], lambda th: _residuals(th, ts, uv, anchors, cam, cfg_f, sigma_px, over_net),
                      iters)
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


# --- The early flight: what the labelling clip shows, flown on to the floor ---------------------

FLY_MAX_S = 5.0  # no shuttle stays up this long (a high clear: ~2 s); a path still up is a bad fit
FORWARD_SIGMA_MPS = 0.5  # soft wall on the shuttle going back towards the hitter's baseline (m/s)
LAUNCH_SIGMA_S = 0.02  # prior sd (s) of the launch time about the contact frame: half a frame at 25 fps
# The start pixel's weight, in track points. It is the shuttle as the clip
# shows it at the contact frame (or the one before), up to half a frame of
# flight from the launch, not where the curves meet (`fit_flight`'s anchor,
# `flight.anchor_weight` = 3): on the floor-ended shots 0.3 put the far
# player's landings 0.3 m nearer than 3, the near player's no worse
# (docs/features.md). Would be `flight.early_anchor_weight`.
EARLY_ANCHOR_WEIGHT = 0.3
EARLY_EPS = np.r_[EPS, 1e-4]  # numeric-Jacobian steps for (P0, V0, tau)


def _crossing(s: np.ndarray, X: np.ndarray, axis: int, level: float = 0.0):
    """First point (and time) where the path `X` `(n, 3)` at times `s` crosses `level` along `axis`; (None, None) if never."""
    d = X[:, axis] - level
    c = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if not len(c):
        return None, None
    i = c[0]
    a = d[i] / (d[i] - d[i + 1])
    return X[i] + a * (X[i + 1] - X[i]), float(s[i] + a * (s[i + 1] - s[i]))


def fly_to_floor(P0: np.ndarray, V0: np.ndarray, v_term: float, t_max: float = FLY_MAX_S, dt: float = 1 / 240):
    """The flight from `P0, V0` until it reaches the floor: `(s, X, land, t_land)`.

    `X` the path at times `s`, ending on the floor; `land` the court (x, y)
    where z reaches 0 and `t_land` when. `land` is None if it starts on or
    under the floor or is still up after `t_max`.
    """
    s = np.arange(int(round(t_max / dt)) + 1) * dt
    X = simulate(P0, V0, s, v_term)[0]
    if X[0, 2] <= 0:
        return s[:1], X[:1], None, None
    P, t_land = _crossing(s, X, 2)
    if P is None:
        return s, X, None, None
    i = int(np.searchsorted(s, t_land))
    return np.r_[s[:i], t_land], np.vstack([X[:i], P]), P[:2].copy(), t_land


def toward_net(side: str) -> float:
    """Which way along court y a `side` player's strokes go: the near player stands at y < 0, so +1; the far, −1."""
    if side not in ("near", "far"):
        raise ValueError(f"hitter side must be 'near' or 'far', not {side!r}")
    return 1.0 if side == "near" else -1.0


def _early_guesses(ts: np.ndarray, uv: np.ndarray, start: Anchor, cam, toward: float) -> np.ndarray:
    """First guesses `(n, 6)`: the hit on the contact pixel's ray (the first track point's, without
    one) at the hitter's depth; the last track point at depths along its own ray from one baseline
    (plus margin) to the other — the depth the one camera cannot see — reached ballistically. The
    fit picks among them."""
    K, R, t = cam
    P0 = ray_at_y(K, R, t, start.uv if start.uv is not None else uv[0], start.feet[1])
    c, d = back_project(K, R, t, uv[-1])
    s1 = max(float(ts[-1]), 1e-3)
    out = []
    for y in np.linspace(-9.0, 9.0, 19):
        if abs(d[1]) < 1e-9:
            continue
        lam = (y - c[1]) / d[1]
        P = c + lam * d
        if lam <= 0 or not 0.0 < P[2] < 15.0:
            continue
        V = (P - P0) / s1
        V[2] += 0.5 * G * s1
        if np.linalg.norm(V) < 120.0:
            out.append(np.r_[P0, V])
    if not out:  # nothing plausible: up and towards the net from the hitter, let the fit move it
        out.append(np.r_[P0, 0.0, toward * 10.0, 5.0])
    return np.array(out)


def _early_residuals(th: np.ndarray, ts: np.ndarray, uv: np.ndarray, start: Anchor, cam, cfg_f, sigma_px: float,
                     toward: float, t_first: float) -> np.ndarray:
    """Residuals of the early fit for a batch of `(P0, V0, tau)` `(B, 7)`.

    The shuttle leaves `P0` at `tau` seconds from the contact frame; the track
    points at `ts` (from the contact frame) are where it is `ts - tau` into its
    flight. The start anchor is at the launch: the feet terms as `fit_flight`
    has them, the pixel term (weight `EARLY_ANCHOR_WEIGHT`) only when
    `start.uv` is given.
    """
    K, R, t = cam
    P0, tau = th[:, :3], th[:, 6]
    X = simulate_rows(P0, th[:, 3:6], ts[None] - tau[:, None], float(cfg_f.v_term_mps))
    r = [((project(K, R, t, X) - uv[None]) / sigma_px).reshape(len(th), -1)]
    if start.uv is not None:
        r.append((project(K, R, t, P0) - start.uv[None]) * EARLY_ANCHOR_WEIGHT / sigma_px)
    d = np.linalg.norm(P0[:, :2] - start.feet[None], axis=1)
    r += [(d / start.spread)[:, None], (np.maximum(0, d - start.reach) / 0.1)[:, None],
          (np.maximum(0, start.z_lo - P0[:, 2]) / 0.05)[:, None], (np.maximum(0, P0[:, 2] - start.z_hi) / 0.05)[:, None]]
    # A stroke sends the shuttle towards the net, never back over the
    # hitter's own baseline. Without this, a flight coming at the camera has
    # a twin going up and away that the first frames can't tell apart.
    r.append((np.maximum(0.0, -toward * th[:, 4]) / FORWARD_SIGMA_MPS)[:, None])
    # The contact is known to about a frame; and the shuttle left before the first point after it.
    r.append((tau / LAUNCH_SIGMA_S)[:, None])
    r.append((np.maximum(0.0, tau - 0.8 * t_first) / 0.005)[:, None])
    return np.concatenate(r, axis=1)


def fit_early_flight(ts: np.ndarray, uv: np.ndarray, start: Anchor, cam, cfg_f, side: str, sigma_px: float = 3.0,
                     iters: int = 40, n_starts: int = 6) -> dict:
    """Fit a flight to its first moments only, then fly it on to the floor.

    The phase 7 inputs (`early_*` in `shots.csv`). Pinned at the start only —
    `start`, the hit within reach of the hitter, pulled towards their feet,
    as `fit_flight` has it; its pixel `start.uv` counts `EARLY_ANCHOR_WEIGHT`
    track points, and may be None (no pixel term: the feet and the launch
    moment place the start) — and fitted to the track
    points `ts, uv` (seconds from the contact frame). The caller passes only
    what the labelling clip shows, the start pixel included. Nothing about
    how the flight ended goes in: no next contact, no landing, no receiver,
    no net constraint — so the result is the same whether the shot was
    returned or not. Then the fitted path is flown on (gravity and drag)
    until it reaches the floor: where (`land`, court x, y) and when
    (`t_land`, from the contact frame), and the height where it crosses the
    net plane on the way (`net_z`; None if it comes down before the net).

    Seven unknowns: `P0, V0` and `tau`, when the shuttle left the racket
    relative to the contact frame (the contact is a rounded frame, good to a
    frame or two; a fixed launch at the frame left the first track points
    ~18 px off on a quarter of the real shots and sent returned shots short of
    the net). Two priors, the same for every shot: `tau` within about a frame
    (`LAUNCH_SIGMA_S`), and a stroke sends the shuttle towards the net
    (`FORWARD_SIGMA_MPS`; which way that is comes from the hitter's `side`,
    'near' or 'far' — not from the sign of their feet's court y, which flips
    at the net). The fit starts from a few guesses along the last track
    point's line of sight (`_early_guesses`) and keeps the best.

    With the end anchor gone, depth along the camera's line of sight comes
    from the dynamics alone: drag slows the shuttle at a rate set by its
    speed, gravity, perspective. For a flight going away from the camera
    (the near player's shots) those cues agree and pin the depth roughly; for
    one coming at it (the far player's) drag and the approach pull the image
    speed opposite ways and the depth is barely pinned — measured in
    `docs/features.md`. Across the court (lateral) it is pinned either way.
    """
    K, R, t = cam
    v_term = float(cfg_f.v_term_mps)
    ts, uv = np.asarray(ts, float), np.asarray(uv, float).reshape(-1, 2)
    toward = toward_net(side)  # the net is this way along y from the hitter
    t_first = float(ts.min()) if len(ts) else 0.04

    def resid(th):
        return _early_residuals(th, ts, uv, start, cam, cfg_f, sigma_px, toward, t_first)

    guesses = _early_guesses(ts, uv, start, cam, toward)
    fwd = toward * guesses[:, 4] >= 0
    if np.isfinite(FORWARD_SIGMA_MPS) and fwd.any():
        guesses = guesses[fwd]
    guesses = np.c_[guesses, np.zeros(len(guesses))]  # tau = 0: launched at the contact frame
    r0 = resid(guesses)
    order = np.argsort((r0 ** 2).sum(1))[:n_starts]
    best, best_cost = None, np.inf
    for k in order:
        theta, cost = _solve(guesses[k][None], resid, iters, EARLY_EPS)
        if cost < best_cost:
            best, best_cost = theta, cost
    P0, V0, tau = best[0, :3], best[0, 3:6], float(best[0, 6])
    px = project(K, R, t, simulate_rows(P0, V0, (ts - tau)[None], v_term)[0]) if len(ts) else np.zeros((0, 2))
    rms = float(np.sqrt(np.mean(np.sum((px - uv) ** 2, axis=1)))) if len(ts) else np.nan
    s, X, land, t_land = fly_to_floor(P0, V0, v_term)
    net, s_net = _crossing(s, X, 1)  # the path ends on the floor: a crossing after that never happens
    return {"P0": P0, "V0": V0, "tau": tau, "speed_mps": float(np.linalg.norm(V0)), "capped": at_speed_cap(V0),
            "rms_px": rms, "n_obs": int(len(ts)), "early_obs": int((ts <= float(cfg_f.early_s)).sum()), "cost": best_cost,
            "land": land, "t_land": None if t_land is None else t_land + tau,
            "net_x": None if net is None else float(net[0]), "net_z": None if net is None else float(net[2]),
            "s_net": None if s_net is None else s_net + tau, "apex_z": float(X[:, 2].max()), "path": X}
