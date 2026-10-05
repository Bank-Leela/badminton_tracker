import numpy as np
import pytest

import flight
from config import load_config
from flight import Anchor, V_CAP_MPS, at_speed_cap, fit_early_flight, fit_flight, fly_to_floor, project, simulate, toward_net

CFG = load_config().flight
CFG_FEATURES = load_config().features
FPS = 30.0


def look_at(eye, target, f=3000.0, size=(1920, 1080)):
    """A broadcast-like camera: `K, R, t` with z up in the world."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return K, R, -R @ eye


CAM = look_at(np.array([0.0, -25.0, 7.0]), np.array([0.0, 0.0, 1.0]))


def test_drag_gives_the_terminal_velocity():
    X = simulate(np.array([0, 0, 100.0]), np.array([0, 0, 0.0]), np.linspace(0, 6, 61), CFG.v_term_mps)[0]
    v_end = np.linalg.norm(X[-1] - X[-2]) / 0.1
    assert v_end == pytest.approx(CFG.v_term_mps, rel=0.02)


def shot(P0, V0, T, end_feet, start_feet, landing=False, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    K, R, t = CAM
    s = np.arange(1, int(T * FPS)) / FPS
    X = simulate(P0, V0, np.r_[0.0, s, T], CFG.v_term_mps)[0]
    uv = project(K, R, t, X) + rng.normal(0, noise, (len(X), 2))
    start = Anchor(0.0, uv[0], np.asarray(start_feet, float))
    end = Anchor(T, uv[-1], None if landing else np.asarray(end_feet, float))
    return s, uv[1:-1], start, end, X


@pytest.mark.parametrize("name, P0, V0, T, speed_tol", [
    # The receiver meets it at T; their feet ~0.5 m from the shuttle.
    ("smash", [0.5, -5.0, 2.8], [-1.0, 38.0, -10.0], 0.42, 0.10),
    ("clear", [-0.8, 4.0, 2.4], [0.5, -16.0, 14.0], 1.30, 0.10),
    ("drive", [1.2, -3.0, 1.5], [-0.5, 24.0, 1.5], 0.45, 0.10),
    # Short and slow: gravity says least about depth here.
    ("net shot", [0.3, -1.6, 0.9], [-0.2, 4.5, 3.0], 0.75, 0.20),
])
def test_fit_recovers_speed_and_net_height(name, P0, V0, T, speed_tol):
    P0, V0 = np.array(P0), np.array(V0)
    X_end = simulate(P0, V0, np.array([T]), CFG.v_term_mps)[0, 0]
    end_feet = X_end[:2] + [0.3, 0.4 * np.sign(X_end[1])]
    s, uv, start, end, X = shot(P0, V0, T, end_feet, P0[:2] + [0.1, -0.3 * np.sign(P0[1])])
    fit = fit_flight(s, uv, start, end, CAM, CFG)
    y = X[:, 1]
    i = np.flatnonzero(np.sign(y[:-1]) != np.sign(y[1:]))[0]
    true_net_z = X[i, 2] + (X[i + 1, 2] - X[i, 2]) * y[i] / (y[i] - y[i + 1])
    assert fit["speed_mps"] == pytest.approx(np.linalg.norm(V0), rel=speed_tol), name
    assert fit["net_z"] == pytest.approx(true_net_z, abs=0.15), name
    assert fit["rms_px"] < 3.0


def test_a_landing_ends_on_the_floor():
    P0, V0 = np.array([0.2, -4.0, 2.2]), np.array([0.8, 12.0, 6.0])
    # Find when it lands, then fit with a floor anchor there.
    ts = np.linspace(0, 3, 3001)
    X = simulate(P0, V0, ts, CFG.v_term_mps)[0]
    T = ts[np.argmax(X[:, 2] <= 0)]
    s, uv, start, end, X = shot(P0, V0, T, None, [0.2, -4.3], landing=True)
    fit = fit_flight(s, uv, start, end, CAM, CFG)
    assert fit["end"][2] == pytest.approx(0.0, abs=0.1)
    assert np.linalg.norm(fit["end"][:2] - X[-1, :2]) < 0.3


def test_a_returned_shot_is_kept_over_the_net():
    """A tight net shot clearing the tape by a few cm: returned, so the fit may not put it into the net."""
    P0, V0, T = np.array([0.2, -1.0, 1.1]), np.array([0.0, 3.0, 3.4]), 0.70  # crosses at 1.57 m, met at 0.9 m
    X_end = simulate(P0, V0, np.array([T]), CFG.v_term_mps)[0, 0]
    s, uv, start, end, X = shot(P0, V0, T, X_end[:2] + [0.2, 0.4], P0[:2] + [0.0, -0.4], noise=2.5, seed=3)
    fit = fit_flight(s, uv, start, end, CAM, CFG, over_net=True)
    assert fit["net_z"] >= 1.524 - 0.03
    assert fit["net_z"] == pytest.approx(1.573, abs=0.15)


# --- The early flight: the clip's frames only, flown on to the floor -------------------------------

def early_clip(P0, V0, n_frames=10, fps=25.0, noise=1.5, seed=0, launch_frames=0.4):
    """What the labelling clip shows of a shot: the contact pixel and `n_frames` track points after it.

    As on video, the shuttle leaves the racket between frames: `launch_frames`
    after the contact frame. The contact pixel is where it left the racket.
    """
    rng = np.random.default_rng(seed)
    K, R, t = CAM
    P0, V0 = np.asarray(P0, float), np.asarray(V0, float)
    s = np.arange(1, n_frames + 1) / fps
    X = simulate(P0, V0, s - launch_frames / fps, CFG.v_term_mps)[0]
    uv = project(K, R, t, X) + rng.normal(0, noise, (n_frames, 2))
    contact = project(K, R, t, P0[None])[0] + rng.normal(0, noise, 2)
    feet = P0[:2] + [0.1, -0.3 * np.sign(P0[1])]  # the feet ~0.3 m from the hit, as `shot`
    return s, uv, Anchor(0.0, contact, feet)


def side_of(P0) -> str:
    """The hitter's end: the near player's half is court y < 0."""
    return "near" if P0[1] < 0 else "far"


def test_fly_to_floor_ends_where_the_path_meets_the_floor():
    P0, V0 = np.array([0.2, -4.0, 2.2]), np.array([0.8, 12.0, 6.0])
    s, X, land, t_land = fly_to_floor(P0, V0, CFG.v_term_mps)
    fine = np.linspace(0, 3, 30001)
    Xf = simulate(P0, V0, fine, CFG.v_term_mps)[0]
    i = np.argmax(Xf[:, 2] <= 0)
    assert t_land == pytest.approx(fine[i], abs=2e-3)
    assert land == pytest.approx(Xf[i, :2], abs=0.01)
    assert X[-1, 2] == pytest.approx(0.0, abs=1e-9) and (X[:-1, 2] > 0).all()
    assert fly_to_floor(np.array([0, 0, -0.1]), V0, CFG.v_term_mps)[2] is None  # starts under the floor


@pytest.mark.parametrize("name, P0, V0, depth_tol", [
    # From the first 10 frames (0.4 s) only: where it would come down. No end anchor, no net constraint.
    # The tolerance is the median depth error over five noise draws; depth is
    # what one camera pins worst (docs/features.md has the real-video numbers).
    ("smash", [0.5, -5.0, 2.8], [-1.0, 38.0, -10.0], 0.4),
    ("drive", [1.2, -3.0, 1.5], [-0.5, 24.0, 1.5], 0.4),
    ("clear", [-0.8, -4.0, 2.4], [0.5, 16.0, 14.0], 0.5),
    ("drop", [0.3, -5.0, 2.5], [-0.5, 12.0, 0.0], 1.6),
    ("net shot", [0.3, -1.6, 0.9], [-0.2, 4.5, 3.0], 1.2),
    # Towards the camera: drag and the approach pull the image speed opposite
    # ways, and the first frames have a twin going up and away (ruled out by
    # the forward prior, `test_without_the_forward_prior...`).
    ("smash from the far end", [0.5, 5.0, 2.8], [-1.0, -38.0, -10.0], 1.2),
    ("clear from the far end", [-0.8, 4.0, 2.4], [0.5, -16.0, 14.0], 1.2),
    ("lift from the far end", [0.3, 2.0, 0.5], [-0.5, -10.0, 12.0], 1.2),
])
def test_early_fit_finds_the_landing_from_the_clip_alone(name, P0, V0, depth_tol):
    P0, V0 = np.array(P0), np.array(V0)
    _, _, land, t_land = fly_to_floor(P0, V0, CFG.v_term_mps)
    t_land += 0.4 / 25  # from the contact frame
    dy = []
    for seed in range(5):
        s, uv, start = early_clip(P0, V0, seed=seed)
        fit = fit_early_flight(s, uv, start, CAM, CFG, side_of(P0))
        assert fit["n_obs"] == 10 and fit["rms_px"] < 2.5, name
        assert not fit["capped"], name
        assert fit["land"][0] == pytest.approx(land[0], abs=0.15), name  # across the court: pinned
        assert fit["t_land"] == pytest.approx(t_land, abs=0.15), name
        dy.append(abs(fit["land"][1] - land[1]))
    assert np.median(dy) < depth_tol, (name, dy)


def test_the_launch_moment_is_fitted():
    """The shuttle leaves between frames. With a firm start pixel (weight 3, as contacts.csv's had), a launch
    fixed at the contact frame put a drive ~1.7 m off; the clip's own pixel counts less, but the launch moment
    is still fitted."""
    P0, V0 = np.array([1.2, -3.0, 1.5]), np.array([-0.5, 24.0, 1.5])
    _, _, land, _ = fly_to_floor(P0, V0, CFG.v_term_mps)
    s, uv, start = early_clip(P0, V0, launch_frames=0.4)
    fit = fit_early_flight(s, uv, start, CAM, CFG, "near")
    assert fit["tau"] * 25 == pytest.approx(0.4, abs=0.15)
    assert fit["land"][1] == pytest.approx(land[1], abs=0.4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(flight, "EARLY_ANCHOR_WEIGHT", 3.0)
        firm = fit_early_flight(s, uv, start, CAM, CFG, "near")
        mp.setattr(flight, "LAUNCH_SIGMA_S", 1e-6)
        fixed = fit_early_flight(s, uv, start, CAM, CFG, "near")
    assert firm["land"][1] == pytest.approx(land[1], abs=0.4)
    assert abs(fixed["land"][1] - land[1]) > 1.0


def test_early_fit_net_crossing_and_a_shot_into_the_net():
    # Over: crosses the net plane on the way, at the right height.
    P0, V0 = np.array([0.5, -5.0, 2.8]), np.array([-1.0, 38.0, -10.0])
    X = simulate(P0, V0, np.linspace(0, 0.5, 5001), CFG.v_term_mps)[0]
    i = np.argmax(X[:, 1] >= 0)
    s, uv, start = early_clip(P0, V0, noise=1.0)
    fit = fit_early_flight(s, uv, start, CAM, CFG, "near")
    assert fit["net_z"] == pytest.approx(X[i, 2], abs=0.15)
    # Short: comes down on the hitter's own side — no net crossing, the landing short of the net.
    P0, V0 = np.array([0.0, -3.0, 0.6]), np.array([0.0, 3.0, 2.5])
    _, _, land, _ = fly_to_floor(P0, V0, CFG.v_term_mps)
    assert land[1] < 0
    s, uv, start = early_clip(P0, V0, noise=1.0)
    fit = fit_early_flight(s, uv, start, CAM, CFG, "near")
    assert fit["net_z"] is None
    assert fit["land"][1] < 0 and fit["land"][1] == pytest.approx(land[1], abs=0.6)


def test_without_the_forward_prior_a_shot_at_the_camera_has_a_twin(monkeypatch):
    """Why the prior is there: noise-free, the far player's clear fits its first frames just as well going up
    and back over their own baseline."""
    P0, V0 = np.array([-0.8, 4.0, 2.4]), np.array([0.5, -16.0, 14.0])
    _, _, land, _ = fly_to_floor(P0, V0, CFG.v_term_mps)
    s, uv, start = early_clip(P0, V0, noise=0.0)
    with_prior = fit_early_flight(s, uv, start, CAM, CFG, "far")
    monkeypatch.setattr(flight, "FORWARD_SIGMA_MPS", np.inf)
    without = fit_early_flight(s, uv, start, CAM, CFG, "far")
    assert with_prior["land"][1] == pytest.approx(land[1], abs=0.5)
    assert without["rms_px"] < 1.0 and without["land"][1] - land[1] > 5.0  # fits the clip, lands behind the hitter


def test_early_fit_is_deterministic():
    """The fit has no end: the same clip gives the same flight, whatever happened next."""
    P0, V0 = np.array([1.2, -3.0, 1.5]), np.array([-0.5, 24.0, 1.5])
    s, uv, start = early_clip(P0, V0, seed=4)
    a = fit_early_flight(s, uv, start, CAM, CFG, "near")
    b = fit_early_flight(s.copy(), uv.copy(), Anchor(0.0, start.uv.copy(), start.feet.copy()), CAM, CFG, "near")
    for k in ("P0", "V0", "land"):
        assert np.array_equal(a[k], b[k])
    assert a["t_land"] == b["t_land"] and a["net_z"] == b["net_z"]


def test_which_way_is_forward_comes_from_the_hitters_side_not_their_feet():
    """A near-player net shot whose feet project just over the net (y > 0, as on 6 real shots): the
    forward prior must still point at the far side. Taken from the sign of the feet's y, it pointed back
    at the hitter's own baseline and sent the landing 8.6 m behind them."""
    assert toward_net("near") == 1.0 and toward_net("far") == -1.0
    with pytest.raises(ValueError):
        toward_net("left")
    P0, V0 = np.array([0.3, -0.3, 1.0]), np.array([-0.2, 4.0, 3.5])
    _, _, land, _ = fly_to_floor(P0, V0, CFG.v_term_mps)
    assert land[1] > 0
    right, wrong = [], []
    for seed in range(5):
        s, uv, start = early_clip(P0, V0, noise=1.0, seed=seed)
        start = Anchor(0.0, start.uv, np.array([0.4, 0.15]))  # the feet across the net
        right.append(fit_early_flight(s, uv, start, CAM, CFG, "near")["land"][1])
        wrong.append(fit_early_flight(s, uv, start, CAM, CFG, "far")["land"][1])  # what −sign(feet y) said
    assert np.median(np.abs(np.array(right) - land[1])) < 0.8, right
    assert sum(y < 0 for y in wrong) >= 3, wrong  # back on the hitter's own side


def test_without_a_start_pixel_the_feet_and_launch_moment_place_the_hit():
    """The shuttle unseen at the contact frame and the one before: no pixel anchor, the fit still runs."""
    P0, V0 = np.array([1.2, -3.0, 1.5]), np.array([-0.5, 24.0, 1.5])
    _, _, land, _ = fly_to_floor(P0, V0, CFG.v_term_mps)
    dy = []
    for seed in range(3):
        s, uv, start = early_clip(P0, V0, noise=1.0, seed=seed)
        fit = fit_early_flight(s, uv, Anchor(0.0, None, start.feet), CAM, CFG, "near")
        assert fit["rms_px"] < 2.5 and not fit["capped"]
        assert fit["land"][0] == pytest.approx(land[0], abs=0.3)
        dy.append(abs(fit["land"][1] - land[1]))
    assert np.median(dy) < 1.0, dy


def test_the_speed_cap():
    assert at_speed_cap(np.array([0.0, V_CAP_MPS, 0.0]))
    assert at_speed_cap(np.array([V_CAP_MPS, 1.0, 0.0]))
    assert not at_speed_cap(np.array([0.0, 0.99 * V_CAP_MPS, 0.0]))
    assert V_CAP_MPS * 3.6 >= CFG_FEATURES.max_shuttle_kmh  # a capped fit breaks the speed invariant too
