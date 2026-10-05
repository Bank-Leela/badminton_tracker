import numpy as np
import pytest

from config import load_config
from flight import Anchor, fit_flight, project, simulate

CFG = load_config().flight
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
