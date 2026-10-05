import cv2
import numpy as np
import pytest

from camera import (
    camera_centre,
    fit_focal,
    net_tape,
    pose_from_homography,
    project_points,
    standing_heights,
)
from config import load_config
from court import painted_segments

CFG = load_config()
W, H = 1920, 1080


def look_at(eye, target, f):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    return K, R, -R @ eye


TRUE_F = 3200.0
K0, R0, T0 = look_at(np.array([0.0, -24.0, 6.5]), np.array([0.0, 0.0, 0.5]), TRUE_F)
C2I = K0 @ np.c_[R0[:, 0], R0[:, 1], T0]  # floor (z = 0) to image


def render(with_net=True):
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), (60, 150, 60), np.uint8)
    img = np.clip(img.astype(int) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    for _, a, b in painted_segments():
        p, q = project_points(K0, R0, T0, np.array([[*a, 0], [*b, 0]])).round().astype(int)
        cv2.line(img, tuple(p), tuple(q), (245, 245, 245), 4, cv2.LINE_AA)
    if with_net:
        uv = project_points(K0, R0, T0, net_tape()).round().astype(int)
        cv2.polylines(img, [uv.reshape(-1, 1, 2)], False, (250, 250, 250), 4, cv2.LINE_AA)
    return img


def test_pose_from_homography_recovers_the_camera():
    K, R, t = pose_from_homography(C2I, TRUE_F, (W, H))
    X = np.array([[1.0, 2.0, 1.5], [-2.0, -4.0, 0.3]])
    assert project_points(K, R, t, X) == pytest.approx(project_points(K0, R0, T0, X), abs=0.5)
    assert camera_centre(R, t) == pytest.approx([0.0, -24.0, 6.5], abs=0.05)


def test_the_net_tape_fixes_the_focal_length():
    fit = fit_focal(render(), np.linalg.inv(C2I), CFG.camera, CFG.court)
    assert fit["f"] == pytest.approx(TRUE_F, rel=0.02)
    assert fit["tape_score"] > 0.5


def test_no_net_no_camera():
    from camera import CameraCheckError

    with pytest.raises(CameraCheckError, match="net tape"):
        fit_focal(render(with_net=False), np.linalg.inv(C2I), CFG.camera, CFG.court)


def test_standing_heights():
    feet = np.array([[0.5, -3.0], [-1.0, 4.0]])
    heads = project_points(K0, R0, T0, np.c_[feet, [1.6, 1.7]])
    assert standing_heights(K0, R0, T0, feet, heads) == pytest.approx([1.6, 1.7], abs=1e-6)
