import cv2
import numpy as np
import pytest

from features import dist_from_singles_lines, net_height, own_frame
from video import FrameClock, repeated_frames


def test_own_frame_mirrors_the_far_end():
    # Near player facing the net: their right is +x. Far player facing the camera: their right is −x.
    assert own_frame(1.0, -3.0, "near") == (3.0, 1.0)
    assert own_frame(1.0, 3.0, "far") == (3.0, -1.0)


def test_net_height():
    assert net_height(0.0) == pytest.approx(1.524)
    assert net_height(3.05) == pytest.approx(1.55)
    assert net_height(-3.05) == pytest.approx(1.55)


def test_distance_from_the_singles_lines():
    assert dist_from_singles_lines(0.0, 6.0) == pytest.approx(0.70)  # 0.7 m inside the baseline
    assert dist_from_singles_lines(2.39, 0.0) == pytest.approx(0.20)  # 0.2 m inside the sideline
    assert dist_from_singles_lines(2.79, 0.0) == pytest.approx(-0.20)  # 0.2 m wide
    assert dist_from_singles_lines(2.89, 7.0) == pytest.approx(-np.hypot(0.3, 0.3))  # out past the corner


def test_frame_clock_skips_repeats():
    # 30 fps container, 25 fps content: one frame in six repeats.
    repeats = np.arange(5, 600, 6)
    clock = FrameClock(30.0, repeats, 600)
    assert clock.source_fps == 25.0
    assert clock.seconds(0, np.array([6])) == pytest.approx([5 / 25])  # frame 5 repeats 4: 6 frames, 5 unique steps
    assert clock.seconds(0, np.array([600]))[0] == pytest.approx(500 / 25)
    plain = FrameClock(25.0, np.array([], int), 600)
    assert plain.seconds(10, np.array([35]))[0] == pytest.approx(1.0)


def test_repeated_frames_are_found(tmp_path):
    path = tmp_path / "rep.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    src = 0
    for i in range(60):
        if i % 6 != 5:
            src += 1
        img = np.full((180, 320, 3), 40, np.uint8)
        cv2.circle(img, (20 + 6 * src, 90), 12, (255, 255, 255), -1)  # moves with the content, not the frame
        w.write(img)
    w.release()
    found = repeated_frames(path, [(0, 60)], threshold=0.5)
    assert found.tolist() == list(range(5, 60, 6))
