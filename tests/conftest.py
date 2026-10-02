import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path):
    """A short video with a bright dot moving on a dark background.

    Frame i has the dot at x = 20 + 4*i, which makes frame-index off-by-ones
    visible: the dot position identifies the frame.
    """
    path = tmp_path / "synthetic.mp4"
    width, height, n_frames, fps = 320, 180, 60, 30.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cv2.circle(frame, (20 + 4 * i, 90), 5, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return {"path": path, "width": width, "height": height, "n_frames": n_frames, "fps": fps}


WHITE = (255, 255, 255)


def _court_frame(width, height, seed, corners):
    """Green mat with white court lines inside `corners` (fractions x0, y0, x1, y1).

    Lines are 3 px at 640 wide, so they survive the 320-px line masks the way
    a broadcast's 4-6 px lines at 1080p do. A little noise keeps the mat from
    being a flat colour.
    """
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), (60, 160, 60), dtype=np.uint8)
    noise = rng.integers(-8, 9, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    x0, y0, x1, y1 = (int(c * s) for c, s in zip(corners, (width, height, width, height)))
    cv2.rectangle(frame, (x0, y0), (x1, y1), WHITE, 3)
    cv2.line(frame, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2), WHITE, 3)
    # A crowd band at the top, dark and busy, like a broadcast.
    frame[: int(height * 0.2)] = rng.integers(20, 90, size=(int(height * 0.2), width, 3), dtype=np.uint8)
    return frame


def _crowd_frame(width, height, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) // 2 + np.array([20, 20, 90], dtype=np.uint8)


@pytest.fixture
def broadcast_video(tmp_path):
    """A fake broadcast, 300 frames:

    - 0-89    play view (the main camera)
    - 90-149  crowd
    - 150-259 play view, with a 3-frame white flash at 200-202
    - 260-299 another camera: green mat and white lines too, but in a
              different layout — a replay angle, which is not play. Kept
              under half of the non-play frames, so its lines are not
              discarded as an overlay and the test sees the layout mismatch.
    Every frame carries a white score-graphic box in the top-left corner.
    """
    path = tmp_path / "broadcast.mp4"
    width, height, fps = 640, 360, 30.0
    scenes = [("play", 0, 90), ("crowd", 90, 150), ("play", 150, 260), ("other", 260, 300)]
    flash = range(200, 203)
    main_camera, other_camera = (0.2, 0.35, 0.8, 0.9), (0.05, 0.3, 0.6, 0.8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for kind, a, b in scenes:
        for i in range(a, b):
            if i in flash:
                frame = np.full((height, width, 3), 245, dtype=np.uint8)
            elif kind == "play":
                frame = _court_frame(width, height, i, main_camera)
            elif kind == "other":
                frame = _court_frame(width, height, i, other_camera)
            else:
                frame = _crowd_frame(width, height, i)
            cv2.rectangle(frame, (20, 20), (80, 40), WHITE, -1)
            writer.write(frame)
    writer.release()
    return {
        "path": path, "fps": fps, "n_frames": 300,
        "flash": (200, 203),
        "score_box": (20, 20, 80, 40),
        "play_spans": [(0, 90), (150, 260)],
        "scenes": scenes,
    }
