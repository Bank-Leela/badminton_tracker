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


def _play_frame(width, height, seed):
    """Green mat with white court lines, plus a little noise so histograms are not degenerate."""
    rng = np.random.default_rng(seed)
    frame = np.full((height, width, 3), (60, 160, 60), dtype=np.uint8)
    noise = rng.integers(-8, 9, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(frame, (int(width * 0.2), int(height * 0.35)), (int(width * 0.8), int(height * 0.9)), (255, 255, 255), 1)
    cv2.line(frame, (int(width * 0.2), int(height * 0.6)), (int(width * 0.8), int(height * 0.6)), (255, 255, 255), 1)
    # A crowd band at the top, dark and busy, like a broadcast.
    frame[: int(height * 0.2)] = rng.integers(20, 90, size=(int(height * 0.2), width, 3), dtype=np.uint8)
    return frame


def _crowd_frame(width, height, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) // 2 + np.array([20, 20, 90], dtype=np.uint8)


@pytest.fixture
def broadcast_video(tmp_path):
    """Play view (0-89), crowd (90-149), play view (150-259) with a 3-frame white flash at 200-202."""
    path = tmp_path / "broadcast.mp4"
    width, height, fps = 640, 360, 30.0
    scenes = [("play", 0, 90), ("crowd", 90, 150), ("play", 150, 260)]
    flash = range(200, 203)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for kind, a, b in scenes:
        for i in range(a, b):
            if i in flash:
                frame = np.full((height, width, 3), 245, dtype=np.uint8)
            elif kind == "play":
                frame = _play_frame(width, height, i)
            else:
                frame = _crowd_frame(width, height, i)
            writer.write(frame)
    writer.release()
    return {
        "path": path, "fps": fps, "n_frames": 260,
        "cuts": [90, 150], "flash": (200, 203),
        "play_spans": [(0, 90), (150, 260)],
    }
