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
