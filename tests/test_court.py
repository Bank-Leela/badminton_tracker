import json

import cv2
import numpy as np
import pandas as pd
import pytest

from config import load_config
from court import (
    CORNERS,
    COURT_LENGTH,
    CROSS_LINES,
    DOUBLES_WIDTH,
    LONG_LINES,
    SINGLES_WIDTH,
    CourtCheckError,
    apply_manual_corners,
    check_dimensions,
    court_dimensions,
    court_points,
    detect_lines,
    homography_from_corners,
    line_mask,
    load_homographies,
    painted_segments,
    project,
    refine_court,
    solve_background,
    solve_homographies,
)

CFG_COURT = load_config().court
W, H = 1280, 720
# Doubles corners (near-left, near-right, far-right, far-left) of a broadcast-like view.
TRUE_CORNERS = [(300.0, 690.0), (980.0, 690.0), (790.0, 330.0), (490.0, 330.0)]
MOVED_CORNERS = [(320.0, 700.0), (1000.0, 695.0), (805.0, 345.0), (505.0, 342.0)]


def render_court(H_true, seed=0, players=True):
    """Green mat on a purple surround, white lines from `H_true`, broadcast clutter."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (150, 40, 160)
    mat = project(H_true, np.array([[-4.5, -8.2], [4.5, -8.2], [4.5, 8.2], [-4.5, 8.2]]))
    cv2.fillPoly(img, [np.round(mat).astype(np.int32)], (60, 160, 60))
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _, a, b in painted_segments():
        p, q = np.round(project(H_true, np.array([a, b]))).astype(int)
        cv2.line(img, tuple(p), tuple(q), (245, 245, 245), 3, cv2.LINE_AA)
    # Net tape, raised above the net line; a score box; ad text; some crowd.
    left, right = project(H_true, np.array([[-3.4, 0.0], [3.4, 0.0]]))
    cv2.line(img, (int(left[0]), int(left[1]) - 70), (int(right[0]), int(right[1]) - 70), (235, 235, 235), 3)
    cv2.rectangle(img, (40, 30), (260, 80), (255, 255, 255), -1)
    cv2.putText(img, "YONEX  NEW DELHI", (330, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (250, 250, 250), 2)
    img[:120, 280:] = rng.integers(0, 200, size=(120, W - 280, 3), dtype=np.uint8)
    if players:  # two dark figures somewhere on court
        for x, y in rng.uniform([-2.5, -6.0], [2.5, 6.0], size=(2, 2)):
            c = project(H_true, np.array([[x, y]]))[0]
            cv2.ellipse(img, (int(c[0]), int(c[1]) - 40), (14, 45), 0, 0, 360, (30, 30, 60), -1)
    return img


@pytest.fixture(scope="module")
def H_true():
    return homography_from_corners(TRUE_CORNERS)


def _max_err(H_fit, H_ref):
    pts = court_points(0.5)
    return float(np.abs(project(H_fit, pts) - project(H_ref, pts)).max())


def test_court_model_matches_bwf_dimensions():
    segs = painted_segments()
    ys = [p[1] for _, a, b in segs for p in (a, b)]
    xs = [p[0] for _, a, b in segs for p in (a, b)]
    assert max(ys) - min(ys) == pytest.approx(COURT_LENGTH)
    assert max(xs) - min(xs) == pytest.approx(DOUBLES_WIDTH)
    assert LONG_LINES["right_singles"] - LONG_LINES["left_singles"] == pytest.approx(SINGLES_WIDTH)
    assert CROSS_LINES["far_baseline"] - CROSS_LINES["far_long_service"] == pytest.approx(0.76)
    assert CROSS_LINES["far_short_service"] == pytest.approx(1.98)
    # The centre line stops at the short service lines.
    centre = [(a, b) for name, a, b in segs if name == "centre"]
    assert sorted(abs(p[1]) for a, b in centre for p in (a, b)) == pytest.approx([1.98, 1.98, 6.70, 6.70])


def test_homography_from_corners_round_trip(H_true):
    assert np.allclose(project(H_true, CORNERS), TRUE_CORNERS, atol=1e-3)


def test_detect_lines_finds_every_court_line(H_true):
    cross, longw = detect_lines(line_mask(render_court(H_true, players=False), CFG_COURT), CFG_COURT)

    def nearest(cands, a, b):
        p, q = project(H_true, np.array([a, b]))
        return min(max(abs(c @ np.r_[p, 1]), abs(c @ np.r_[q, 1])) for c in cands)

    for name, a, b in painted_segments():
        cands = cross if name in CROSS_LINES else longw
        assert nearest(cands, a, b) < 4, name


def test_fit_recovers_the_true_homography(H_true):
    fit = solve_background(render_court(H_true), CFG_COURT)
    assert _max_err(fit["court_to_image"], H_true) < 1.0
    assert fit["length_m"] == pytest.approx(COURT_LENGTH, abs=0.05)
    assert fit["width_m"] == pytest.approx(DOUBLES_WIDTH, abs=0.05)
    assert fit["cost_px"] < 0.5


def test_dimension_check_measures_the_image_not_the_model(H_true):
    """A homography that disagrees with the fitted image lines fails the check."""
    mask = line_mask(render_court(H_true, players=False), CFG_COURT)
    H, lines = refine_court(H_true, mask)
    length, width = court_dimensions(H, lines)
    check_dimensions(length, width, 0.2)  # the true one passes
    too_small = H @ np.diag([1.06, 1.06, 1.0])  # court drawn 6% too big: measures 6% short
    length, width = court_dimensions(too_small, lines)
    assert length == pytest.approx(COURT_LENGTH / 1.06, abs=0.05)
    with pytest.raises(CourtCheckError, match="length"):
        check_dimensions(length, width, 0.2)
    with pytest.raises(CourtCheckError, match="width"):
        check_dimensions(COURT_LENGTH, 5.18, 0.2)  # singles width is not the doubles court


def test_fit_fails_loudly_without_a_court():
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 120, size=(H, W, 3), dtype=np.uint8)
    with pytest.raises(CourtCheckError):
        solve_background(noise, CFG_COURT)


@pytest.fixture
def match_cache(tmp_path, H_true):
    """A static-camera video whose camera is bumped in the last play span, plus phase 2 output.

    0-59 play (true view), 60-69 crowd, 70-109 play (moved view).
    """
    path = tmp_path / "match.mp4"
    H_moved = homography_from_corners(MOVED_CORNERS)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    rng = np.random.default_rng(5)
    for i in range(110):
        if i < 60:
            frame = render_court(H_true, seed=i)
        elif i < 70:
            frame = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        else:
            frame = render_court(H_moved, seed=i)
        writer.write(frame)
    writer.release()
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}"])
    out = tmp_path / "cache" / "synthetic"
    out.mkdir(parents=True)
    pd.DataFrame({"segment_id": [0, 1, 2], "start_frame": [0, 60, 70], "end_frame": [60, 70, 110],
                  "is_play": [1, 0, 1], "score": [0.95, 0.02, 0.95]}).to_csv(out / "view_segments.csv", index=False)
    return {"cfg": cfg, "video": path, "out": out, "H_moved": H_moved}


def test_solve_homographies_end_to_end(match_cache, H_true, monkeypatch):
    cfg, video, out = match_cache["cfg"], match_cache["video"], match_cache["out"]
    doc = solve_homographies(cfg, "synthetic", video)
    for name in ["homography.json", "court_overlay.png", "court_background.png", "court_lines.png"]:
        assert (out / name).exists(), name
    assert [s["segment_id"] for s in doc["segments"]] == [0, 2]  # play spans only
    first, moved = doc["segments"]
    assert first["source"] == "auto:match"
    assert moved["source"] == "auto:span"  # the bumped camera was caught and refitted
    for seg, H_ref in [(first, H_true), (moved, match_cache["H_moved"])]:
        H_fit = np.linalg.inv(np.array(seg["image_to_court"]))
        assert _max_err(H_fit, H_ref) < 1.5
        assert seg["length_m"] == pytest.approx(COURT_LENGTH, abs=0.2)

    # image_to_court maps pixels to metres: the projected near-left corner is (-3.05, -6.70).
    rows = load_homographies(out / "homography.json")
    corner_px = project(H_true, CORNERS[:1])
    assert project(rows.H.iloc[0], corner_px)[0] == pytest.approx(CORNERS[0], abs=0.05)

    # Cached: a rerun returns the file without fitting.
    import court

    monkeypatch.setattr(court, "solve_background", lambda *a, **k: (_ for _ in ()).throw(AssertionError("refit")))
    again = solve_homographies(cfg, "synthetic", video)
    assert again["segments"] == json.loads((out / "homography.json").read_text())["segments"]


def test_manual_corners_are_refined_and_written(match_cache, H_true):
    cfg, video, out = match_cache["cfg"], match_cache["video"], match_cache["out"]
    clicked = np.array(TRUE_CORNERS) + np.array([[3, -2], [-2, 3], [2, 2], [-3, -1]])  # a sloppy click
    doc = apply_manual_corners(cfg, "synthetic", video, clicked)
    segs = {s["segment_id"]: s for s in doc["segments"]}
    assert set(segs) == {0, 2}
    assert segs[0]["source"] == "manual:refined"
    assert _max_err(np.linalg.inv(np.array(segs[0]["image_to_court"])), H_true) < 1.0
