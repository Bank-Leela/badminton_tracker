import json

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from config import load_config
from court import CORNERS, homography_from_corners, project
from players import (
    COLOUR_COLUMNS,
    KEYPOINTS,
    PLAYER_COLUMNS,
    PlayerCheckError,
    assign_identity,
    choose_players,
    clothing_colours,
    detect_poses,
    foot_points,
    keypoints_array,
    load_poses,
    movement_violations,
    player_speeds,
    rally_mask,
    savgol_coeffs,
    select_players,
    smooth_keypoints,
    to_court,
)

CFG = load_config()
W, H = 1280, 720
TRUE_CORNERS = [(300.0, 690.0), (980.0, 690.0), (790.0, 330.0), (490.0, 330.0)]
C2I = homography_from_corners(TRUE_CORNERS)  # court -> image
I2C = np.linalg.inv(C2I)
FPS = 30.0
RED, BLUE = (40, 40, 200), (200, 60, 30)  # BGR shirts; shorts are the other colour


def skeleton_at(foot_xy, height_px, conf=0.95):
    """A standing person's 17 keypoints with the ankle midpoint at `foot_xy`."""
    fx, fy = foot_xy
    w = 0.25 * height_px
    kp = np.zeros((17, 3), np.float32)
    rows = {  # (dx as a fraction of w, height above the feet as a fraction of height)
        "nose": (0, 0.93), "left_eye": (-0.05, 0.95), "right_eye": (0.05, 0.95),
        "left_ear": (-0.1, 0.94), "right_ear": (0.1, 0.94),
        "left_shoulder": (-0.5, 0.82), "right_shoulder": (0.5, 0.82),
        "left_elbow": (-0.6, 0.65), "right_elbow": (0.6, 0.65),
        "left_wrist": (-0.6, 0.5), "right_wrist": (0.6, 0.5),
        "left_hip": (-0.35, 0.52), "right_hip": (0.35, 0.52),
        "left_knee": (-0.3, 0.27), "right_knee": (0.3, 0.27),
        "left_ankle": (-0.3, 0.0), "right_ankle": (0.3, 0.0),
    }
    for i, name in enumerate(KEYPOINTS):
        dx, up = rows[name]
        kp[i] = (fx + dx * w, fy - up * height_px, conf)
    return kp


def box_of(kp, pad=6):
    return np.array([kp[:, 0].min() - pad, kp[:, 1].min() - pad, kp[:, 0].max() + pad, kp[:, 1].max() + 2])


def draw_person(img, kp, shirt, shorts):
    ls, rs, lh, rh, lk = (kp[KEYPOINTS.index(n)] for n in
                          ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee"])
    cv2.rectangle(img, (int(ls[0]), int(ls[1])), (int(rh[0]), int(rh[1])), shirt, -1)
    cv2.rectangle(img, (int(lh[0]), int(lh[1])), (int(rh[0]), int(lk[1])), shorts, -1)


# --- Geometry ---------------------------------------------------------------------------


def test_feet_are_the_ankle_midpoint_or_the_box_bottom():
    kp = np.stack([skeleton_at((600, 500), 150), skeleton_at((300, 400), 100)])
    kp[1, KEYPOINTS.index("left_ankle"), 2] = 0.2  # one ankle unsure: use the box
    boxes = np.stack([box_of(k) for k in kp])
    foot, ankles = foot_points(boxes, kp, 0.5)
    assert ankles.tolist() == [True, False]
    assert foot[0] == pytest.approx([600, 500])
    assert foot[1] == pytest.approx([(boxes[1, 0] + boxes[1, 2]) / 2, boxes[1, 3]])


def test_to_court_round_trips_and_blanks_the_sky():
    court = np.array([[0.0, 0.0], [-3.05, -6.7], [2.0, 5.0]])
    assert to_court(I2C, project(C2I, court)) == pytest.approx(court, abs=1e-6)
    horizon_y = project(C2I, np.array([[0.0, 1e6]]))[0, 1]  # vanishing point of the court's length
    above = to_court(I2C, np.array([[640.0, horizon_y - 50]]))
    assert np.isnan(above).all()
    # Same answer whatever the homography's scale, sign included.
    assert to_court(-3 * I2C, project(C2I, court)) == pytest.approx(court, abs=1e-6)


def test_clothing_colours_read_shirt_and_shorts():
    img = np.full((H, W, 3), (60, 160, 60), np.uint8)
    kp = skeleton_at((600, 600), 200)
    draw_person(img, kp, RED, BLUE)
    lab = clothing_colours(img, kp[None], 0.5)[0]
    expect = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0] for c in (RED, BLUE)]
    assert lab[:3] == pytest.approx(expect[0], abs=2)
    assert lab[3:] == pytest.approx(expect[1], abs=2)
    kp[KEYPOINTS.index("left_hip"), 2] = 0.1
    assert np.isnan(clothing_colours(img, kp[None], 0.5)).all()  # no hips, no torso, no shorts


# --- Choosing the players -----------------------------------------------------------------


def _placed(rows):
    """Candidates from court positions; boxes and feet in the image from the test homography."""
    df = pd.DataFrame(rows, columns=["frame", "segment_id", "track_id", "conf", "edge_x", "edge_y"])
    df["court_x"], df["court_y"] = df["edge_x"], df["edge_y"]
    feet = project(C2I, df[["court_x", "court_y"]].to_numpy())
    df["foot_x"], df["foot_y"] = feet[:, 0], feet[:, 1]
    h = np.where(df["court_y"] < 0, 160, 95)  # nearer is bigger
    df["x1"], df["x2"] = feet[:, 0] - h / 4, feet[:, 0] + h / 4
    df["y1"], df["y2"] = feet[:, 1] - h, feet[:, 1] + 2
    df["body_x1"], df["body_y1"], df["body_x2"] = df["x1"], df["y1"] + 0.15 * h, df["x2"]
    df["foot_src"], df["ankle_conf"] = "ankles", 0.95
    return df


def test_choose_players_skips_officials_and_passers_by():
    rows = []
    for f in range(100):
        rows += [(f, 1, 1, 0.9, 0.5, -3.0),  # near player
                 (f, 1, 2, 0.8, 0.0, 4.0),  # far player
                 (f, 1, 3, 0.95, 5.4, 0.5),  # umpire: beyond the side margin
                 (f, 1, 4, 0.9, -1.0, 9.6)]  # line judge behind the far baseline
        if 40 <= f < 55:
            rows.append((f, 1, 5, 0.97, 1.0, -2.0))  # someone crossing the near half, briefly
    chosen = choose_players(_placed(rows), CFG.players)
    assert len(chosen) == 200
    assert set(chosen.loc[chosen.side == "near", "track_id"]) == {1}
    assert set(chosen.loc[chosen.side == "far", "track_id"]) == {2}


def test_far_feet_hidden_by_the_near_player_are_blanked():
    rows = []
    for f in range(20):
        rows.append((f, 1, 1, 0.9, 0.0, -2.0))  # near player
        rows.append((f, 1, 2, 0.9, 1.0 if f < 10 else 0.0, 4.0))  # far player walks in behind them
    chosen = choose_players(_placed(rows), CFG.players)
    near_box = chosen[(chosen.side == "near") & (chosen.frame == 15)].iloc[0]
    # The pose model guesses the hidden ankles on the near player's legs, a
    # little unsure of them in frames 10-14; in 15-19 it still sees them clearly.
    far = chosen.index[(chosen.side == "far") & (chosen.frame >= 10)]
    chosen.loc[far, "foot_y"] = near_box.y2 - 30
    chosen.loc[far, "ankle_conf"] = np.where(chosen.loc[far, "frame"] < 15, 0.7, 0.95)
    from players import mark_occluded

    out = mark_occluded(chosen, CFG.players.hidden_ankle_conf)
    hidden = out[out.foot_src == "occluded"]
    assert sorted(hidden.frame) == list(range(10, 15)) and set(hidden.side) == {"far"}
    assert hidden[["court_x", "court_y"]].isna().all().all()
    assert out.loc[out.side == "near", "court_y"].notna().all()
    assert out.loc[(out.side == "far") & (out.frame >= 15), "court_y"].notna().all()


BREAK = int(60 * FPS)  # a game break: long enough to change ends in
POINT = int(10 * FPS)  # between points: too short


def _spans(n_spans, frames=60, gaps=None):
    """Play spans of `frames` each; `gaps[i]` frames of non-play before span i (default: breaks)."""
    gaps = gaps or [BREAK] * n_spans
    rows, f = [], 0
    for i in range(n_spans):
        rows.append((2 * i, f, f + gaps[i], 0))
        rows.append((2 * i + 1, f + gaps[i], f + gaps[i] + frames, 1))
        f += gaps[i] + frames
    return pd.DataFrame(rows, columns=["segment_id", "start_frame", "end_frame", "is_play"])


def _chosen(views, near_colours, rng):
    rows = []
    for (span, (cn, cf)) in zip(views[views.is_play == 1].itertuples(), near_colours):
        for f in range(span.start_frame, span.end_frame):
            for side, c in (("near", cn), ("far", cf)):
                rows.append([f, span.segment_id, side, *(np.array(c, float) + rng.normal(0, 3, 6))])
    return pd.DataFrame(rows, columns=["frame", "segment_id", "side", *COLOUR_COLUMNS])


A = [120, 180, 160, 40, 128, 128]  # red shirt, black shorts
B = [100, 140, 90, 230, 128, 128]  # blue shirt, white shorts


def test_identity_follows_the_change_of_ends():
    rng = np.random.default_rng(0)
    views = _spans(6)
    chosen = _chosen(views, [(A, B), (A, B), (B, A), (B, A), (B, A), (A, B)], rng)
    near_id, report = assign_identity(chosen, views, FPS, CFG.players)
    assert list(near_id.values()) == [0, 0, 1, 1, 1, 0]
    assert [s["segment_id"] for s in report["switches"]] == [5, 11]
    assert report["separation"] > 100


def test_identity_ignores_one_span_of_bad_colour():
    """A span lit badly (both players grey) inside a stretch of play changes nothing."""
    rng = np.random.default_rng(1)
    views = _spans(8, gaps=[BREAK, POINT, POINT, POINT, BREAK, POINT, POINT, POINT])
    murky = [128, 128, 128, 128, 128, 128]
    chosen = _chosen(views, [(A, B), (A, B), (murky, murky), (A, B), (B, A), (B, A), (B, A), (B, A)], rng)
    near_id, report = assign_identity(chosen, views, FPS, CFG.players)
    assert list(near_id.values()) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [s["segment_id"] for s in report["switches"]] == [9]


def test_no_change_of_ends_is_kept():
    """A retirement in game 1: nobody changes ends, whatever the breaks."""
    rng = np.random.default_rng(4)
    views = _spans(5)
    near_id, report = assign_identity(_chosen(views, [(A, B)] * 5, rng), views, FPS, CFG.players)
    assert set(near_id.values()) == {0} and report["switches"] == []


def test_ends_change_only_in_a_break():
    """Colours that flip after a 10 s gap are noise, not a change of ends: the
    stretch between breaks is judged as a whole."""
    rng = np.random.default_rng(2)
    views = _spans(6, gaps=[BREAK, POINT, POINT, BREAK, POINT, POINT])
    chosen = _chosen(views, [(A, B), (B, A), (A, B), (B, A), (B, A), (B, A)], rng)
    near_id, report = assign_identity(chosen, views, FPS, CFG.players)
    assert list(near_id.values()) == [0, 0, 0, 1, 1, 1]
    assert [s["segment_id"] for s in report["switches"]] == [7] and report["breaks"] == 1


def test_far_end_colour_shift_is_learned():
    """The far end reads 60 L darker and redder for both players (lighting,
    boards behind them): still two players, changing ends twice."""
    rng = np.random.default_rng(3)
    far = lambda c: list(np.array(c) + [-60, 15, 5, -10, 5, 0])
    A2 = [150, 130, 128, 120, 135, 130]  # dressed alike: only shorts and shade differ
    B2 = [175, 128, 126, 70, 145, 124]
    views = _spans(4)
    chosen = _chosen(views, [(A2, far(B2)), (B2, far(A2)), (B2, far(A2)), (A2, far(B2))], rng)
    near_id, report = assign_identity(chosen, views, FPS, CFG.players)
    assert list(near_id.values()) == [0, 1, 1, 0]
    assert report["far_shift"][0] == pytest.approx(-60, abs=5)


# --- Speed ---------------------------------------------------------------------------------


def _track(xy, segment_id=1, player_id=0, f0=0):
    xy = np.asarray(xy, float)
    return pd.DataFrame({"frame": np.arange(f0, f0 + len(xy)), "segment_id": segment_id, "player_id": player_id,
                         "court_x": xy[:, 0], "court_y": xy[:, 1]})


def test_speed_of_a_steady_run_is_measured_without_lag():
    t = np.arange(90) / FPS
    df = _track(np.c_[3.0 * t, np.zeros_like(t)])  # 3 m/s along x
    v = player_speeds(df, FPS, CFG.players)
    assert np.nanmedian(v) == pytest.approx(3.0, rel=0.01)
    assert np.isnan(v[:3]).all() and np.isfinite(v[10:80]).all()


def test_one_bad_frame_does_not_make_a_sprint():
    t = np.arange(90) / FPS
    xy = np.c_[np.zeros_like(t), -3 + 0.5 * t]
    xy[45] += [0.0, 1.5]  # one frame of keypoints on the wrong spot
    df = _track(xy)
    df["speed"] = player_speeds(df, FPS, CFG.players)
    assert df["speed"].max() < 1.0


def _violations(xy):
    df = _track(xy)
    df["speed"] = player_speeds(df, FPS, CFG.players)
    return df, movement_violations(df, FPS, CFG.players)


def test_a_jump_and_a_sprint_are_play():
    """Both read over the plan's 4 m/s; neither is an error."""
    t = np.arange(150) / FPS
    hop = 1.2 * np.exp(-((t - 2) / 0.1) ** 2)  # far-end jump: feet read 1.2 m further away and back
    df, bad = _violations(np.c_[0.2 * t, 3 + hop])
    assert df["speed"].max() > CFG.players.report_speed_mps
    assert bad.empty
    # Back court to the net: 4 m in 1 s, peaking at 6 m/s.
    y = -6 + 4 / (1 + np.exp(-(t - 2.5) / 0.17))
    df, bad = _violations(np.c_[np.zeros_like(t), y])
    assert df["speed"].max() > CFG.players.report_speed_mps
    assert bad.empty


def test_a_swap_to_someone_else_is_a_step():
    t = np.arange(120) / FPS
    _, bad = _violations(np.c_[np.zeros_like(t), np.where(t < 2, -3.0, -5.5)])
    assert bad[["kind", "start_frame", "value"]].values.tolist() == [["step", 59, 2.5]]


def test_flickering_between_two_people_is_steps():
    t = np.arange(120) / FPS
    _, bad = _violations(np.c_[np.zeros_like(t), np.where((np.arange(120) // 5) % 2 == 0, -3.0, -5.0)])
    assert (bad["kind"] == "step").sum() >= 20


def test_too_fast_for_too_long_is_sustained():
    """No single jump, but 7 m/s for over a second: a wrong scale or a drifting track."""
    t = np.arange(120) / FPS
    _, bad = _violations(np.c_[7.0 * t - 7, np.zeros_like(t)])
    assert set(bad["kind"]) == {"sustained"} and bad["value"].max() == pytest.approx(7.0, abs=0.05)


def test_rally_mask():
    segs = pd.DataFrame({"segment_id": [1, 1, 1, 3], "start_frame": [0, 10, 30, 50], "end_frame": [10, 30, 40, 70],
                         "is_play": [1, 1, 1, 1], "rally_id": [-1, 0, -1, 1]})
    m = rally_mask(np.array([0, 9, 10, 29, 30, 45, 50, 69, 70]), segs)
    assert m.tolist() == [False, False, True, True, False, False, True, True, False]


# --- Keypoint smoothing ------------------------------------------------------------------


def test_savgol_weights():
    assert savgol_coeffs(5, 2) * 35 == pytest.approx([-3, 12, 17, 12, -3])
    with pytest.raises(ValueError):
        savgol_coeffs(4, 2)


def test_smoothing_keeps_a_swing_peak_where_it_was():
    """A wrist swing: a 0.1 s burst of speed, with 1.5 px keypoint noise.

    The smoothed speed peak must stay on the true frame (±1) and keep most of
    its height, while the frame-to-frame noise drops. This is the check the
    plan asks for on the smoothing window.
    """
    rng = np.random.default_rng(3)
    n = 90
    t = np.arange(n)
    true_x = 400 + 120 / (1 + np.exp(-(t - 45) / 1.5))  # 120 px swing, fastest at frame 45
    kp = np.zeros((n, 17, 3))
    kp[:, :, 2] = 0.9
    kp[:, :, 0] = true_x[:, None] + rng.normal(0, 1.5, (n, 17))
    kp[:, :, 1] = 300 + rng.normal(0, 1.5, (n, 17))
    sm = smooth_keypoints(kp, t)
    wrist = KEYPOINTS.index("right_wrist")
    true_v = np.diff(true_x)
    v_sm = np.diff(sm[:, wrist, 0])
    v_raw = np.diff(kp[:, wrist, 0])
    assert abs(int(np.argmax(v_sm)) - int(np.argmax(true_v))) <= 1
    assert v_sm.max() == pytest.approx(true_v.max(), rel=0.2)
    still = slice(0, 30)
    assert np.std(v_sm[still]) < 0.6 * np.std(v_raw[still])


def test_smoothing_never_invents_points():
    t = np.arange(40)
    kp = np.zeros((40, 17, 3))
    kp[:, :, 0], kp[:, :, 1], kp[:, :, 2] = t[:, None] * 2.0, 100.0, 0.9
    kp[20, 3, 2] = 0.1  # an unsure ear
    sm = smooth_keypoints(kp, t)
    assert np.isnan(sm[20, 3]).all()
    assert sm[21, 3, 0] == pytest.approx(42.0)  # neighbours bridge it: a line stays a line
    assert np.isfinite(sm[:, 0]).all()


# --- End to end, with a stand-in for the network ---------------------------------------------


class FakeResultsModel:
    """Stands in for YOLO: returns the scenario's people for each frame it is fed, in order."""

    def __init__(self, scenario, frame_order):
        self.scenario = scenario
        self.order = iter(frame_order)

    def predict(self, frames, **kw):
        from ultralytics.engine.results import Results

        out = []
        for img in frames:
            people = self.scenario(next(self.order))
            kp = np.stack([p["kp"] for p in people]).astype(np.float32)
            boxes = np.array([[*box_of(p["kp"]), p["conf"], 0] for p in people], np.float32)
            out.append(Results(img, path="fake", names={0: "person"},
                               boxes=torch.from_numpy(boxes), keypoints=torch.from_numpy(kp)))
        return out


def scenario(frame):
    """Near and far player plus the umpire. They change ends at frame 80."""
    swapped = frame >= 80
    near_xy = (0.5 + 0.6 * np.sin(frame / 15), -3.5)
    far_xy = (-0.3, 4.0 + 0.5 * np.cos(frame / 20))
    near = {"kp": skeleton_at(project(C2I, np.array([near_xy]))[0], 160), "conf": 0.9,
            "colours": (BLUE, RED) if swapped else (RED, BLUE)}
    far = {"kp": skeleton_at(project(C2I, np.array([far_xy]))[0], 95), "conf": 0.85,
           "colours": (RED, BLUE) if swapped else (BLUE, RED)}
    umpire = {"kp": skeleton_at(project(C2I, np.array([[5.2, 0.3]]))[0], 120), "conf": 0.9,
              "colours": ((20, 20, 20), (20, 20, 20))}
    return [near, far, umpire]


@pytest.fixture
def match(tmp_path, monkeypatch):
    """Frames 0-59 and 80-139 are play (the players change ends between); 60-79 is not."""
    path = tmp_path / "match.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(140):
        img = np.full((H, W, 3), (60, 150, 60), np.uint8)
        for p in scenario(f):
            draw_person(img, p["kp"], *p["colours"])
        writer.write(img)
    writer.release()
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}", "device=cpu",
                                 f"paths.pose_model={path}",  # any existing file: the model is faked
                                 "players.min_break_s=0.5"])  # the 20-frame gap stands in for a game break
    out = tmp_path / "cache" / "m"
    out.mkdir(parents=True)
    pd.DataFrame({"segment_id": [0, 1, 2], "start_frame": [0, 60, 80], "end_frame": [60, 80, 140],
                  "is_play": [1, 0, 1], "score": [0.9, 0.1, 0.9]}).to_csv(out / "view_segments.csv", index=False)
    pd.DataFrame({"segment_id": [0, 0, 1, 2, 2], "start_frame": [0, 5, 60, 80, 85],
                  "end_frame": [5, 60, 80, 85, 140], "is_play": [1, 1, 0, 1, 1],
                  "rally_id": [-1, 0, -1, -1, 1]}).to_csv(out / "segments.csv", index=False)
    seg = lambda sid, a, b: {"segment_id": sid, "start_frame": a, "end_frame": b, "image_to_court": I2C.tolist(),
                             "source": "auto:match", "cost_px": 0.1, "length_m": 13.4, "width_m": 6.1}
    (out / "homography.json").write_text(json.dumps({"image_size": [W, H], "segments": [seg(0, 0, 60), seg(2, 80, 140)]}))
    import players

    order = list(range(0, 60)) + list(range(80, 140))
    monkeypatch.setattr(players, "load_pose_model", lambda cfg: FakeResultsModel(scenario, order))
    return {"cfg": cfg, "video": path, "out": out}


def test_detect_then_select_end_to_end(match):
    cfg, out = match["cfg"], match["out"]
    detect_poses(cfg, "m", match["video"])
    poses, kp = load_poses(out / "poses.parquet")
    assert len(poses) == 3 * 120  # everyone in the region, tracked, both spans
    assert poses.groupby("segment_id")["track_id"].nunique().tolist() == [3, 3]
    assert kp.shape == (360, 17, 3)

    players = select_players(cfg, "m")
    assert list(players.columns) == PLAYER_COLUMNS
    assert len(players) == 2 * 120  # the umpire is not a player
    assert players.groupby("frame").size().eq(2).all()
    # Player 0 starts near in red, changes ends at frame 80 and stays in red.
    p0 = players[players.player_id == 0]
    assert set(p0.loc[p0.frame < 60, "side"]) == {"near"}
    assert set(p0.loc[p0.frame >= 80, "side"]) == {"far"}
    meta = json.loads((out / "players.meta.json").read_text())
    assert [s["frame"] for s in meta["identity"]["switches"]] == [80]
    # Feet on court where the scenario put them.
    row = players[(players.frame == 30) & (players.side == "near")].iloc[0]
    assert (row.court_x, row.court_y) == pytest.approx((0.5 + 0.6 * np.sin(2), -3.5), abs=0.03)
    assert keypoints_array(players).shape == (240, 17, 3)
    assert players["speed"].max() == pytest.approx(0.6 / 15 * FPS, abs=0.1)  # the near player's top speed

    # Cached: unchanged inputs are reused; changed play spans stop detection.
    detect_poses(cfg, "m", match["video"])
    views = pd.read_csv(out / "view_segments.csv")
    views.loc[2, "end_frame"] = 130
    views.to_csv(out / "view_segments.csv", index=False)
    with pytest.raises(SystemExit, match="play spans"):
        detect_poses(cfg, "m", match["video"])


def test_overlay_renders_the_stretch(match):
    from players import write_players_overlay
    from video import probe_video

    cfg, out = match["cfg"], match["out"]
    detect_poses(cfg, "m", match["video"])
    select_players(cfg, "m")
    path = write_players_overlay(cfg, "m", match["video"], out / "o.mp4", 50, 90)
    assert probe_video(path).frame_count == 40


def _select_with(match, monkeypatch, damage):
    """Run select with `damage(chosen)` applied to the chosen players' positions."""
    cfg = match["cfg"]
    detect_poses(cfg, "m", match["video"])
    import players

    real = players.choose_players
    monkeypatch.setattr(players, "choose_players", lambda placed, cfg_sel: damage(real(placed, cfg_sel)))
    return select_players(cfg, "m", force=True)


def test_one_bad_spot_is_blanked_and_passes(match, monkeypatch):
    """From frame 30 the near "player" reads 2 m further back: one step, blanked."""
    def swap(chosen):
        chosen.loc[(chosen.side == "near") & chosen.frame.between(30, 59), "court_y"] -= 2.0
        return chosen

    players = _select_with(match, monkeypatch, swap)
    meta = json.loads((match["out"] / "players.meta.json").read_text())
    assert [v["kind"] for v in meta["violations"]] == ["step"]
    pad = round(CFG.players.blank_s * FPS)
    near = players[players.side == "near"].set_index("frame")
    step = meta["violations"][0]["start_frame"]
    blanked = near.index[near.foot_src == "flagged"]
    assert blanked.min() == step - pad and blanked.max() == step + 1 + pad
    assert near.loc[blanked, ["court_x", "court_y"]].isna().all().all()
    assert near.loc[blanked, "speed"].isna().all()
    assert near.drop(blanked)["court_y"].notna().all()  # everything else kept


def test_many_bad_spots_fail(match, monkeypatch):
    """The near "player" flicks to someone 2 m away and back, every 5 frames."""
    def flicker(chosen):
        hit = (chosen.side == "near") & chosen.frame.between(10, 59) & ((chosen.frame // 5) % 2 == 1)
        chosen.loc[hit, "court_y"] -= 2.0
        return chosen

    with pytest.raises(PlayerCheckError, match="systematic"):
        _select_with(match, monkeypatch, flicker)
