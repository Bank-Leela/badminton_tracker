import json

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import features
from config import load_config
from contacts import clean_track, meeting_point, shuttle_pieces
from features import (EARLY_CLEAN_BACK, EARLY_COLUMNS, clip_track, dist_from_receiver_court, dist_from_singles_lines,
                      early_flight_features, early_window, net_height, own_frame, shot_features, start_pixel)
from flight import fly_to_floor, project, simulate
from labeler import clip_range
from test_flight import CAM
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


# --- The early flight (phase 7's inputs) -----------------------------------------------------------

def test_dist_from_receiver_court():
    assert dist_from_receiver_court(0.0, 6.0) == pytest.approx(dist_from_singles_lines(0.0, 6.0))
    assert dist_from_receiver_court(2.79, 1.0) == pytest.approx(-0.20)
    assert dist_from_receiver_court(0.0, -0.4) == pytest.approx(-0.40)  # short of the net, on the hitter's side
    assert dist_from_receiver_court(2.89, -0.3) == pytest.approx(-np.hypot(0.3, 0.3))  # and wide


def test_early_window_is_the_clip():
    # labeler.clip_range shows frames contact+1 .. contact+after_frames, stopping `margin` short of the next event.
    assert early_window(100, None, 10, 2) == 110
    assert early_window(100, 140, 10, 2) == 110
    assert early_window(100, 113, 10, 2) == 110
    assert early_window(100, 108, 10, 2) == 105  # fast exchange: the reply's flight is never in it
    assert early_window(100, 102, 10, 2) == 100  # nothing to fit
    for nxt in (None, 140, 113, 108, 102):  # the same last frame as the clip
        _, end = clip_range(100, (0, 1000), 25.0, 1.5, 10, nxt, 2)
        assert early_window(100, nxt, 10, 2) == end - 1


FPS = 25.0
V_TERM = load_config().flight.v_term_mps
HIT = 100  # the shot under test: the near player's drive at frame 100
DRIVE = (np.array([1.2, -3.0, 1.5]), np.array([-0.5, 24.0, 1.5]))


def _flight(P0, V0, f0, f1):
    """Court points of a flight launched at frame `f0`, at frames f0..f1."""
    s = (np.arange(f0, f1 + 1) - f0) / FPS
    return simulate(np.asarray(P0, float), np.asarray(V0, float), s, V_TERM)[0]


def write_match(root, match_id, events, shuttle, far_at, contact_px=None):
    """A one-rally match cache `shot_features` can read, filmed by the test camera.

    `events`: (frame, kind, side, court xyz of the shuttle); `shuttle`: {frame: court xyz};
    `far_at(frame)`: the far player's feet. The near player stands at (1.3, -3.3). `contact_px`:
    {frame: pixel} for contacts.csv in place of the event's own projection.
    """
    d = root / match_id
    d.mkdir(parents=True)
    K, R, t = CAM
    size = [1920, 1080]
    (d / "camera.json").write_text(json.dumps({"image_size": size, "f": float(K[0, 0]),
                                               "segments": [{"segment_id": 0, "R": R.tolist(), "t": t.tolist()}]}))
    H = np.linalg.inv(K @ np.c_[R[:, 0], R[:, 1], t])  # image -> court, on the floor
    (d / "homography.json").write_text(json.dumps({"image_size": size, "segments": [
        {"segment_id": 0, "start_frame": 0, "end_frame": 400, "image_to_court": H.tolist()}]}))
    (d / "poses.meta.json").write_text(json.dumps({"fps": FPS, "video": "none.mp4"}))
    rows = []
    for k, (frame, kind, side, X) in enumerate(events):
        u, v = project(K, R, t, np.asarray(X, float)[None])[0]
        if contact_px and frame in contact_px:
            u, v = contact_px[frame]
        hit = kind != "landing"
        rows.append({"rally_id": 0, "shot_index": k + 1 if hit else np.nan, "kind": kind, "frame": frame,
                     "player_id": {"near": 0, "far": 1}[side] if hit else np.nan, "side": side if hit else np.nan,
                     "x": u, "y": v, "reach": 0.3 if hit else np.nan, "gap": 1.0 if hit else np.nan,
                     "miss_px": 5.0 if hit else np.nan})
    contacts = pd.DataFrame(rows)
    contacts.to_csv(d / "contacts.csv", index=False)
    a, b = int(contacts["frame"].min()) - 5, int(contacts["frame"].max()) + 30
    np.savez(d / "repeats.npz", spans=np.array([[a, b]], dtype=np.int64), repeats=np.zeros(0, np.int64),
             n_frames=b - a)
    fr = np.array(sorted(shuttle))
    uv = project(K, R, t, np.array([shuttle[f] for f in fr]))
    pd.DataFrame({"frame": fr, "x": uv[:, 0], "y": uv[:, 1], "visible": 1, "confidence": 0.9}).to_csv(
        d / "shuttle.csv", index=False)
    prow = []
    for f in range(40, 220):
        for pid, side, xy in ((0, "near", np.array([1.3, -3.3])), (1, "far", np.asarray(far_at(f), float))):
            prow.append({"frame": f, "player_id": pid, "side": side, "segment_id": 0, "court_x": xy[0],
                         "court_y": xy[1], "speed": 0.5, "y2": 800.0,
                         "keypoints": [[0.0, 0.0, 0.0]] * 17})  # no pose: the pose features stay empty
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(prow)), d / "players.parquet")


def _rally(root, match, next_kind, next_frame):
    """The drive at HIT, then either the receiver's reply at `next_frame` or the drive on the floor there.

    Up to `next_frame` the shuttle track is the same drive either way; after
    it, the reply's flight back (then its landing) or the shuttle lying still.
    The receiver walks across the court at 0.5 m/s.
    """
    P0, V0 = DRIVE
    path = _flight(P0, V0, HIT, next_frame)
    shuttle = {HIT + i: X for i, X in enumerate(path)}
    events = [(HIT, "hit", "near", path[0])]
    if next_kind == "reply":
        X1, V1 = path[-1], np.array([0.3, -14.0, 7.0])
        _, _, land, t_land = fly_to_floor(X1, V1, V_TERM)
        f_land = next_frame + int(np.ceil(t_land * FPS))
        back = _flight(X1, V1, next_frame, f_land - 1)
        shuttle.update({next_frame + i: X for i, X in enumerate(back)})
        shuttle.update({f: np.r_[land, 0.0] for f in range(f_land, f_land + 10)})
        events += [(next_frame, "hit", "far", X1), (f_land, "landing", None, np.r_[land, 0.0])]
    else:
        rest = np.r_[path[-1][:2], 0.0]  # a different next event is all this needs, not a real bounce
        shuttle.update({f: rest for f in range(next_frame, next_frame + 10)})
        events += [(next_frame, "landing", None, rest)]
    write_match(root, match, events, shuttle, lambda f: (0.6 + 0.5 * (f - HIT) / FPS, 4.2))


def _shot(cfg, match):
    df = shot_features(cfg, match)
    return df[df["frame"] == HIT].iloc[0]


def _same(a, b, col):
    return (pd.isna(a[col]) and pd.isna(b[col])) or a[col] == b[col]


@pytest.fixture
def early_cfg(tmp_path):
    # The construction is the point here, not the invariants: a one-rally match is never "systematic".
    return load_config(overrides=[f"paths.cache_dir={tmp_path}", "features.max_blanked_frac=1.0"])


def test_early_features_do_not_know_whether_the_shot_came_back(tmp_path, early_cfg):
    """Same contact, same early track; then a reply, or the floor. The early_* values must be identical."""
    _rally(tmp_path, "returned", "reply", 115)
    _rally(tmp_path, "floor", "landing", 119)
    a, b = _shot(early_cfg, "returned"), _shot(early_cfg, "floor")
    # The columns measured to the next event do differ: the test has teeth.
    assert (a["ended"], b["ended"]) == (0, 1) and b["landing_src"] == "floor" != a["landing_src"]
    assert a["flight_time"] != b["flight_time"]
    for c in EARLY_COLUMNS:
        assert _same(a, b, c), c
    assert a["early_fit_n_obs"] == early_cfg.features.early_frames
    for c in ("early_landing_depth", "early_landing_lateral", "early_dist_from_lines", "early_flight_time",
              "early_shot_length", "early_avg_speed", "early_cross_court", "early_net_clearance",
              "early_shuttle_speed", "early_opponent_dist", "early_opponent_toward"):
        assert np.isfinite(a[c]), c
    # And they describe the drive: where it would have come down, seen from the receiver's end.
    _, _, land, t_land = fly_to_floor(*DRIVE, V_TERM)
    assert a["early_landing_depth"] == pytest.approx(land[1], abs=0.6)
    assert a["early_landing_lateral"] == pytest.approx(-land[0], abs=0.3)  # the far player's right is −x
    assert a["early_flight_time"] == pytest.approx(t_land, abs=0.1)
    assert a["early_shuttle_speed"] == pytest.approx(np.linalg.norm(DRIVE[1]), rel=0.15)


def test_a_fast_exchange_cuts_the_window_before_the_next_event(tmp_path, early_cfg):
    """A reply 8 frames after the hit: the window stops 2 frames short of it, as the clip does — and a
    landing at that frame gives the very same early flight, though the track after it differs."""
    _rally(tmp_path, "fast_reply", "reply", 108)
    _rally(tmp_path, "fast_floor", "landing", 108)
    _rally(tmp_path, "slow", "landing", 119)
    a, b, c = _shot(early_cfg, "fast_reply"), _shot(early_cfg, "fast_floor"), _shot(early_cfg, "slow")
    assert a["early_fit_n_obs"] == 5 == b["early_fit_n_obs"]  # frames 101-105
    assert c["early_fit_n_obs"] == 10
    for col in EARLY_COLUMNS:
        assert _same(a, b, col), col
    assert np.isfinite(a["early_landing_depth"])


def _jitter(frame):
    """The same ~1 px of tracking noise on a frame whatever happens later (court m)."""
    return np.random.default_rng(int(frame)).normal(0, 0.01, 3)


def contacts_pixel(shuttle, frame, cfg):
    """contacts.csv's pixel for the hit at `frame`, as `contacts.label_rally` computes it: where the
    incoming and outgoing pieces' curves meet — the outgoing one fitted over its whole span."""
    K, R, t = CAM
    fr = np.array(sorted(shuttle), float)
    uv = project(K, R, t, np.array([shuttle[f] for f in fr]))
    c = cfg.contacts
    tt, x, y = clean_track(fr, uv[:, 0], uv[:, 1], float(c.top_px), float(c.jump_px))
    pieces = shuttle_pieces(tt, x, y, FPS, float(c.break_penalty_px2), int(c.min_points),
                            int(round(float(c.max_piece_s) * FPS)))
    for a, b in zip(pieces, pieces[1:]):
        f, pt, _ = meeting_point(a, b)
        if int(round(f)) == frame:
            return pt
    raise AssertionError(f"no break at frame {frame}: {[(p.t0, p.t1) for p in pieces]}")


def _rally_after(root, match, reply_frame, cfg):
    """The far player's shot coming in, the near player's drive at HIT, the far player's reply at
    `reply_frame` (after the early window either way); contacts.csv's pixel for HIT as contacts.py has it."""
    P0, V0 = DRIVE
    Q = np.array([-0.5, 4.0, 2.2])
    shuttle = {}
    for k in range(21):  # in: frames HIT-20 .. HIT, landing at the drive's start
        a = k / 20
        shuttle[HIT - 20 + k] = Q + (P0 - Q) * a + np.array([0.0, 0.0, 1.2 * 4 * a * (1 - a)])
    out = _flight(P0, V0, HIT, reply_frame)
    shuttle.update({HIT + i: X for i, X in enumerate(out)})
    X1, V1 = out[-1], np.array([0.4, -12.0, 8.0])
    _, _, land, t_land = fly_to_floor(X1, V1, V_TERM)
    f_land = reply_frame + int(np.ceil(t_land * FPS))
    shuttle.update({reply_frame + i: X for i, X in enumerate(_flight(X1, V1, reply_frame, f_land - 1))})
    shuttle = {f: X + _jitter(f) for f, X in shuttle.items()}
    events = [(HIT, "hit", "near", P0), (reply_frame, "hit", "far", X1), (f_land, "landing", None, np.r_[land, 0.0])]
    px = contacts_pixel(shuttle, HIT, cfg)
    write_match(root, match, events, shuttle, lambda f: (0.6 + 0.5 * (f - HIT) / FPS, 4.2), contact_px={HIT: px})
    return px


def test_early_features_never_see_the_track_after_the_window(tmp_path, early_cfg):
    """Two rallies identical up to the window's end (HIT + 10), then the reply at HIT + 13 or HIT + 17.

    contacts.csv's pixel for the hit is where the incoming and outgoing curves
    meet, the outgoing one fitted up to the reply: it differs between the two.
    The early_* columns may not."""
    assert early_window(HIT, HIT + 13, 10, 2) == early_window(HIT, HIT + 17, 10, 2) == HIT + 10
    pa = _rally_after(tmp_path, "reply13", HIT + 13, early_cfg)
    pb = _rally_after(tmp_path, "reply17", HIT + 17, early_cfg)
    assert np.hypot(*(pa - pb)) > 0.05  # the old start pixel moved with the flight after the clip
    a, b = _shot(early_cfg, "reply13"), _shot(early_cfg, "reply17")
    assert a["flight_time"] != b["flight_time"]  # the measured columns do differ
    for c in EARLY_COLUMNS:
        assert _same(a, b, c), c
    assert np.isfinite(a["early_landing_depth"]) and a["early_fit_n_obs"] == 10


def test_the_start_pixel_is_what_the_clip_shows():
    f = np.array([96.0, 97, 98, 99, 100, 101, 102, 103])
    x, y = f * 10, f * 5
    assert start_pixel(100, f, x, y).tolist() == [1000.0, 500.0]  # the contact frame
    keep = f != 100
    assert start_pixel(100, f[keep], x[keep], y[keep]).tolist() == [990.0, 495.0]  # else the frame before
    keep = (f != 100) & (f != 99)
    assert start_pixel(100, f[keep], x[keep], y[keep]) is None  # never the outgoing flight's next frame


def test_the_clip_track_is_cleaned_without_the_frames_after_it():
    """A lone far point on the window's last frame: the global clean drops it or not depending on the
    frame after the window; cleaned on the clip's own frames it is the same either way."""
    cfg = load_config()
    f = np.arange(80.0, 125)
    x, y = 500 + 5 * (f - 80), np.full(len(f), 600.0)
    x[f == 110] += 200  # a false detection on the window's last frame
    top, jump = float(cfg.contacts.top_px), float(cfg.contacts.jump_px)
    later = x.copy()
    later[f == 111] = x[f == 110]  # ...or the track after the window goes that way
    assert (clean_track(f, x, y, top, jump)[0] == 110).sum() != (clean_track(f, later, y, top, jump)[0] == 110).sum()
    a, b = clip_track(100, 110, f, x, y, top, jump), clip_track(100, 110, f, later, y, top, jump)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    assert a[0].min() == 100 - EARLY_CLEAN_BACK and a[0].max() == 110


def _fake_fit(**kw):
    """A fit_early_flight result: the near player's drive, landing 4.5 m into the far half."""
    fit = {"P0": np.array([1.2, -3.0, 1.5]), "V0": np.array([-0.5, 24.0, 1.5]), "tau": 0.0, "speed_mps": 24.1,
           "capped": False, "rms_px": 1.0, "n_obs": 10, "early_obs": 5, "cost": 1.0, "land": np.array([1.0, 4.5]),
           "t_land": 0.9, "net_x": 1.1, "net_z": 1.9, "s_net": 0.15, "apex_z": 1.9, "path": None}
    fit.update(kw)
    return fit


@pytest.mark.parametrize("name, fit, broke, kept", [
    ("fine", {}, None, EARLY_COLUMNS),
    ("speed known only roughly (few points in 0.2 s)", {"early_obs": 1}, None,
     [c for c in EARLY_COLUMNS if c != "early_shuttle_speed"]),
    # Broken fits: nothing from them is kept, whatever the points in the first 0.2 s.
    ("launched over 500 km/h", {"speed_mps": 150.0, "early_obs": 1}, "speed", ["early_fit_rms_px", "early_fit_n_obs"]),
    ("stopped at the solver's cap", {"speed_mps": 160.0, "capped": True}, "speed", ["early_fit_rms_px", "early_fit_n_obs"]),
    ("averaging over 500 km/h", {"t_land": 0.05}, "speed", ["early_fit_rms_px", "early_fit_n_obs"]),
    ("on the floor before the contact", {"t_land": -0.01}, "time", ["early_fit_rms_px", "early_fit_n_obs"]),
    ("landing off the court", {"land": np.array([1.0, 12.0])}, "position",
     ["early_shuttle_speed", "early_net_clearance", "early_fit_rms_px", "early_fit_n_obs"]),
])
def test_a_broken_early_fit_keeps_nothing(monkeypatch, name, fit, broke, kept):
    cfg = load_config()
    if name == "stopped at the solver's cap":  # broken even where the speed rule would let it through
        cfg = load_config(overrides=["features.max_shuttle_kmh=1000"])
    monkeypatch.setattr(features, "fit_early_flight", lambda *a, **k: _fake_fit(**fit))
    f = np.arange(95.0, 115)
    out, flags = early_flight_features({"frame": 100, "side": "near", "x": 0.0, "y": 0.0}, None, f, 500 + 10 * f,
                                       np.full(len(f), 500.0), FrameClock(FPS, np.zeros(0, int), 1000), "cam",
                                       np.array([1.3, -3.3]), np.array([0.5, 4.0]), np.array([0.5, 0.0]), "far",
                                       cfg, 1.0)
    assert sorted(out) == sorted(kept), name
    assert [k for k, v in flags.items() if v] == ([broke] if broke else []), name
