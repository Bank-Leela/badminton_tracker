import http.client
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from config import load_config
from labeler import (
    CLASSES,
    LabelServer,
    LabelStore,
    MatchData,
    build_queue,
    clip_range,
    latest_labels,
    load_labels,
    render_clip,
)

FPS, W, H, N = 30.0, 640, 360, 200
SPANS = [(0, 100, 1), (100, 120, 0), (120, 200, 1)]  # play, a cut away, play
BOX = {"near": [100.0, 50.0, 160.0, 250.0], "far": [300.0, 40.0, 340.0, 120.0]}  # both players, apart
SHOTS = [(0, 1, 30, "near"), (0, 2, 60, "far"), (0, 3, 95, "near"), (1, 1, 125, "far"), (1, 2, 160, "near")]


def marker_x(i):
    return 10 + 3 * i  # the frame index, readable back off any frame


@pytest.fixture
def match(tmp_path):
    video = tmp_path / "m.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        img = np.full((H, W, 3), 30, np.uint8)
        cv2.rectangle(img, (marker_x(i) - 1, 0), (marker_x(i) + 1, H - 1), (255, 255, 255), -1)
        w.write(img)
    w.release()
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}", f"paths.labels_dir={tmp_path / 'labels'}",
                                 "labeler.clip_width=640"])
    out = tmp_path / "cache" / "m"
    out.mkdir(parents=True)
    (out / "poses.meta.json").write_text(json.dumps({"video": str(video), "fps": FPS}))
    pd.DataFrame([(k, a, b, p) for k, (a, b, p) in enumerate(SPANS)],
                 columns=["segment_id", "start_frame", "end_frame", "is_play"]).to_csv(out / "view_segments.csv", index=False)
    pd.DataFrame([{"match_id": "m", "rally_id": r, "shot_index": s, "frame": f, "hitter_id": int(side == "far"),
                   "hitter_side": side, "is_serve": int(s == 1)} for r, s, f, side in SHOTS]).to_csv(out / "shots.csv", index=False)
    rows = [{"frame": f, "side": side, **dict(zip(("x1", "y1", "x2", "y2"), BOX[side]))}
            for f in range(N) for side in ("near", "far")]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), out / "players.parquet")
    return {"cfg": cfg, "video": video, "out": out}


def frame_shown(jpg: bytes) -> int:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    x = np.flatnonzero(img[H // 2] > 128).mean()  # the marker's centre, robust to JPEG ringing
    return int(round((x - 10) / 3))


def test_clip_range_stops_after_the_hit_and_stays_in_the_span():
    assert clip_range(60, (0, 100), FPS, 1.5, 10) == (15, 71)
    assert clip_range(95, (0, 100), FPS, 1.5, 10) == (50, 100)  # a cut comes first
    assert clip_range(125, (120, 200), FPS, 1.5, 10) == (120, 136)  # the span starts late
    with pytest.raises(ValueError):
        clip_range(110, (0, 100), FPS, 1.5, 10)


def test_rendered_clips_show_exactly_the_right_frames(match):
    md = MatchData(match["cfg"], "m")
    queue = {r["key"]: r for r in build_queue(match["cfg"], ["m"])}
    for r, s, f, side in SHOTS:
        clip = render_clip(md, queue[f"m:{f}"], match["cfg"].labeler)
        meta = clip["meta"]
        a, b = clip_range(f, md.span_of(f), FPS, 1.5, 10)
        assert meta["start_frame"] == a and meta["n"] == b - a
        assert meta["contact_index"] == f - a
        shown = [frame_shown(clip["frames"][j]) for j in (0, meta["contact_index"], meta["n"] - 1)]
        assert shown == [a, f, b - 1]
        # Never the reply: nothing past contact + after_frames.
        assert meta["start_frame"] + meta["n"] - 1 <= f + 10
        receiver = "far" if side == "near" else "near"
        assert meta["hitter_boxes"][0] == BOX[side] and meta["receiver_boxes"][0] == BOX[receiver]


def test_label_store_latest_wins_and_survives_a_torn_line(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    base = {"match_id": "m", "rally_id": 0, "shot_index": 1, "hitter_id": 0, "hitter_side": "near", "labeler": "t"}
    store.append({**base, "frame": 30, "label": "3"})
    store.append({**base, "frame": 60, "label": "1"})
    store.append({**base, "frame": 30, "label": "4"})  # changed my mind
    store.append({**base, "frame": 60, "label": "clear"})  # and took this one back
    with open(store.path, "a") as fh:
        fh.write("2026-10-05T00:00:00+00:00,m,95,0,3,0,near,")  # power cut mid-write
    latest = latest_labels(store.path)
    assert {k: r["label"] for k, r in latest.items()} == {"m:30": "4"}
    # The next label after the restart must start on its own line, not glue onto the torn one.
    store.append({**base, "frame": 125, "label": "2"})
    latest = latest_labels(store.path)
    assert {k: r["label"] for k, r in latest.items()} == {"m:30": "4", "m:125": "2"}
    assert store.path.read_text().count("time_utc") == 1  # one header
    df = load_labels(store.path)
    assert df[["match_id", "frame", "label"]].values.tolist() == [["m", 30, "4"], ["m", 125, "2"]]


def test_clip_range_stops_before_a_quick_reply():
    # A reply 6 frames after the hit: the clip ends 2 frames before it, not at contact + 10.
    assert clip_range(60, (0, 100), FPS, 1.5, 10, next_event=66) == (15, 64)
    assert clip_range(60, (0, 100), FPS, 1.5, 10, next_event=200) == (15, 71)
    assert clip_range(60, (0, 100), FPS, 1.5, 10, next_event=61) == (15, 61)  # the hit itself always shows


def test_clips_show_the_reply_and_stop_before_the_next_event(match):
    contacts = [(0, 1, "serve", 30), (0, 2, "hit", 60), (0, 3, "hit", 66), (0, 4, "hit", 95), (0, 5, "landing", 98),
                (1, 1, "serve", 125), (1, 2, "hit", 160)]
    pd.DataFrame(contacts, columns=["rally_id", "shot_index", "kind", "frame"]).to_csv(match["out"] / "contacts.csv", index=False)
    md = MatchData(match["cfg"], "m")
    queue = {r["key"]: r for r in build_queue(match["cfg"], ["m"])}
    assert [queue[f"m:{f}"]["next_event"] for f in (30, 60, 95, 125, 160)] == [60, 66, 98, 160, None]
    assert [queue[f"m:{f}"]["next_kind"] for f in (30, 60, 95, 125, 160)] == ["hit", "hit", "landing", "hit", None]
    assert [queue[f"m:{f}"]["after_reply"] for f in (30, 60, 95, 125, 160)] == [66, 95, None, None, None]
    # Through the reply, until 2 frames before the event after it; a shot that ended the rally: until
    # 2 frames before it lands; reply but nothing known after it: 10 frames past the reply; nothing
    # known: 10 frames.
    cases = [(30, 63, "reply", 60), (60, 92, "reply", 66), (95, 95, "landing", None),
             (125, 170, "reply", 160), (160, 170, "cut", None)]
    for f, last, stop, reply in cases:
        clip = render_clip(md, queue[f"m:{f}"], match["cfg"].labeler)
        meta = clip["meta"]
        assert meta["start_frame"] + meta["n"] - 1 == last and meta["stop"] == stop, f
        assert meta["reply_index"] == (None if reply is None else reply - meta["start_frame"])
        assert frame_shown(clip["frames"][-1]) == last
    # Earlier rules, one config switch away: stop just before the reply; or cut at 10 frames.
    before = load_config(overrides=["labeler.show_reply=false"]).labeler
    meta = render_clip(md, queue["m:30"], before)["meta"]
    assert meta["start_frame"] + meta["n"] - 1 == 57 and meta["stop"] == "hit" and meta["reply_index"] is None
    old = load_config(overrides=["labeler.show_reply=false", "labeler.until_next_event=false"]).labeler
    meta = render_clip(md, queue["m:30"], old)["meta"]
    assert meta["start_frame"] + meta["n"] - 1 == 40 and meta["stop"] == "cut"


def test_a_next_hit_too_far_off_is_not_the_reply(match):
    # 69 frames at 30 fps = 2.3 s: no shot flies that long, so hits were missed in between (fast net
    # play): cut 10 frames after the contact, as when nothing is known.
    pd.DataFrame([(0, 1, "serve", 30), (0, 2, "hit", 99), (1, 1, "serve", 125), (1, 2, "hit", 160)],
                 columns=["rally_id", "shot_index", "kind", "frame"]).to_csv(match["out"] / "contacts.csv", index=False)
    md = MatchData(match["cfg"], "m")
    queue = {r["key"]: r for r in build_queue(match["cfg"], ["m"])}
    meta = render_clip(md, queue["m:30"], match["cfg"].labeler)["meta"]
    assert meta["start_frame"] + meta["n"] - 1 == 40 and meta["stop"] == "cut" and meta["reply_index"] is None


def test_queue_order(match):
    q1 = [r["key"] for r in build_queue(match["cfg"], ["m"], "random", seed=1)]
    q2 = [r["key"] for r in build_queue(match["cfg"], ["m"], "random", seed=1)]
    rally = [r["key"] for r in build_queue(match["cfg"], ["m"], "rally")]
    assert q1 == q2 and sorted(q1) == sorted(rally)
    assert rally == [f"m:{f}" for _, _, f, _ in SHOTS]


@pytest.fixture
def server(match):
    rows = build_queue(match["cfg"], ["m"], "rally")
    store = LabelStore(match["cfg"].paths.labels_dir + "/shot_labels.csv")
    srv = LabelServer(("127.0.0.1", 0), match["cfg"], rows, store, "tester")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", store
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def post(url, obj):
    req = urllib.request.Request(url, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status, json.loads(r.read())


def test_server_end_to_end(server):
    base, store = server
    status, ctype, body = get(base + "/")
    assert status == 200 and "text/html" in ctype and b"Shot labeller" in body
    _, _, body = get(base + "/api/session")
    session = json.loads(body)
    assert [q["key"] for q in session["queue"]][:2] == ["m:30", "m:60"] and session["classes"] == CLASSES
    _, _, body = get(base + "/api/clip/m%3A60")
    meta = json.loads(body)
    assert meta["n"] == 56 and meta["contact_index"] == 45
    status, ctype, jpg = get(base + f"/api/frame/m%3A60/{meta['contact_index']}.jpg")
    assert ctype == "image/jpeg" and frame_shown(jpg) == 60
    status, res = post(base + "/api/label", {"key": "m:60", "label": "2", "seconds": 3.2})
    assert res["labels"] == {"m:60": "2"}
    row = latest_labels(store.path)["m:60"]
    assert row["label"] == "2" and row["hitter_side"] == "far" and row["labeler"] == "tester" and row["seconds"] == "3.2"
    post(base + "/api/prefetch", {"keys": ["m:95", "m:125"]})


def test_server_keeps_the_connection_open(server):
    # A clip is ~50 frame requests; a fresh connection for each one was the slow part.
    base, _ = server
    conn = http.client.HTTPConnection(base.removeprefix("http://"), timeout=20)
    for path in ["/api/clip/m%3A60", "/api/frame/m%3A60/0.jpg", "/api/frame/m%3A60/1.jpg"]:
        conn.request("GET", path)
        r = conn.getresponse()
        assert r.status == 200 and r.version == 11
        r.read()
    sock = conn.sock
    conn.request("GET", "/api/frame/m%3A60/2.jpg")
    conn.getresponse().read()
    assert conn.sock is sock  # the same socket all along
    conn.close()


@pytest.mark.parametrize("path, code", [
    ("/api/clip/..%2F..%2Fetc%2Fpasswd", 404),
    ("/api/frame/m%3A60/999.jpg", 404),
    ("/../../configs/default.yaml", 404),
    ("/api/clip/m%3A61", 404),  # not a shot
])
def test_server_refuses(server, path, code):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + path)
    assert e.value.code == code


def test_server_rejects_bad_labels(server):
    base, store = server
    for body, code in [({"key": "m:60", "label": "7"}, 400), ({"key": "m:61", "label": "1"}, 404)]:
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base + "/api/label", body)
        assert e.value.code == code
    assert not store.path.exists()
