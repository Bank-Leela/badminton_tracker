"""Phase 6: the labelling tool — server side.

`bda label` serves `tools/labeler/index.html` on localhost and everything it
needs: the queue of shots, each shot's clip as JPEG frames, and a place to
save labels. The page is keyboard-only (1-5 label, x not-a-shot, space
replay, arrows navigate); see the page for the keys.

Clips. From `before_s` before the contact until just before the receiver
hits the shuttle — or just before it lands, for a shot that ended the rally
(`until_next_event`; at most `max_after_s`). Never the reply itself: the
labeller judges the shot, not its outcome (`clip_range` asserts it). When
the next hit isn't known, `after_frames` (10) — a reply the hit detection
missed must not come into view. Never across a camera cut: the clip stays
inside the play span. (Tool version 1 always cut at `after_frames`.) Frames are
sent as JPEGs and played on a canvas — exact frames, instant replay, slow
motion, and no codec (this OpenCV writes no browser-playable H.264). Clips
are rendered on demand and the page asks for the next few ahead of time.

Labels. One append-only CSV, `data/labels/shot_labels.csv`: a row per key
press, flushed and fsynced before the page is told it is saved (the page
keeps unsaved labels and retries). The latest row for a shot wins, so
relabelling is just labelling again; a half-written last line from a crash
is skipped on read. Labels join `shots.csv` on `(match_id, frame)`.
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import REPO_ROOT, Config, cache_dir

TOOL_VERSION = 2  # 2: clips run until just before the reply (or the landing), not 10 frames
CLASSES = {
    "1": "Outright winner or opponent error",
    "2": "Opponent forced into a weak reply",
    "3": "Neutral, rally continues even",
    "4": "Hitter now on defense",
    "5": "Hitter erred (out or into net)",
    "x": "Not a shot / can't judge",
}
LABEL_COLUMNS = ["time_utc", "match_id", "frame", "rally_id", "shot_index", "hitter_id", "hitter_side",
                 "label", "seconds", "labeler", "tool_version"]
PAGE_DIR = REPO_ROOT / "tools" / "labeler"
KEY_RE = re.compile(r"^[\w-]+:\d+$")  # "<match_id>:<frame>"


class ClipLeakError(AssertionError):
    """A clip would show past `after_frames` after the contact — the reply (raised explicitly)."""


# --- Labels --------------------------------------------------------------------------------


class LabelStore:
    """The append-only label file; thread-safe."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def append(self, row: dict) -> None:
        line = io.StringIO()
        csv.writer(line).writerow([row.get(c, "") for c in LABEL_COLUMNS])
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            size = self.path.stat().st_size if self.path.exists() else 0
            torn = False
            if size:  # a crash can leave a line without its end: finish it, or this row is glued on and lost
                with open(self.path, "rb") as fh:
                    fh.seek(-1, os.SEEK_END)
                    torn = fh.read(1) != b"\n"
            with open(self.path, "a", newline="", encoding="utf-8") as fh:
                if not size:
                    csv.writer(fh).writerow(LABEL_COLUMNS)
                if torn:
                    fh.write("\r\n")
                fh.write(line.getvalue())
                fh.flush()
                os.fsync(fh.fileno())

    def latest(self) -> dict[str, dict]:
        """`{key: row}` — the last label per shot; `clear` removes one."""
        return latest_labels(self.path)


def latest_labels(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return out
    header = rows[0]
    for r in rows[1:]:
        if len(r) != len(header):  # a line cut short by a crash
            continue
        row = dict(zip(header, r))
        if row.get("label") not in (*CLASSES, "clear"):
            continue
        key = f"{row['match_id']}:{row['frame']}"
        if row["label"] == "clear":
            out.pop(key, None)
        else:
            out[key] = row
    return out


def load_labels(path: str | Path) -> pd.DataFrame:
    """Latest label per shot as a DataFrame (`match_id, frame, label, ...`), for phase 7."""
    rows = list(latest_labels(path).values())
    df = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    if len(df):
        df["frame"] = df["frame"].astype(int)
    return df


# --- The queue -----------------------------------------------------------------------------


def build_queue(cfg: Config, matches: list[str] | None = None, order: str = "random", seed: int = 0) -> list[dict]:
    """Every shot of the chosen matches (all with `shots.csv` by default), in labelling order.

    `random`: one deterministic shuffle over all shots — each label then adds
    a match-balanced, unbiased sample, which is what phase 7's learning curve
    and match-split need. `rally`: match by match, in play order.
    """
    cache = Path(cfg.paths.cache_dir)
    ids = matches or sorted(p.parent.name for p in cache.glob("*/shots.csv"))
    rows = []
    for m in ids:
        f = cache / m / "shots.csv"
        if not f.exists():
            raise SystemExit(f"no {f}; run `bda shots --match-id {m}` first")
        s = pd.read_csv(f, usecols=["match_id", "rally_id", "shot_index", "frame", "hitter_id", "hitter_side", "is_serve"])
        rows.append(s)
    if not rows:
        raise SystemExit("no shots.csv found; run `bda shots` first")
    df = pd.concat(rows, ignore_index=True)
    next_events = _next_events(cache, ids)
    nxt = [next_events.get((m, int(f))) for m, f in zip(df["match_id"], df["frame"])]
    df["next_event"] = [n[0] if n else None for n in nxt]
    df["next_kind"] = [n[1] if n else None for n in nxt]
    if order == "random":
        df = df.iloc[np.random.default_rng(seed).permutation(len(df))]
    elif order == "rally":
        df = df.sort_values(["match_id", "frame"])
    else:
        raise ValueError(f"order must be random or rally, got {order!r}")
    out = []
    for r in df.itertuples(index=False):
        out.append({"key": f"{r.match_id}:{int(r.frame)}", "match_id": r.match_id, "frame": int(r.frame),
                    "rally_id": int(r.rally_id), "shot_index": int(r.shot_index),
                    "hitter_id": None if pd.isna(r.hitter_id) else int(r.hitter_id),
                    "hitter_side": r.hitter_side, "is_serve": int(r.is_serve),
                    "next_event": None if r.next_event is None or pd.isna(r.next_event) else int(r.next_event),
                    "next_kind": r.next_kind if isinstance(r.next_kind, str) else None})
    return out


def _next_events(cache: Path, ids: list[str]) -> dict[tuple[str, int], tuple[int, str]]:
    """`(match_id, hit frame) -> (frame, kind)` of the rally's next event — the reply ('hit')
    or the landing ('landing') — from `contacts.csv`."""
    out = {}
    for m in ids:
        f = cache / m / "contacts.csv"
        if not f.exists():
            continue
        c = pd.read_csv(f, usecols=["rally_id", "kind", "frame"]).sort_values(["rally_id", "frame"])
        for _, g in c.groupby("rally_id"):
            fr, kinds = g["frame"].to_numpy(), g["kind"].to_numpy()
            for a, b, k, kb in zip(fr[:-1], fr[1:], kinds[:-1], kinds[1:]):
                if k != "landing":
                    out[(m, int(a))] = (int(b), "landing" if kb == "landing" else "hit")
    return out


# --- Clips -----------------------------------------------------------------------------------


def clip_range(contact: int, span: tuple[int, int], fps: float, before_s: float, after_frames: int,
               next_event: int | None = None, margin: int = 2) -> tuple[int, int]:
    """`[start, end)` of a shot's clip: `before_s` before the contact to `after_frames` after, inside the span.

    And before the rally's next event: in a fast exchange the reply (or the
    shuttle landing) comes within `after_frames` — 742 of 30,266 shots — so
    the clip stops `margin` frames short of it (contact frames are good to
    a frame or two). Showing the reply is the one thing this tool must not do.
    """
    a, b = span
    start = max(a, contact - int(round(before_s * fps)))
    end = min(b, contact + after_frames + 1)
    if next_event is not None:
        end = min(end, max(contact + 1, next_event - margin))
    if end > contact + after_frames + 1 or (next_event is not None and end > max(contact + 1, next_event - margin)):
        raise ClipLeakError(f"clip would run to {end}: past contact {contact} + {after_frames} "
                            f"or into the next event at {next_event}")
    if not start <= contact < end:
        raise ValueError(f"contact {contact} outside its play span {span}")
    return start, end


class MatchData:
    """Per-match lookups the clips need: video, fps, play spans, the hitter's box per frame."""

    def __init__(self, cfg: Config, match_id: str):
        out = cache_dir(cfg, match_id)
        meta = json.loads((out / "poses.meta.json").read_text())
        self.video, self.fps = meta["video"], float(meta["fps"])
        views = pd.read_csv(out / "view_segments.csv")
        play = views[views["is_play"] == 1]
        self.spans = np.c_[play["start_frame"].to_numpy(), play["end_frame"].to_numpy()]
        import pyarrow.parquet as pq

        p = pq.read_table(out / "players.parquet", columns=["frame", "side", "x1", "y1", "x2", "y2"]).to_pandas()
        self.boxes = {}
        for side, g in p.groupby("side"):
            g = g.sort_values("frame")
            self.boxes[side] = (g["frame"].to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32))

    def span_of(self, frame: int) -> tuple[int, int]:
        i = np.flatnonzero((self.spans[:, 0] <= frame) & (frame < self.spans[:, 1]))
        if not len(i):
            raise ValueError(f"frame {frame} is in no play span")
        return int(self.spans[i[0], 0]), int(self.spans[i[0], 1])

    def player_box(self, side: str, frame: int):
        """The player at `side`'s box at `frame` (source pixels), or None where not found."""
        if side not in self.boxes:
            return None
        f, b = self.boxes[side]
        i = int(np.searchsorted(f, frame))
        return [float(v) for v in b[i]] if i < len(f) and f[i] == frame else None


def render_clip(md: MatchData, shot: dict, cfg_l: Config) -> dict:
    """A shot's clip: JPEG frames plus what the page draws over them."""
    from video import iter_frames

    contact = int(shot["frame"])
    # Until just before the receiver hits it (or it lands) when that is known
    # (`until_next_event`, at most `max_after_s`); otherwise `after_frames` —
    # a reply the hit detection missed must not come into view.
    nxt = shot.get("next_event")
    after = int(cfg_l.after_frames)
    if cfg_l.get("until_next_event") and nxt is not None:
        after = int(round(float(cfg_l.max_after_s) * md.fps))
    start, end = clip_range(contact, md.span_of(contact), md.fps, float(cfg_l.before_s), after,
                            nxt, int(cfg_l.next_event_margin))
    if nxt is not None and end >= nxt - int(cfg_l.next_event_margin):
        stop = shot.get("next_kind") or "hit"     # stopped just before the reply ('hit') or the landing
    else:
        stop = "cut"
    width = int(cfg_l.clip_width)
    receiver = {"near": "far", "far": "near"}.get(shot["hitter_side"])
    frames, boxes, opp_boxes, scale, height = [], [], [], None, None
    for idx, img in iter_frames(md.video, start, end):
        if scale is None:
            scale = width / img.shape[1]
            height = int(round(img.shape[0] * scale))
        small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(cfg_l.jpeg_quality)])
        if not ok:
            raise RuntimeError(f"JPEG encode failed at frame {idx}")
        frames.append(jpg.tobytes())
        # Both players: the hitter, and the receiver (where they stand is half
        # of "forced into a weak reply" vs "neutral").
        for out, side in ((boxes, shot["hitter_side"]), (opp_boxes, receiver)):
            box = md.player_box(side, idx) if side else None
            out.append(None if box is None else [round(v * scale, 1) for v in box])
    if len(frames) <= contact - start:
        raise RuntimeError(f"video ended before the contact of {shot['key']}")
    return {"frames": frames, "meta": {
        "key": shot["key"], "n": len(frames), "fps": md.fps, "start_frame": start,
        "contact_index": contact - start, "width": width, "height": height, "stop": stop,
        "hitter_boxes": boxes, "receiver_boxes": opp_boxes}}


class ClipCache:
    """Rendered clips, least recently used dropped first.

    Background workers render the clips the page says are coming next
    (decoding and JPEG encoding release the GIL, so they overlap), and one
    more loads every match's lookups at start, so no clip waits on a match.
    """

    def __init__(self, cfg: Config, shots: dict[str, dict], size: int = 24, workers: int = 3):
        self.cfg, self.shots, self.size = cfg, shots, size
        self.clips: OrderedDict[str, dict] = OrderedDict()
        self.matches: dict[str, MatchData] = {}
        self.lock = threading.Lock()
        self.match_locks: dict[str, threading.Lock] = {}
        self.pending: dict[str, threading.Event] = {}
        self.todo: queue.Queue = queue.Queue()
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()
        order = list(dict.fromkeys(s["match_id"] for s in shots.values()))
        threading.Thread(target=self._warm, args=(order,), daemon=True).start()

    def _match(self, m: str) -> MatchData:
        with self.lock:
            if m in self.matches:
                return self.matches[m]
            lock = self.match_locks.setdefault(m, threading.Lock())
        with lock:  # one load per match, however many ask at once
            with self.lock:
                if m in self.matches:
                    return self.matches[m]
            md = MatchData(self.cfg, m)
            with self.lock:
                self.matches[m] = md
            return md

    def _warm(self, matches: list[str]) -> None:
        for m in matches:
            try:
                self._match(m)
            except Exception as exc:
                print(f"labeler: cannot load {m}: {exc}")

    def get(self, key: str) -> dict:
        with self.lock:
            if key in self.clips:
                self.clips.move_to_end(key)
                return self.clips[key]
            ev = self.pending.get(key)
            mine = ev is None
            if mine:
                ev = self.pending[key] = threading.Event()
        if not mine:
            ev.wait()
            with self.lock:
                if key in self.clips:
                    return self.clips[key]
            raise RuntimeError(f"rendering {key} failed")
        try:
            clip = render_clip(self._match(self.shots[key]["match_id"]), self.shots[key], self.cfg.labeler)
            with self.lock:
                self.clips[key] = clip
                while len(self.clips) > self.size:
                    self.clips.popitem(last=False)
            return clip
        finally:
            with self.lock:
                self.pending.pop(key, None)
            ev.set()

    def prefetch(self, keys: list[str]) -> None:
        for k in keys:
            if k in self.shots:
                self.todo.put(k)

    def _worker(self) -> None:
        while True:
            k = self.todo.get()
            try:
                self.get(k)
            except Exception as exc:  # a bad clip shows as an error in the page when it is opened
                print(f"labeler: prefetch {k} failed: {exc}")


# --- The server ------------------------------------------------------------------------------


class LabelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, cfg: Config, queue_rows: list[dict], store: LabelStore, labeler: str):
        super().__init__(addr, Handler)
        self.cfg, self.queue_rows, self.store, self.labeler = cfg, queue_rows, store, labeler
        self.shots = {r["key"]: r for r in queue_rows}
        self.clips = ClipCache(cfg, self.shots, int(cfg.labeler.cache_clips))


class Handler(BaseHTTPRequestHandler):
    server: LabelServer
    # Keep-alive: a clip is ~50 JPEG requests; a new connection for each (HTTP/1.0)
    # costs seconds a clip from a Windows browser into WSL. Every reply sets
    # Content-Length, which keep-alive needs.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet: one line per label is printed instead
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._send(200, (PAGE_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/session":
            return self._json({"queue": self.server.queue_rows, "labels": self._labels(), "classes": CLASSES,
                               "labeler": self.server.labeler,
                               "after_frames": int(self.server.cfg.labeler.after_frames)})
        m = re.match(r"^/api/clip/([^/]+)$", path)
        if m:
            key = _key(m.group(1))
            if key not in self.server.shots:
                return self._error(404, "no such shot")
            try:
                return self._json(self.server.clips.get(key)["meta"])
            except Exception as exc:
                return self._error(500, f"clip failed: {exc}")
        m = re.match(r"^/api/frame/([^/]+)/(\d+)\.jpg$", path)
        if m:
            key = _key(m.group(1))
            if key not in self.server.shots:
                return self._error(404, "no such shot")
            try:
                frames = self.server.clips.get(key)["frames"]
            except Exception as exc:
                return self._error(500, f"clip failed: {exc}")
            i = int(m.group(2))
            if i >= len(frames):
                return self._error(404, "no such frame")
            return self._send(200, frames[i], "image/jpeg")
        return self._error(404, "not found")

    def do_POST(self):
        self.path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64 * 1024:
            self.close_connection = True  # the body was not read
            return self._error(413, "too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._error(400, "bad json")
        if self.path == "/api/prefetch":
            self.server.clips.prefetch([_key(k) for k in body.get("keys", [])][:8])
            return self._json({"ok": True})
        if self.path == "/api/label":
            key, label = _key(str(body.get("key", ""))), str(body.get("label", ""))
            shot = self.server.shots.get(key)
            if shot is None:
                return self._error(404, "no such shot")
            if label not in (*CLASSES, "clear"):
                return self._error(400, f"label must be one of {list(CLASSES)} or clear")
            seconds = body.get("seconds")
            row = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{
                c: shot[c] for c in ("match_id", "frame", "rally_id", "shot_index", "hitter_id", "hitter_side")},
                "label": label, "seconds": "" if seconds is None else round(float(seconds), 2),
                "labeler": self.server.labeler, "tool_version": TOOL_VERSION}
            try:
                self.server.store.append(row)
            except OSError as exc:
                return self._error(500, f"could not save: {exc}")
            print(f"label: {key} -> {label}")
            return self._json({"ok": True, "labels": self._labels()})
        return self._error(404, "not found")

    def _labels(self) -> dict[str, str]:
        return {k: r["label"] for k, r in self.server.store.latest().items() if k in self.server.shots}


def _key(s: str) -> str:
    from urllib.parse import unquote

    s = unquote(s)
    return s if KEY_RE.match(s) else ""


def serve(cfg: Config, matches: list[str] | None, order: str, seed: int, port: int, labeler: str,
          host: str = "127.0.0.1") -> LabelServer:
    """Build the queue and start the server (blocking `serve_forever` is the caller's)."""
    rows = build_queue(cfg, matches, order, seed)
    store = LabelStore(Path(cfg.paths.labels_dir) / "shot_labels.csv")
    server = LabelServer((host, port), cfg, rows, store, labeler)
    latest = store.latest()  # read once: per shot, it took minutes on a big label file
    done = sum(1 for r in rows if r["key"] in latest)
    print(f"labeler: {len(rows)} shots ({done} labelled) from {len({r['match_id'] for r in rows})} matches; "
          f"labels -> {store.path}")
    # 127.0.0.1, not localhost: from Windows, localhost may try IPv6 first and stall into WSL.
    print(f"labeler: open http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)")
    return server
