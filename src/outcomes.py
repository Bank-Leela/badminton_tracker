"""Rally outcomes: who won each rally, read off the broadcast's score graphic.

Phase 7's free baseline labels need each rally's winner, and its verdict
layer needs the score. Phase 5's own signals can't give the winner: the
next rally's server is found mostly at the far end, and the final landing
is seen in 70% of rallies (the two agree on 54%). The score graphic says it
outright: **between two rallies only the winner's score changes.**

No OCR, and no per-broadcast setup: all 32 broadcasts (2018-2026, several
graphic templates) put the score top-left, and the score is found by what
a score does.

1. Per rally, the per-pixel median of a few frames from its start: the
   graphic as it stood before that rally (`rally_stills`).
2. Candidate pixels in the top-left `search_box` change between some
   rallies but not most (`candidate_pixels`).
3. Cells: per rally change, digit-sized blobs of changed pixels; a place
   painted again and again is a cell (`digit_groups`).
4. The display: one player's digit above the other's, a complementary
   pair — between two rallies one or the other changes, not both, and
   while that column is in play almost every rally changes it. The best
   pair fixes the two rows; other pairs on those rows (tens, other games'
   columns) join unless they contradict it (`find_display`). Spectators,
   the server icon and sponsor boards fail one test or another.
5. Per rally change (`verdict`, `decide`): one row changed is that row's
   point; both rows changed is a reset (a new game) or a banner, never a
   point. A rally where the graphic is covered or gone (`display_absent`),
   or a run of unreadable changes, is bridged by comparing the rallies
   either side. A game end seen only as a reset goes to whoever that point
   ends the game for (21 with a 2-point lead, or 30). The score is
   replayed rally by rally.

Validated against the real final scores of 20 matches (`docs/match_scores.csv`).

Rows are matched to phase 4's player ids by a vote (`map_rows`): the winner
serves next, and the server stands nearer the centre line as the rally
starts (`central_servers`, 86-100% per match); where the shuttle came to
rest, clearly in or clearly out, says who won (`landing_winners`, ~80%).

Output `rallies.csv`: per rally, the winning row and player id, how it was
decided, the game, and the score before it from that player's side.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import Config, cache_dir
from court import load_homographies, project
from features import dist_from_singles_lines
from video import sample_frames

SCHEMA_VERSION = 2                     # rallies.csv's columns; a cached file of another version is redone
# Per rally: the winning row of the display and (once rows are matched) player id; the
# score before the rally, by row and by player id; `score_exact` = no unread rally so far
# this game.
RALLY_COLUMNS = ["rally_id", "winner_row", "winner_id", "how", "game", "points_0", "points_1",
                 "games_0", "games_1", "score_exact", "points_r1", "points_r2", "games_r1", "games_r2"]


class OutcomeError(RuntimeError):
    """No usable score graphic in a match."""


# --- Frames ----------------------------------------------------------------------------------


def rally_windows(cfg: Config, match_id: str) -> pd.DataFrame:
    """`rally_id, start_frame, end_frame` of each rally's first play segment (`segments.csv`)."""
    seg = pd.read_csv(Path(cfg.paths.cache_dir) / match_id / "segments.csv")
    seg = seg[seg["rally_id"] >= 0].sort_values("start_frame")
    first = seg.groupby("rally_id", as_index=False).first()
    return first[["rally_id", "start_frame", "end_frame"]].astype(int)


def sample_positions(start: int, end: int, fps: float, cfg_o: Config) -> list[int]:
    """Frames to sample from one rally: evenly from its start, never near its end (the score updates after the point)."""
    dur = (end - start) / fps
    a = start + cfg_o.sample_from_s * fps
    b = start + min(cfg_o.sample_to_s, cfg_o.sample_to_frac * dur) * fps
    if b <= a:
        return [int(start + (end - start) // 2)]
    k = int(cfg_o.frames_per_rally)
    return sorted({int(round(f)) for f in np.linspace(a, b, k)})


def rally_stills(video: str, windows: pd.DataFrame, fps: float, cfg_o: Config) -> tuple[np.ndarray, np.ndarray]:
    """Per rally: the grey median of its sampled frames, and where that median is trusted.

    A pixel is trusted when at least `min_agree` of the frames are within
    `agree_level` of the median — a banner over two of seven frames, or a
    player crossing, doesn't stop the rest of the graphic being read.
    """
    stills, trust = [], []
    for w in windows.itertuples():
        frames = sample_frames(video, sample_positions(w.start_frame, w.end_frame, fps, cfg_o))
        g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
        med = np.median(g, axis=0).astype(np.uint8)
        agree = (np.abs(g.astype(np.int16) - med) <= cfg_o.agree_level).mean(axis=0) >= cfg_o.min_agree
        stills.append(med)
        trust.append(agree)
    return np.stack(stills), np.stack(trust)


# --- Score pixels ----------------------------------------------------------------------------


def candidate_pixels(stills: np.ndarray, trust: np.ndarray, cfg_o: Config,
                     region: np.ndarray | None = None, two_level: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pixels (within `region`) that change between some rallies but not most (and, with
    `two_level` a share, sit at two grey levels in that share of rallies): `(ys, xs)`."""
    changed = np.zeros(stills.shape[1:], np.int32)
    valid = np.zeros(stills.shape[1:], np.int32)
    for r in range(len(stills) - 1):
        ok = trust[r] & trust[r + 1]
        diff = np.abs(stills[r + 1].astype(np.int16) - stills[r]) > cfg_o.change_level
        changed += diff & ok
        valid += ok
    frac = changed / np.maximum(valid, 1)
    keep = (frac >= cfg_o.min_change_frac) & (frac <= cfg_o.max_change_frac) & (valid >= 0.5 * (len(stills) - 1))
    if region is not None:
        keep &= region
    ys, xs = np.nonzero(keep)
    if two_level is None:
        return ys, xs
    # A digit pixel is ink or background, nothing between: over the rallies its
    # values sit at two levels. Spectators sit still within a rally but shift
    # between rallies — they pass the test above, and fail this one.
    v = stills[:, ys, xs].astype(np.float32)
    ok = trust[:, ys, xs]
    lo = np.nanpercentile(np.where(ok, v, np.nan), 5, axis=0)
    hi = np.nanpercentile(np.where(ok, v, np.nan), 95, axis=0)
    span = hi - lo
    band = cfg_o.two_level_band * span
    at_ends = ((np.abs(v - lo) <= band) | (np.abs(v - hi) <= band)) | ~ok
    keep = (span >= cfg_o.min_contrast) & (at_ends.mean(axis=0) >= two_level)
    return ys[keep], xs[keep]


def _changes(s: np.ndarray, t: np.ndarray, i: int, j: int, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Which pixels changed from rally i to rally j, and where that can be judged."""
    ok = t[i] & t[j]
    return (np.abs(s[j].astype(np.int16) - s[i]) > level) & ok, ok


@dataclass
class Group:
    """One cell of a score display (a digit position, or the server icon)."""

    idx: np.ndarray                     # indices into the candidate pixel arrays inside the cell
    x0: int
    y0: int
    x1: int
    y1: int
    timeline: np.ndarray | None = None  # changed between rally r and r+1
    row: int = 0                        # 1 = top row, 2 = bottom row of its display
    col: int = -1

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def digit_groups(ys: np.ndarray, xs: np.ndarray, ch: np.ndarray, cfg_o: Config) -> list[Group]:
    """Find the cells: places where a small patch changes, again and again.

    Pixels of one digit don't change together (1->2 and 2->3 touch different
    strokes), but the patch that changes is always in the same place. Per
    rally change, the changed candidate pixels form blobs; each digit-sized
    blob (at least `min_group_px` changed pixels, at most `max_cell_px` each
    way) paints its box onto a heat map; a cell is a region painted at least
    `min_cell_events` times. A graphic redrawn whole (a game break) is one
    big blob, and a player is too big: neither paints.
    """
    if len(ys) == 0:
        return []
    h, w = int(ys.max()) + 2, int(xs.max()) + 2
    heat = np.zeros((h, w), np.int32)
    for r in range(ch.shape[1]):
        img = np.zeros((h, w), np.uint8)
        img[ys[ch[:, r]], xs[ch[:, r]]] = 1
        if not img.any():
            continue
        # No dilation: the 2018-19 graphic sits on the crowd, and a digit's
        # change must not merge with a spectator moving beside it.
        n, _, stats, _ = cv2.connectedComponentsWithStats(img)
        for x, y, bw, bh, area in stats[1:]:
            if area >= cfg_o.min_group_px and bw <= cfg_o.max_cell_px and bh <= cfg_o.max_cell_px:
                heat[y:y + bh, x:x + bw] += 1
    # Different digit flips touch different strokes: fragments of one digit a
    # few px apart are one cell. Rows and columns sit further apart than this.
    kernel = np.ones((2 * cfg_o.cell_join_px + 1,) * 2, np.uint8)
    # A painted region wider than a digit is crowd — or a digit with crowd
    # around it (a game column that showed spectators until its game began).
    # A digit is painted on nearly every point, spectators now and then: raise
    # the bar inside an oversized region until digit-sized cores come apart.
    groups, open_ = [], np.ones((h, w), bool)
    for level in range(cfg_o.min_cell_events, int(heat.max()) + 1):
        painted = cv2.dilate(((heat >= level) & open_).astype(np.uint8), kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(painted)
        if n <= 1:
            break
        for x, y, bw, bh, _ in stats[1:]:
            if bw > cfg_o.max_cell_px or bh > cfg_o.max_cell_px:
                continue                       # still too wide: try a higher bar
            open_[y:y + bh, x:x + bw] = False
            inside = np.flatnonzero((xs >= x) & (xs < x + bw) & (ys >= y) & (ys < y + bh))
            if len(inside) >= cfg_o.min_group_px:
                groups.append(Group(inside, int(x), int(y), int(x + bw - 1), int(y + bh - 1)))
    return groups


def cell_changed(g: Group, s: np.ndarray, t: np.ndarray, i: int, j: int, cfg_o: Config) -> tuple[bool, bool]:
    """(changed, readable) for one cell between rally i and rally j."""
    ok = t[i, g.idx] & t[j, g.idx]
    if ok.mean() < 0.5:
        return False, False
    diff = np.abs(s[j, g.idx].astype(np.int16) - s[i, g.idx]) > cfg_o.cell_change_level
    n = int((diff & ok).sum())
    return bool(n >= cfg_o.min_group_px and n >= cfg_o.cell_changed * ok.sum()), True


@dataclass
class ScoreObject:
    """A two-row score display: its cells, split into rows and columns."""

    groups: list[Group]
    box: tuple[int, int, int, int]
    cut_y: float
    columns: list[list[Group]] = field(default_factory=list)
    active: list[np.ndarray] = field(default_factory=list)   # per column: rally changes where it is in play


def _stacked(a: Group, b: Group, loose: bool = False) -> bool:
    """`b` sits right under `a` (less than a digit's height between them: 0.3-0.7 in
    every template), in the same column, at about the same size. `loose`: the rows
    are already known, only the column and the order matter."""
    ha, hb = a.y1 - a.y0 + 1, b.y1 - b.y0 + 1
    overlap = min(a.x1, b.x1) - max(a.x0, b.x0) + 1
    narrow = min(a.x1 - a.x0, b.x1 - b.x0) + 1
    gap = b.y0 - a.y1
    if loose:
        return overlap >= 0.5 * narrow and gap >= 0
    return overlap >= 0.5 * narrow and 0 <= gap <= max(ha, hb) and min(ha, hb) >= 0.6 * max(ha, hb)


def _merge(parts: list[Group], s: np.ndarray, t: np.ndarray, cfg_o: Config) -> Group:
    """Fragments of one cell as one cell (a single part is returned as is)."""
    if len(parts) == 1:
        return parts[0]
    g = Group(np.unique(np.concatenate([p.idx for p in parts])), min(p.x0 for p in parts), min(p.y0 for p in parts),
              max(p.x1 for p in parts), max(p.y1 for p in parts))
    g.timeline = np.array([cell_changed(g, s, t, r, r + 1, cfg_o)[0] for r in range(len(s) - 1)])
    return g


def _dense(changed: np.ndarray, cfg_o: Config) -> np.ndarray | None:
    """While a score column is in play it changes on almost every rally.

    A column can be in play in bursts — a digit position used only while a
    score is single-digit, a game's own column — so the changes are cut into
    stretches wherever `pair_gap` rallies pass without one. Stretches of at
    least three changes must hold 85% of them (a banner or a layout shift
    may add a few strays) and fill `min_pair_density` of their length.
    Spectators fail either way: often, they change at a low rate all match
    long (one long thin stretch); rarely, in isolated changes.

    Returns where the column is in play (its stretches), or None.
    """
    on = np.flatnonzero(changed)
    if len(on) < 3:
        return None
    runs = np.split(on, np.flatnonzero(np.diff(on) > cfg_o.pair_gap) + 1)
    runs = [r for r in runs if len(r) >= 3]
    held = sum(len(r) for r in runs)
    if held < 0.85 * len(on) or held / sum(r[-1] - r[0] + 1 for r in runs) < cfg_o.min_pair_density:
        return None
    active = np.zeros(len(changed), bool)
    for r in runs:
        active[r[0]:r[-1] + 1] = True
    return active


def find_display(groups: list[Group], s: np.ndarray, t: np.ndarray, cfg_o: Config) -> ScoreObject | None:
    """Assemble the score display from its cells, by what a score does.

    The two players' digits in one column are a complementary pair: one
    above the other, and between two rallies one or the other changes —
    almost never both (a reset), almost never neither while that column is
    in play. The pair covering the most rally changes fixes the two rows.
    Every other complementary pair on those rows (tens digits, the other
    games' columns) joins when it never contradicts the columns already in:
    spectators that happen to pair up get contradicted at once. A column
    whose rows change together (the server icon) is never complementary.

    A column only counts where it is in play (`_dense`): a later game's
    column is crowd until that game starts.
    """
    if len(groups) < 2:
        return None
    for g in groups:
        g.timeline = np.array([cell_changed(g, s, t, r, r + 1, cfg_o)[0] for r in range(len(s) - 1)])
    pairs = []
    for a in groups:
        for b in groups:
            if a is b or not _stacked(a, b):
                continue
            active = _dense(a.timeline | b.timeline, cfg_o)
            if active is None:
                continue
            u = int(((a.timeline | b.timeline) & active).sum())
            i = int((a.timeline & b.timeline & active).sum())
            if u < cfg_o.min_pair_changes or i > cfg_o.pair_overlap * u:
                continue
            pairs.append((u - 3 * i, a, b, active))
    if not pairs:
        return None
    pairs.sort(key=lambda p: -p[0])
    _, a0, b0, act0 = pairs[0]
    pad = 0.25 * (a0.y1 - a0.y0 + 1)
    in_row = lambda g, r: r.y0 - pad <= g.cy <= r.y1 + pad

    def votes(a: Group, b: Group, active: np.ndarray) -> np.ndarray:
        return np.where(active & a.timeline & ~b.timeline, 1, np.where(active & b.timeline & ~a.timeline, 2, 0))

    cols, actives = [[a0, b0]], [act0]
    used = {id(a0), id(b0)}
    vote = votes(a0, b0, act0)
    # With the rows known, the other columns' cells may come in fragments (two
    # halves of a digit): within each row, fragments overlapping in x are one cell.
    bands = []
    for row_of in (a0, b0):
        frags = sorted((g for g in groups if id(g) not in used and in_row(g, row_of)), key=lambda g: g.x0)
        merged: list[list[Group]] = []
        for g in frags:
            if merged and g.x0 <= max(h.x1 for h in merged[-1]):
                merged[-1].append(g)
            else:
                merged.append([g])
        bands.append([_merge(m, s, t, cfg_o) for m in merged])
    more = []
    for a in bands[0]:
        for b in bands[1]:
            if not _stacked(a, b, loose=True):
                continue
            active = _dense(a.timeline | b.timeline, cfg_o)
            if active is None:
                continue
            u = int(((a.timeline | b.timeline) & active).sum())
            i = int((a.timeline & b.timeline & active).sum())
            if u >= 3 and i <= cfg_o.pair_overlap * u:
                more.append((u - 3 * i, a, b, active))
    for _, a, b, active in sorted(more, key=lambda p: -p[0]):
        if id(a) in used or id(b) in used:
            continue
        mine = votes(a, b, active)
        both = (mine > 0) & (vote > 0)
        if (mine[both] != vote[both]).sum() > cfg_o.max_contradictions:
            continue
        cols.append([a, b])
        actives.append(active)
        used |= {id(a), id(b)}
        vote = np.where(vote > 0, vote, mine)
    cells = [g for col in cols for g in col]
    for k, col in enumerate(cols):
        col[0].row, col[1].row = 1, 2
        for g in col:
            g.col = k
    box = (min(g.x0 for g in cells), min(g.y0 for g in cells), max(g.x1 for g in cells), max(g.y1 for g in cells))
    return ScoreObject(cells, box, (a0.y1 + b0.y0) / 2, columns=cols, active=actives)


# --- Decisions -------------------------------------------------------------------------------


def verdict(obj: ScoreObject, s: np.ndarray, t: np.ndarray, i: int, j: int, cfg_o: Config) -> tuple[str, bool]:
    """What happened on one display between rally i and rally j, and whether a new column opened.

    Code '1'/'2': a point to that row; 'b': only both-row changes (a reset
    or a banner); 'x': columns disagree; '.': nothing changed; '?': the
    display couldn't be read. The flag says a column not in play before i
    appeared (changed in both rows): with a single-row change elsewhere, that
    is a 2022-26 graphic opening the next game's column — the point ended a
    game. Only columns in play somewhere in i..j are read. A column whose
    one row changed while the other can't be read says nothing reliable:
    '?' (it may be a reset with one row hidden).
    """
    live = [(col, act) for col, act in zip(obj.columns, obj.active) if act[i:max(j, i + 1)].any()]
    if not live:
        return ".", False
    states = {id(g): cell_changed(g, s, t, i, j, cfg_o) for col, _ in live for g in col}
    if np.mean([ok for _, ok in states.values()]) < 0.5:
        return "?", False
    single, both, opened = set(), False, False
    for col, act in live:
        hit = {1: False, 2: False}
        blind = {1: False, 2: False}
        for g in col:
            changed, ok = states[id(g)]
            hit[g.row] |= changed and ok
            blind[g.row] |= not ok
        if (hit[1] and blind[2]) or (hit[2] and blind[1]):
            return "?", False
        if hit[1] and hit[2]:
            both = True
            opened |= not act[:i].any()
        elif hit[1] or hit[2]:
            single.add("1" if hit[1] else "2")
    if len(single) == 1:
        return single.pop(), opened
    if len(single) == 2:
        return "x", False
    return ("b" if both else "."), opened


def display_absent(obj: ScoreObject, stills: np.ndarray, trust: np.ndarray, cfg_o: Config) -> np.ndarray:
    """Per rally: the display is off screen or covered (a banner, a cutaway, a blank).

    The names beside the digits (every template puts them on the left)
    never change while the graphic is shown — unlike the digits' own boxes,
    which gain a column each game in 2022-26 graphics. A rally where that
    band (`panel_band_px` wide) differs from its usual look by more than
    `absent_level` grey levels on average can't be read, and a point read
    across it would be a change of graphic, not of score.
    """
    h, w = stills.shape[1:]
    x0, y0, x1, y1 = obj.box
    panel = np.zeros((h, w), bool)
    panel[y0:y1 + 1, max(0, x0 - cfg_o.panel_band_px):max(0, x0 - 4)] = True
    if not panel.any():
        return np.zeros(len(stills), bool)
    v = stills[:, panel].astype(np.float32)
    ok = trust[:, panel]
    usual = np.median(v, axis=0)
    dev = np.array([np.abs(v[r] - usual)[ok[r]].mean() if ok[r].mean() > 0.5 else np.inf for r in range(len(v))])
    return dev > cfg_o.absent_level


def per_game_columns(obj: ScoreObject) -> bool:
    """The display keeps a column per game (2022-26 graphics): two columns each in play for a
    stretch of at least 10 rally changes, barely overlapping. (Tens and units of one score
    column are in play at the same time.)"""
    long = [a for a in obj.active if a.sum() >= 10]
    for i, a in enumerate(long):
        for b in long[i + 1:]:
            if (a & b).sum() <= 0.2 * min(a.sum(), b.sum()):
                return True
    return False


def _game_over(a: int, b: int) -> bool:
    return (max(a, b) >= 21 and abs(a - b) >= 2) or max(a, b) >= 30


def decide(obj: ScoreObject, s: np.ndarray, t: np.ndarray, cfg_o: Config,
           absent: np.ndarray | None = None) -> pd.DataFrame:
    """Per rally: winning row, how it was found, game, and the score (rows) before it.

    Transition r is rally r -> r+1, so it carries rally r's point; the score
    shown at a rally's start is the score before it.

    - One row changed: that row's point. If a new column opened alongside
      (2022-26 graphics), that point ended the game.
    - A lone both-rows change: a reset — the end of a game (both rows go
      back to 0), if one is reachable from the count; the point goes to
      whoever it ends the game for. Otherwise (a missed rally split between
      the players) it is left unread.
    - A run of unreadable changes (a banner, a cutaway, a missing graphic) is
      bridged by comparing the rallies either side. One row changed: every
      point in the run went to it. Anything else, the run's points stay
      unread; a game may have ended inside it (where, unknown) — unless a
      reset follows within a few rallies, which says the run was a split.

    "Reachable": the leader's count, plus the points not read this game,
    plus this one, is within `reset_slack` of 21 (a rally phase 2 missed
    leaves the count short); with no certain start to the game, any count.
    A count that reaches game over on a plain point, with no reset in
    sight, ends the game too — uncertainly (the reset may have been
    unreadable); with a reset coming within a few rallies, the count was
    short and the game goes on.

    The replay marks `score_exact` while every rally so far in the game was
    read and the game's start is certain. A game is credited to the winner
    of its last rally when that was read and the end was seen; otherwise
    the games tally stops (NaN).
    """
    n = len(s)
    absent = np.zeros(n, bool) if absent is None else absent
    v = [("?", False) if absent[r] or absent[r + 1] else verdict(obj, s, t, r, r + 1, cfg_o) for r in range(n - 1)]
    row: list[int | None] = [None] * n
    how = ["none"] * n
    ends: list[str | None] = [None] * n       # after rally r: 'point' (seen), 'reset' (lone), 'somewhere' (in a run)
    wopen_at: dict[int, bool] = {}            # a bridged run's end: did a new column open inside it
    r = 0
    while r < n - 1:
        code, opened = v[r]
        if code in ("1", "2"):
            row[r], how[r] = int(code), "digits"
            if opened:
                ends[r] = "point"
            r += 1
            continue
        if code == ".":
            r += 1
            continue
        end = r
        while end + 1 < n - 1 and v[end + 1][0] in "bx?":
            end += 1
        if end == r and code == "b":
            ends[r] = "reset"
        elif end > r and not absent[r] and not absent[end + 1]:
            w, wopened = verdict(obj, s, t, r, end + 1, cfg_o)
            if w in ("1", "2") and not wopened:
                for k in range(r, end + 1):       # every point in the run went to that row
                    row[k], how[k] = int(w), "skip"
            elif wopened or w == "b":
                ends[end] = "somewhere"
                wopen_at[end] = wopened
        r = end + 1
    if per_game_columns(obj):
        # A graphic with a column per game shows every game end as a new column;
        # a both-rows change without one is a banner or a split, never an end.
        for r in range(n):
            if ends[r] == "reset" or (ends[r] == "somewhere" and not wopen_at.get(r, False)):
                ends[r] = None
    look = int(cfg_o.end_lookahead)
    seen_soon = lambda r: any(ends[k] in ("reset", "point") for k in range(r + 1, min(n, r + 1 + look)))

    out = []
    game, pts, games, games_known, exact, unread, sure_start = 1, [0, 0], [0, 0], True, True, 0, True
    for r in range(n):
        # A game's count is short by any rally phase 2 missed (reset_slack); more
        # so when the game's start itself wasn't seen (uncertain_slack).
        slack = int(cfg_o.reset_slack) if sure_start else int(cfg_o.uncertain_slack)
        reachable = max(pts) + unread + 1 >= 21 - slack
        if ends[r] == "somewhere" and seen_soon(r):
            ends[r] = None                        # a reset follows: the run was a split, not the game's end
        if ends[r] in ("reset", "point", "somewhere") and not reachable:
            ends[r] = None                        # no game can end here (a split, or a column change mid-game)
        if ends[r] == "reset" and row[r] is None:
            fin = [k for k in (0, 1) if _game_over(pts[0] + (k == 0), pts[1] + (k == 1))]
            if len(fin) == 1 and unread == 0:
                row[r], how[r] = fin[0] + 1, "game_end"
            elif pts[0] != pts[1]:
                row[r], how[r] = int(np.argmax(pts)) + 1, "game_end_lead"
        out.append({"winner_row": row[r], "how": how[r], "game": game, "points_r1": pts[0], "points_r2": pts[1],
                    "games_r1": games[0] if games_known else np.nan,
                    "games_r2": games[1] if games_known else np.nan, "score_exact": exact})
        if row[r] is None:
            unread += 1
            exact = False
        else:
            pts[row[r] - 1] += 1
        if ends[r] is None:
            if not _game_over(*pts):
                continue
            if r + 1 < n and seen_soon(r):
                exact = False                     # the count is short: the real game ends at that reset
                continue
        # The game is over. Its end was seen ('point', 'reset'), or only counted.
        seen = ends[r] in ("point", "reset")
        if seen and row[r] is not None and how[r] != "game_end_lead" and games_known:
            games[row[r] - 1] += 1
        else:
            games_known = False                   # who won it is a guess: stop the tally
        sure_start = seen or (ends[r] is None and unread == 0)
        game, pts, unread, exact = game + 1, [0, 0], 0, sure_start
    return pd.DataFrame(out)


# --- Rows -> players -------------------------------------------------------------------------


def landing_winners(cfg: Config, match_id: str, cfg_o: Config) -> dict[int, int]:
    """`rally_id -> winner's player id`, from where the shuttle came to rest (`contacts.csv`).

    Clearly in: the player at that end failed to return it — they lost.
    Clearly out: it was hit out to that end — they won. None of this needs
    to know who hit last (the hit most often missed at a rally's end).
    Phase 5's serves can't help here: its detected server is nearly always
    the far-end player.

    The shuttle is placed where it first came to rest (`contacts.csv` keeps
    the middle of the rest, by when it is often in a player's hand: a metre
    up, projected metres past the far baseline). Near the net it may be
    lying on the tape: not used (`landing_net_gap_m`). On a hand-checked
    match this vote agrees with the score graphic 81% of the time; a
    match's majority is what counts.
    """
    d = Path(cfg.paths.cache_dir) / match_id
    shots = pd.read_csv(d / "shots.csv", usecols=["rally_id", "hitter_id", "hitter_side"]).dropna()
    out = {}
    for rid, (x, y) in landing_points(cfg, match_id).items():
        dist = dist_from_singles_lines(x, y)
        if not np.isfinite(dist) or abs(dist) < cfg_o.clear_landing_m or abs(y) < cfg_o.landing_net_gap_m:
            continue
        # Ends change between games (and at 11 in the third): who was where in *this* rally.
        sides = shots[shots["rally_id"] == rid].drop_duplicates("hitter_side")
        at_end = sides.loc[sides["hitter_side"] == ("near" if y < 0 else "far"), "hitter_id"]
        if at_end.empty:
            continue
        pid = int(at_end.iloc[0])
        out[rid] = pid if dist < 0 else 1 - pid
    return out


def landing_points(cfg: Config, match_id: str) -> dict[int, tuple[float, float]]:
    """`rally_id -> (x, y)` court metres where the shuttle first came to rest (near half: y < 0).

    `contacts.csv` keeps a landing's frame (the start of the rest) and the
    middle of the rest as its pixel — by then the shuttle is often in a
    player's hand, a metre up, which the floor homography throws metres
    away. So the track's own point at the landing frame (or just after;
    just before, the shuttle is still in the air) is used.
    """
    d = Path(cfg.paths.cache_dir) / match_id
    contacts = pd.read_csv(d / "contacts.csv")
    homs = load_homographies(d / "homography.json")
    track = pd.read_csv(d / "shuttle.csv")
    track = track[track["visible"] == 1].set_index("frame")
    out = {}
    for land in contacts[contacts["kind"] == "landing"].itertuples():
        f0 = int(land.frame)
        at = [f for f in (f0, f0 + 1, f0 + 2, f0 - 1, f0 - 2) if f in track.index]
        seg = homs[(homs["start_frame"] <= f0) & (homs["end_frame"] > f0)]
        if not at or seg.empty or seg["H"].iloc[0] is None:
            continue
        p = track.loc[at[0]]
        x, y = project(seg["H"].iloc[0], np.array([[p["x"], p["y"]]]))[0]
        if np.isfinite(x) and np.isfinite(y):
            out[int(land.rally_id)] = (float(x), float(y))
    return out


def central_servers(cfg: Config, match_id: str, cfg_o: Config) -> dict[int, int]:
    """`rally_id -> the server's player id`: whoever stands nearer the centre line as the rally starts.

    In singles the server stands by the centre line and the receiver in the
    middle of their service court: at the rally's first `server_frames`
    frames the server is the more central in 86-100% of rallies (checked on
    five matches against the score graphic). It needs no hit detection, and
    holds even when the serve itself happened before the main camera cut
    in. The winner of a rally serves the next one.
    """
    d = Path(cfg.paths.cache_dir) / match_id
    windows = rally_windows(cfg, match_id)
    pl = pd.read_parquet(d / "players.parquet", columns=["frame", "player_id", "court_x"])
    out = {}
    for w in windows.itertuples():
        p = pl[(pl["frame"] >= w.start_frame) & (pl["frame"] < w.start_frame + cfg_o.server_frames)]
        lat = p.groupby("player_id")["court_x"].median().abs().dropna()
        if len(lat) == 2 and abs(lat.iloc[0] - lat.iloc[1]) >= cfg_o.server_gap_m:
            out[int(w.rally_id)] = int(lat.idxmin())
    return out


def row_votes(winner_row: pd.Series, rally_ids: np.ndarray, landed: dict[int, int], served: dict[int, int]) -> dict:
    """Votes for "row 1 is player 0" (`same`) vs "row 1 is player 1" (`swapped`).

    Each rally with a known winning row votes through where its shuttle
    landed (`landed`: that rally's winner) and who served the next rally
    (`served`: the winner again).
    """
    votes = {"same": 0, "swapped": 0, "landing": 0, "serve": 0}
    for k, rid in enumerate(rally_ids):
        w = winner_row.iloc[k]
        if w is None or pd.isna(w):
            continue
        says = []
        if int(rid) in landed:
            says.append(("landing", landed[int(rid)]))
        if k + 1 < len(rally_ids) and int(rally_ids[k + 1]) in served:
            says.append(("serve", served[int(rally_ids[k + 1])]))
        for src, pid in says:
            votes[src] += 1
            votes["same" if (int(w) == 1) == (pid == 0) else "swapped"] += 1
    return votes


def map_rows(cfg: Config, match_id: str) -> dict:
    """Match the display's rows to phase 4's player ids, and fill `rallies.csv`'s id columns.

    Reads the row-based columns `find_outcomes` wrote, so it can be rerun
    without the video (`bda outcomes --remap`). Rows stay unmatched — every
    id column empty — unless at least `min_votes` votes agree `min_vote_share`
    of the time.
    """
    path = Path(cfg.paths.cache_dir) / match_id / "rallies.csv"
    df = pd.read_csv(path)
    lacking = [c for c in RALLY_COLUMNS if c not in df.columns]
    if lacking:
        raise OutcomeError(f"{path}: written by an older version (no {', '.join(lacking)}); "
                           f"rerun `bda outcomes --match {match_id} --force`")
    df, info = apply_mapping(cfg, match_id, df)
    _write_atomic(df[RALLY_COLUMNS], path)
    return info


def apply_mapping(cfg: Config, match_id: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`df` (row-based rallies) with its player-id columns filled from the vote, and the vote."""
    cfg_o = cfg.outcomes
    df = df.copy()
    votes = row_votes(df["winner_row"], df["rally_id"].to_numpy(),
                      landing_winners(cfg, match_id, cfg_o), central_servers(cfg, match_id, cfg_o))
    n = votes["same"] + votes["swapped"]
    share = max(votes["same"], votes["swapped"]) / max(n, 1)
    mapped = n >= cfg_o.min_votes and share >= cfg_o.min_vote_share
    swap = votes["swapped"] > votes["same"]
    first, second = ("r2", "r1") if swap else ("r1", "r2")       # the row that is player 0, player 1
    if mapped:
        df["winner_id"] = df["winner_row"].map({1: 1.0 if swap else 0.0, 2: 0.0 if swap else 1.0})
        df["points_0"], df["points_1"] = df[f"points_{first}"], df[f"points_{second}"]
        df["games_0"], df["games_1"] = df[f"games_{first}"], df[f"games_{second}"]
    else:
        df[["winner_id", "points_0", "points_1", "games_0", "games_1"]] = np.nan
    return df, {"votes": votes, "vote_share": round(share, 3), "rows_mapped": bool(mapped), "row1_is_player": int(swap)}


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV via a temporary file, so a failure never leaves a half-written one behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# Config keys only the rows -> players vote uses: changing them redoes the vote, not the video read.
MAPPING_KEYS = ("server_frames", "server_gap_m", "min_votes", "min_vote_share", "clear_landing_m", "landing_net_gap_m")
MAPPING_FILES = ("contacts.csv", "shots.csv", "players.parquet", "shuttle.csv", "homography.json")


def _stamp(path: Path) -> list[int] | None:
    return [path.stat().st_size, path.stat().st_mtime_ns] if path.exists() else None


def _inputs_stamp(cfg: Config, match_id: str) -> dict:
    """What `rallies.csv` depends on: `reading` (the video read: rallies, config, schema) and
    `mapping` (the rows -> players vote: phase 4-5 outputs and the vote's settings)."""
    d = Path(cfg.paths.cache_dir) / match_id
    conf = cfg.outcomes.to_dict()
    return json.loads(json.dumps({
        "reading": {"schema": SCHEMA_VERSION, "segments": _stamp(d / "segments.csv"),
                    "config": {k: v for k, v in conf.items() if k not in MAPPING_KEYS}},
        "mapping": {"files": {f: _stamp(d / f) for f in MAPPING_FILES},
                    "config": {k: conf.get(k) for k in MAPPING_KEYS}},
    }))


def _write_json_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


# --- The stage -------------------------------------------------------------------------------


def read_display(stills: np.ndarray, trust: np.ndarray, cfg_o: Config) -> tuple[ScoreObject, pd.DataFrame, int] | None:
    """Find the score display in the rally stills and read every rally off it.

    Returns the display, its per-rally table (`decide`) and how many
    candidate displays there were; None when there is none. Only
    `search_box` (fractions of the frame) is searched: all 32 broadcasts put
    the score graphic top-left, and outside it are arena boards, line
    judges' heads and the crowd.
    """
    h, w = stills.shape[1:]
    bx0, by0, bx1, by1 = cfg_o.search_box
    region = np.zeros((h, w), bool)
    region[int(by0 * h):int(by1 * h), int(bx0 * w):int(bx1 * w)] = True
    best = None
    # Several ways to pick candidate pixels: every changing pixel, or only those
    # at two grey levels in `two_level_share` of the rallies (each listed share
    # tried). 2018-19 graphics sit on the crowd and need the two-level test — a
    # loose one for a game column that is crowd until its game starts; the
    # 2022-26 ones gray out finished games (a third level) and read best with
    # none.
    # ...and two bars for a cell's change: high (ignores a box that changes
    # shade when the serve changes hands), and lower for templates whose digits
    # contrast less with their box. Every combination is read; the best kept.
    settings = [(share, level) for level in cfg_o.cell_change_levels for share in [None, *cfg_o.two_level_shares]]
    for share, level in settings:
        cfg_v = Config({**cfg_o.to_dict(), "cell_change_level": level})
        ys, xs = candidate_pixels(stills, trust, cfg_v, region, share)
        if len(ys) == 0:
            continue
        s, t = stills[:, ys, xs], trust[:, ys, xs]      # (rallies, pixels): all later work is on these
        ch = np.stack([_changes(s, t, r, r + 1, cfg_v.change_level)[0] for r in range(len(s) - 1)], axis=1)
        groups = digit_groups(ys, xs, ch, cfg_v)
        obj = find_display(groups, s, t, cfg_v)
        if obj is None:
            continue
        dec = decide(obj, s, t, cfg_v, display_absent(obj, stills, trust, cfg_v))
        # A reading counts its decided rallies — less when nearly all go one
        # way: even a lopsided real match (21-5, 21-6) gives the loser ~20%; a
        # pair of spectators "decides" plenty, almost all for one row.
        won = dec["winner_row"].dropna().astype(int)
        merit = len(won) * min(1.0, min((won == 1).mean(), (won == 2).mean()) / 0.15) if len(won) else 0.0
        if best is None or merit > best[0]:
            best = (merit, obj, dec, len(groups))
    return None if best is None else best[1:]


def find_outcomes(cfg: Config, match_id: str, force: bool = False) -> Path:
    """Write `rallies.csv` (and `outcomes.meta.json`, `score_graphic.png`) for one match.

    Reused when it exists and was read from the same rallies (`segments.csv`),
    `outcomes:` config and file schema; if only the vote's inputs changed
    (phase 4-5 outputs, the vote settings), only the vote is redone.
    Otherwise (or with `force`) redone from the video.
    """
    out_dir = cache_dir(cfg, match_id)
    out = out_dir / "rallies.csv"
    meta_file = out_dir / "outcomes.meta.json"
    if out.exists() and meta_file.exists() and not force:
        try:
            had = json.loads(meta_file.read_text()).get("inputs") or {}
        except json.JSONDecodeError:
            had = {}
        now = _inputs_stamp(cfg, match_id)
        if had.get("reading") == now["reading"]:
            if had.get("mapping") != now["mapping"]:
                remap(cfg, match_id)
            return out
        print(f"outcomes {match_id}: rallies, config or schema changed since rallies.csv was made: redoing it")
    cfg_o = cfg.outcomes
    t0 = time.time()
    meta = json.loads((out_dir / "poses.meta.json").read_text())
    windows = rally_windows(cfg, match_id)
    stills, trust = rally_stills(meta["video"], windows, float(meta["fps"]), cfg_o)
    t_read = time.time() - t0
    found = read_display(stills, trust, cfg_o)
    if found is None:
        raise OutcomeError(f"{match_id}: no two-row score display found")
    obj, dec, n_objs = found

    df = pd.DataFrame({"rally_id": windows["rally_id"].to_numpy()})
    df["winner_row"] = dec["winner_row"].astype("float")
    for c in ("how", "game", "points_r1", "points_r2", "games_r1", "games_r2", "score_exact"):
        df[c] = dec[c].to_numpy()
    for c in ("winner_id", "points_0", "points_1", "games_0", "games_1"):
        df[c] = np.nan
    df, mapping = apply_mapping(cfg, match_id, df)
    _write_atomic(df[RALLY_COLUMNS], out)

    games = []
    for _, g in dec.groupby("game"):
        w = g["winner_row"].dropna()
        games.append([int((w == 1).sum()), int((w == 2).sum())])
    info = {
        "rallies": len(df), "decided": int(df["winner_row"].notna().sum()),
        "how": df["how"].value_counts().to_dict(),
        "display_box": list(obj.box), "cells_found": n_objs, "columns": len(obj.columns),
        **mapping,
        "games_by_row": games, "read_s": round(t_read, 1), "total_s": round(time.time() - t0, 1),
        "inputs": _inputs_stamp(cfg, match_id),
    }
    _write_json_atomic(info, meta_file)
    write_display_sheet(stills, obj, out_dir / "score_graphic.png")
    v = mapping["votes"]
    print(f"outcomes {match_id}: {info['decided']}/{info['rallies']} rallies decided, rows "
          f"{'mapped' if mapping['rows_mapped'] else 'NOT mapped'} ({mapping['vote_share']:.0%} of "
          f"{v['same'] + v['swapped']} votes), games by row {games}, {info['total_s']} s")
    return out


def check_scores(cfg: Config, scores_csv: str | Path) -> pd.DataFrame:
    """Each match's games as read (points won per row) against its real final score.

    `scores_csv`: `match_id, player, games` with games like "21-14 18-21 21-16".
    Rows are compared either way round (which row is which player isn't known
    here), the same way for every game. `points_off` is the total of the
    per-game differences, plus 21 for each game missing or extra.
    """
    real = pd.read_csv(scores_csv)
    out = []
    for r in real.itertuples():
        path = Path(cfg.paths.cache_dir) / r.match_id / "rallies.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        got = [(int((g["winner_row"] == 1).sum()), int((g["winner_row"] == 2).sum())) for _, g in df.groupby("game")]
        got = [g for g in got if sum(g)]
        want = [tuple(int(v) for v in g.split("-")) for g in r.games.split()]
        best = None
        for flip in (False, True):
            gg = [(b, a) if flip else (a, b) for a, b in got]
            off = sum(abs(a - c) + abs(b - d) for (a, b), (c, d) in zip(gg, want)) + 21 * abs(len(gg) - len(want))
            if best is None or off < best[0]:
                best = (off, gg)
        out.append({"match_id": r.match_id, "rallies": len(df), "decided": int(df["winner_row"].notna().sum()),
                    "rows_mapped": bool(df["winner_id"].notna().any()),
                    "games_read": " ".join(f"{a}-{b}" for a, b in best[1]), "games_real": r.games,
                    "points_off": best[0]})
    return pd.DataFrame(out)


def remap(cfg: Config, match_id: str) -> dict:
    """Redo only the rows -> players match of an existing `rallies.csv` (no video)."""
    out_dir = Path(cfg.paths.cache_dir) / match_id
    mapping = map_rows(cfg, match_id)
    meta_file = out_dir / "outcomes.meta.json"
    info = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    info.update(mapping)
    info.setdefault("inputs", {})["mapping"] = _inputs_stamp(cfg, match_id)["mapping"]
    _write_json_atomic(info, meta_file)
    v = mapping["votes"]
    print(f"outcomes {match_id}: rows {'mapped' if mapping['rows_mapped'] else 'NOT mapped'} "
          f"({mapping['vote_share']:.0%} of {v['same'] + v['swapped']} votes: {v['serve']} serve, {v['landing']} landing)")
    return mapping


def write_display_sheet(stills: np.ndarray, obj: ScoreObject, path: Path, n: int = 12) -> None:
    """The chosen display in `n` rallies, the row split marked: a quick visual check."""
    x0, y0, x1, y1 = obj.box
    pad = 20
    ys = slice(max(0, y0 - pad), y1 + pad)
    xs = slice(max(0, x0 - 4 * pad), x1 + pad)
    picks = np.linspace(0, len(stills) - 1, min(n, len(stills))).astype(int)
    tiles = []
    for k in picks:
        tile = cv2.cvtColor(stills[k][ys, xs], cv2.COLOR_GRAY2BGR)
        cy = int(obj.cut_y) - ys.start
        cv2.line(tile, (0, cy), (tile.shape[1] - 1, cy), (0, 0, 255), 1)
        tiles.append(tile)
    cv2.imwrite(str(path), np.vstack(tiles))
