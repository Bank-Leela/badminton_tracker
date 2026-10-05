import json
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest

from config import load_config
from outcomes import _game_over, landing_winners, map_rows, read_display, row_votes, sample_positions

H, W = 720, 1280


def script_match(seed: int = 0, deuce: bool = False) -> list[tuple[int, int, int, int]]:
    """A two-game match, rally by rally: (game, row-1 points, row-2 points) shown before the rally, and its winner row.

    `deuce`: game 1 goes 20-20 then 22-20, game 2 goes 29-29 then 30-29.
    """
    rng = np.random.default_rng(seed)
    out, game, pts = [], 1, [0, 0]
    while game <= 2:
        if deuce:
            cap = 20 if game == 1 else 29
            if min(pts) < cap:
                w = 1 if pts[0] <= pts[1] else 2        # alternate up to cap-cap
            else:
                w = 1                                   # then row 1 pulls away / takes the 30th
        else:
            w = 1 if rng.random() < 0.55 else 2
        out.append((game, pts[0], pts[1], w))
        pts[w - 1] += 1
        if _game_over(*pts):
            game, pts = game + 1, [0, 0]
    return out


class Crowd:
    """Spectators above the court: a fixed picture where a few people shift between rallies."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.img = rng.integers(30, 120, (H // 8, W // 8)).astype(np.uint8)

    def next(self) -> np.ndarray:
        for _ in range(12):
            y, x = int(self.rng.integers(0, H // 8)), int(self.rng.integers(0, W // 8))
            self.img[y, x] = int(self.rng.integers(30, 160))
        return cv2.resize(self.img, (W, H), interpolation=cv2.INTER_NEAREST)


def draw(cols: list[tuple[int, int]], crowd: Crowd, banner: bool = False, blank: bool = False) -> np.ndarray:
    """One rally's still: the crowd, and the score graphic top-left over it (one column per entry of `cols`)."""
    img = crowd.next()
    if blank:
        return img
    if banner:
        cv2.rectangle(img, (20, 10), (520, 100), 230, -1)
        cv2.putText(img, "GAME POINT", (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 20, 3)
        return img
    cv2.rectangle(img, (20, 10), (300, 100), 235, -1)     # name panel
    cv2.putText(img, "PLAYER A", (32, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2)
    cv2.putText(img, "PLAYER B", (32, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 20, 2)
    for k, (a, b) in enumerate(cols):
        x0 = 308 + 72 * k
        cv2.rectangle(img, (x0, 10), (x0 + 64, 100), 15, -1)
        for v, y in ((a, 46), (b, 92)):                   # centred in the box, as broadcasts do
            (tw, _), _ = cv2.getTextSize(str(v), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.putText(img, str(v), (x0 + 32 - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    return img


def make_stills(style: str, rallies, banner_at=(), blank_at=(), seed: int = 0):
    """style 'reset': one points column, reset to 0-0 each game (2018-19).
    style 'columns': a column per game; a finished game keeps its final score (2022-26)."""
    crowd = Crowd(np.random.default_rng(seed + 1))
    final = {}
    for game, p1, p2, w in rallies:
        final[game] = (p1 + (w == 1), p2 + (w == 2))
    stills = []
    for k, (game, p1, p2, _) in enumerate(rallies):
        cols = [(p1, p2)] if style == "reset" else [final[g] for g in range(1, game)] + [(p1, p2)]
        stills.append(draw(cols, crowd, banner=k in banner_at, blank=k in blank_at))
    stills = np.stack(stills)
    return stills, np.ones_like(stills, bool)


def oriented(dec: pd.DataFrame, truth: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """The read table and winners with rows flipped if the reader numbered them the other way."""
    got = dec["winner_row"].astype(float).to_numpy()
    if np.nanmean(got[:-1] == truth[:-1]) >= 0.5:
        return dec, got
    d = dec.copy()
    d["points_r1"], d["points_r2"] = dec["points_r2"], dec["points_r1"]
    d["games_r1"], d["games_r2"] = dec["games_r2"], dec["games_r1"]
    return d, np.where(np.isnan(got), np.nan, 3 - got)


@pytest.fixture
def cfg_o():
    return load_config().outcomes


@pytest.mark.parametrize("deuce", [False, True])
@pytest.mark.parametrize("style", ["reset", "columns"])
def test_reads_every_point_the_score_and_the_games(cfg_o, style, deuce):
    rallies = script_match(0, deuce)
    stills, trust = make_stills(style, rallies)
    _, dec, _ = read_display(stills, trust, cfg_o)
    truth = np.array([w for *_, w in rallies], float)
    dec, got = oriented(dec, truth)
    # Every rally but the last (no next state to compare with) is read, and read right.
    assert np.isfinite(got[:-1]).all()
    assert (got[:-1] == truth[:-1]).all()
    # The replayed score before each rally, the games, all exact.
    assert dec["game"].tolist() == [g for g, *_ in rallies]
    assert dec["points_r1"].tolist() == [p for _, p, _, _ in rallies]
    assert dec["points_r2"].tolist() == [p for _, _, p, _ in rallies]
    assert dec["score_exact"].all()
    end = max(k for k, (g, *_) in enumerate(rallies) if g == 1)
    won_g1 = int(truth[end])
    assert (dec.iloc[-1]["games_r1"], dec.iloc[-1]["games_r2"]) == ((1, 0) if won_g1 == 1 else (0, 1))
    # The game-ending rally: in the reset style only the rules can tell who won
    # it (both rows go to 0); in the columns style the old column shows it.
    assert dec["how"].iloc[end] == ("game_end" if style == "reset" else "digits")


@pytest.mark.parametrize("style", ["reset", "columns"])
def test_a_covered_rally_is_bridged_when_one_player_won_both(cfg_o, style):
    rallies = script_match(3)
    truth = np.array([w for *_, w in rallies], float)
    both = [k for k in range(1, 30) if truth[k - 1] == truth[k]][0]          # covered rally, same winner either side
    stills, trust = make_stills(style, rallies, banner_at=(both,), seed=3)
    _, dec, _ = read_display(stills, trust, cfg_o)
    dec, got = oriented(dec, truth)
    known = np.isfinite(got[:-1])
    assert (got[:-1][known] == truth[:-1][known]).all()                      # no exemption: bridged points are right too
    assert dec["how"].iloc[both - 1] == dec["how"].iloc[both] == "skip"
    assert known.mean() >= 0.95
    assert dec["score_exact"].all()


@pytest.mark.parametrize("style", ["reset", "columns"])
def test_a_covered_rally_with_a_split_is_left_unread(cfg_o, style):
    rallies = script_match(3)
    truth = np.array([w for *_, w in rallies], float)
    split = [k for k in range(1, 30) if truth[k - 1] != truth[k]][0]         # each player won one of the two points
    stills, trust = make_stills(style, rallies, blank_at=(split,), seed=3)
    _, dec, _ = read_display(stills, trust, cfg_o)
    dec, got = oriented(dec, truth)
    assert np.isnan(got[split - 1]) and np.isnan(got[split])                 # which was whose is unknown
    known = np.isfinite(got[:-1])
    assert (got[:-1][known] == truth[:-1][known]).all()
    # No game was invented; the score is exact up to the gap and flagged after it.
    assert dec["game"].tolist() == [g for g, *_ in rallies]
    assert dec["score_exact"].iloc[:split].all() and not dec["score_exact"].iloc[split + 1]


def test_no_display_no_answer(cfg_o):
    crowd = Crowd(np.random.default_rng(0))
    stills = np.stack([draw([], crowd, blank=True) for _ in range(60)])
    assert read_display(stills, np.ones_like(stills, bool), cfg_o) is None


def test_sample_positions_stay_early_in_the_rally(cfg_o):
    pos = sample_positions(1000, 1500, 25.0, cfg_o)           # a 20 s rally
    assert pos[0] >= 1000 + 0.3 * 25 - 1 and pos[-1] <= 1000 + 6.0 * 25 + 1
    short = sample_positions(1000, 1050, 25.0, cfg_o)         # a 2 s rally: not past 70% of it
    assert max(short) <= 1000 + 0.7 * 50 + 1


def test_landing_vote_reads_in_out_and_skips_the_net(tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    d = tmp_path / "m"
    d.mkdir()
    # Image == court metres x 100, offset so everything is positive (H maps image -> court).
    Hm = np.array([[0.01, 0, -5.0], [0, 0.01, -10.0], [0, 0, 1]])
    (d / "homography.json").write_text(json.dumps(
        {"segments": [{"segment_id": 0, "start_frame": 0, "end_frame": 10_000, "image_to_court": Hm.tolist()}]}))
    to_img = lambda x, y: ((x + 5.0) * 100, (y + 10.0) * 100)
    # Rallies 0-4: id 0 near (court y < 0). Rallies 5-7: the ends have changed, id 0 far.
    ends = {r: ((0, "near"), (1, "far")) if r < 5 else ((0, "far"), (1, "near")) for r in range(8)}
    pd.DataFrame([{"rally_id": r, "hitter_id": h, "hitter_side": s} for r, e in ends.items() for h, s in e]
                 ).to_csv(d / "shots.csv", index=False)
    rest = {0: (0.0, -5.0),     # in, near half: near player (0) lost -> 1 won
            1: (0.0, 7.5),      # out past the far baseline: far player (1) won
            2: (0.5, 1.0),      # by the net: may be on the tape, not used
            3: (2.5, -5.0),     # 9 cm inside the sideline: too close to call
            4: (-3.2, 4.0),     # out wide on the far side: far player (1) won
            5: (0.0, -5.0),     # ends changed: in on the near half, near player is 1 -> 0 won
            6: (0.0, 7.5),      # out past the far baseline, far player is 0 -> 0 won
            7: (1.0, 4.0)}      # in on the far half, far player is 0 -> 1 won
    pd.DataFrame([{"rally_id": r, "kind": "landing", "frame": 100 * (r + 1), "x": 0.0, "y": 0.0} for r in rest]
                 ).to_csv(d / "contacts.csv", index=False)
    pd.DataFrame([{"frame": 100 * (r + 1), "x": to_img(*rest[r])[0], "y": to_img(*rest[r])[1], "visible": 1}
                  for r in rest]).to_csv(d / "shuttle.csv", index=False)
    got = landing_winners(cfg, "m", cfg.outcomes)
    assert got == {0: 1, 1: 1, 4: 1, 5: 0, 6: 0, 7: 1}
    # Rows the other way round from ids: row 1 won where player 1 did -> all 'swapped'.
    votes = row_votes(pd.Series([1, 1, 2, None, 1, 2, 2, 1]), np.arange(8), got, {})
    assert (votes["same"], votes["swapped"], votes["landing"]) == (0, 6, 6)


def test_rows_matched_to_players_by_who_serves_next(tmp_path):
    """The winner serves next, and the server stands nearer the centre line."""
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}", "outcomes.min_votes=5"])
    d = tmp_path / "m"
    d.mkdir()
    rng = np.random.default_rng(0)
    n = 30
    winner_row = rng.integers(1, 3, n)                 # the display's rows; row 1 is player 1 here
    server_id = [0] + [1 if w == 1 else 0 for w in winner_row[:-1]]
    pd.DataFrame([(0, 100 * k, 100 * k + 50, 1, k) for k in range(n)],
                 columns=["segment_id", "start_frame", "end_frame", "is_play", "rally_id"]).to_csv(d / "segments.csv", index=False)
    rows = []
    for k in range(n):
        for f in range(100 * k, 100 * k + 5):
            for pid in (0, 1):
                rows.append({"frame": f, "player_id": pid, "court_x": 0.2 if pid == server_id[k] else 1.4})
    pd.DataFrame(rows).to_parquet(d / "players.parquet")
    pd.DataFrame(columns=["rally_id", "kind", "frame", "x", "y"]).to_csv(d / "contacts.csv", index=False)
    pd.DataFrame(columns=["rally_id", "hitter_id", "hitter_side"]).to_csv(d / "shots.csv", index=False)
    pd.DataFrame(columns=["frame", "x", "y", "visible"]).to_csv(d / "shuttle.csv", index=False)
    (d / "homography.json").write_text(json.dumps({"segments": []}))
    df = pd.DataFrame({"rally_id": np.arange(n), "winner_row": winner_row.astype(float), "how": "digits", "game": 1,
                       "points_r1": np.arange(n), "points_r2": 0, "games_r1": 0, "games_r2": 0, "score_exact": True})
    for c in ("winner_id", "points_0", "points_1", "games_0", "games_1"):
        df[c] = np.nan
    df.to_csv(d / "rallies.csv", index=False)
    info = map_rows(cfg, "m")
    assert info["rows_mapped"] and info["row1_is_player"] == 1 and info["votes"]["serve"] == n - 1
    out = pd.read_csv(d / "rallies.csv")
    assert (out["winner_id"] == np.where(winner_row == 1, 1.0, 0.0)).all()
    assert (out["points_1"] == out["points_r1"]).all() and (out["points_0"] == out["points_r2"]).all()


def test_an_old_rallies_file_asks_for_a_rerun(tmp_path):
    from outcomes import OutcomeError
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    (tmp_path / "m").mkdir()
    pd.DataFrame({"rally_id": [0], "winner_row": [1.0], "winner_id": [0.0], "how": ["digits"], "game": [1],
                  "points_0": [0], "points_1": [0], "games_0": [0], "games_1": [0], "score_exact": [True]}
                 ).to_csv(tmp_path / "m" / "rallies.csv", index=False)
    with pytest.raises(OutcomeError, match="--force"):
        map_rows(cfg, "m")


# --- The replay, from scripted readings -------------------------------------------------------


def replay(monkeypatch, codes, bridges=None):
    """Run decide() on scripted per-transition readings: codes[r] = (code, new column opened)."""
    import outcomes
    bridges = bridges or {}
    monkeypatch.setattr(outcomes, "verdict", lambda obj, s, t, i, j, cfg_o: codes[i] if j == i + 1 else bridges[(i, j)])
    n = len(codes) + 1
    obj = SimpleNamespace(active=[])         # one reset column: game ends show as resets
    return outcomes.decide(obj, np.zeros((n, 1)), np.ones((n, 1), bool), load_config().outcomes)


def points(*rows):
    return [(str(r), False) for r in rows]


def test_a_stray_new_column_mid_game_is_not_a_game_end(monkeypatch):
    dec = replay(monkeypatch, points(1, 2) + [("1", True)] + points(2, 1, 1))
    assert (dec["game"] == 1).all() and dec["score_exact"].all()
    assert dec["points_r1"].tolist() == [0, 1, 1, 2, 2, 3, 4]


def test_a_reset_with_a_rally_missing_still_ends_the_game(monkeypatch):
    # Row 1 really wins 21-15, but phase 2 merged two of its points: the count reads 19-15 at the reset.
    codes = points(*([1] * 19 + [2] * 15)) + [("b", False)] + points(2, 2, 1)
    dec = replay(monkeypatch, codes)
    assert dec["game"].tolist() == [1] * 35 + [2] * 4
    assert dec["winner_row"].iloc[34] == 1 and dec["how"].iloc[34] == "game_end_lead"
    assert dec["games_r1"].isna().iloc[35:].all()                  # who won was a guess: the tally stops
    assert dec["score_exact"].iloc[35:].all()                       # the new game's start was seen


def test_a_short_count_waits_for_the_reset(monkeypatch):
    # Two of row 1's points never made it into the rallies; true 21-21 reads 19-21 on a plain point, then
    # the game goes on to 23-21 and the reset shows.
    codes = points(*([1, 2] * 19 + [2, 2])) + points(1, 1, 1) + [("b", False)] + points(2)
    dec = replay(monkeypatch, codes)
    reset_at = len(codes) - 2
    assert (dec["game"].iloc[:reset_at + 1] == 1).all() and dec["game"].iloc[reset_at + 1] == 2
    assert not dec["score_exact"].iloc[41:reset_at + 1].any()      # the count was over but the game went on


def test_a_split_under_a_banner_at_game_point_is_not_the_games_end(monkeypatch):
    # 19-18; a banner covers rally 38: rallies 37 and 38 split (20-19), then 20-20, 21-20, 22-20 and the reset.
    codes = points(*([1, 2] * 18 + [1])) + [("?", False), ("?", False)] + points(2, 1, 1) + [("b", False)] + points(2)
    dec = replay(monkeypatch, codes, bridges={(37, 39): ("b", False)})
    assert np.isnan(dec["winner_row"].iloc[37]) and np.isnan(dec["winner_row"].iloc[38])
    end = len(codes) - 2
    assert (dec["game"].iloc[:end + 1] == 1).all() and dec["game"].iloc[end + 1] == 2
