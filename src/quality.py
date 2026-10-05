"""Phase 7: the shot-quality model.

One LightGBM multiclass model over the labeller's five outcome classes
(1 outright winner or opponent error ... 5 hitter erred). It outputs the
whole distribution; expected quality, risk and a good / bad / risky / neutral
verdict are derived from it in a thin layer (`derive`) that is where game
state will come in once its rules are decided. Classes 1..5 are indices 0..4
inside this module; files and reports use 1..5 (`p1`..`p5`).

Inputs: `cfg.quality.features`, shots.csv columns plus `hitter_near`, from
an allow-list (`allowed_input`; anything else is a `LeakageError`): what is
known at contact (`CONTACT_INPUTS`), and the `early_*` columns measured over
the same short window after it for every shot — not the early fit's QA
(`EARLY_QA`: how many track points the window held depends on when the next
event came). `LEAKY` names the worst: `ended` and `landing_src` say outright
whether the shot came back, the other after-contact columns — the flight's
QA (`fit_rms_px`, `fit_n_obs`: fitted up to the reply or the floor;
`fit_rms_px` alone predicts `ended` with AUC 0.78) and the contact's
(`contact_reach`, `contact_gap`: from track pieces spanning the outgoing
flight) included — are measured up to the reply or the floor (so how far
they got, and whether they exist at all, gives it away: grouped-CV AUC 0.99
for `ended`), and `hitter_recovery_time` / `is_serve` are unreliable. The
labeller never sees the reply. Missing values stay NaN — LightGBM learns
which way to send them, and imputing a court position would invent one. The
early depth columns of a far player's shot are blanked
(`quality.far_unmeasured`; a saved model keeps the list it was trained with).
A leak check (`leak_check`, in the report) guards the rest: which inputs are
missing must not predict `ended` (`quality.max_missing_auc`).

Labels. Hand labels from phase 6 (`x` dropped). Free baseline labels from
the rally outcomes in `rallies.csv`: the result credited backwards from the
last shot with a discount per shot (`baseline_labels`); when the shuttle came
to rest on the last detected hitter's own side, away from the net, at least
`quality.baseline.min_missed_gap_s` after that hit (time for two flights),
the true final hit was missed and the credit starts one shot further back
(`missed_final_hits`). The plan's acceptance check (`compare`): the
hand-labelled model must beat a model trained on those — with its outputs
moved to the hand labels' class prior, so a prior that merely matches the
hand labels better doesn't pass — and the hand-label prior itself, or the
features are wrong.

The model is `lightgbm.train` with an explicit `num_class=5`, not
`LGBMClassifier`. The classifier label-encodes only the classes it sees, so a
fold without a class-1 shot (~50 in 1000 labels) would give a 4-column
`predict_proba` in shifted order. With `num_class=5` an absent class is just
never predicted and every matrix is 5 columns summing to 1.
`class_weight: balanced` becomes per-row weights n / (k x count) over the k
classes present — sklearn's formula. The early-stopping set is weighted the
same way over its own classes, so early stopping minimises the objective that
is being trained. Balanced training is training under a uniform class prior,
so the raw outputs follow that prior (an uninformative model says ~0.2 each,
and everything looks risky); `QualityModel.predict_proba` maps them back to
the training prior (p_k proportional to raw p_k x training share of k) and
then blends in `quality.proba_floor` (P = (1 - a) P + a / 5), so a class
absent from a training fold costs ln(5 / a) on a test shot, not ~35 nats.
Every score, verdict and file sees these probabilities. The Booster stays
reachable (`QualityModel.booster`) for phase 8's SHAP.

Evaluation is split by match, never by shot (shots in a rally are
correlated): GroupKFold over matches, and inside each training fold ~15 % of
its matches (at least one) held out whole for early stopping. The primary
score is multiclass log loss — it scores the whole distribution, which is what
the model is for (of the prior-corrected, floored probabilities, like
everything else). Balanced log loss (the mean over classes), macro-F1,
balanced accuracy, per-class precision / recall / support and the confusion
matrix ride along: class 1 is rare and the risk score leans on it most. The
class prior it is compared with is floored the same way.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import Config, cache_dir
from labeler import CLASSES as LABEL_CLASSES
from labeler import load_labels

CLASSES = (1, 2, 3, 4, 5)
N_CLASSES = len(CLASSES)
PROBA_COLUMNS = [f"p{c}" for c in CLASSES]
SOURCES = ("hand", "baseline")
VERDICTS = ("good", "neutral", "bad", "risky")
# Never model inputs. Whether the shot came back (known only after the clip
# the labeller sees ends), the columns measured up to the next event (the
# reply or the floor: how far they got gives the answer away) — the flight
# fit's QA and the contact's QA (from track pieces spanning the outgoing
# flight) too — and two that are unreliable (`hitter_recovery_time`;
# `is_serve` marks the far player's first hit, not the serve).
LEAKY = ("ended", "landing_src", "landing_depth", "landing_lateral", "dist_from_lines", "flight_time",
         "shot_length", "avg_speed", "cross_court", "net_clearance", "shuttle_speed", "opponent_dist",
         "opponent_toward", "fit_rms_px", "fit_n_obs", "contact_reach", "contact_gap",
         "hitter_recovery_time", "is_serve")
# What a model input may be (`allowed_input`): known at contact...
CONTACT_INPUTS = ("shot_index", "time_since_prev_contact", "hitter_near", "hitter_depth", "hitter_lateral",
                  "hitter_speed", "contact_height", "hitter_lean", "stance_width", "opponent_depth",
                  "opponent_lateral", "opponent_speed")
# ...or an `early_*` column, but not the early fit's QA: how many track points
# the early window held depends on when the next event came (the window stops
# short of it), and how well they fit is the tracker's, not the shot's.
EARLY_QA = ("early_fit_rms_px", "early_fit_n_obs")
RALLY_COLUMNS = ["rally_id", "winner_id", "game", "points_0", "points_1", "games_0", "games_1"]
# `score_exact`: False when a rally earlier in the game wasn't read (the score may be off).
STATE_COLUMNS = ["game", "hitter_points", "opponent_points", "hitter_games", "opponent_games", "score_exact"]
ID_COLUMNS = ["match_id", "rally_id", "shot_index", "frame", "hitter_id", "hitter_side"]
QUALITY_COLUMNS = [*ID_COLUMNS, *PROBA_COLUMNS, "expected_quality", "risk", "verdict", *STATE_COLUMNS]
# Share of a training fold's matches held out for early stopping, when the
# config has no `quality.early_stopping_frac`.
ES_FRAC = 0.15
# Least time from the last detected hit to the landing for a rest point on
# that hitter's own side to count as a missed final hit (two flights: theirs
# and the missed reply), when the config has no
# `quality.baseline.min_missed_gap_s`. Sooner, it is a net or short error.
MIN_MISSED_GAP_S = 0.8
# The hand learning curve's reference line: the acceptance check's baseline.
REFERENCE_BASELINE = "baseline model, recalibrated"


class Landing(NamedTuple):
    """Where and when a rally's shuttle first came to rest (`load_landings`).

    Court metres (near half y < 0); the landing frame (the start of the rest,
    `contacts.csv`) and the match's fps (`poses.meta.json`). A bare `(x, y)`
    works as a Landing with no time: no missed hit can be inferred from it.
    """

    x: float
    y: float
    frame: float = math.nan
    fps: float = math.nan


class LeakageError(AssertionError):
    """A model input says whether the shot was returned (raised explicitly, like `features.FeatureCheckError`)."""


class MissingFeatures(SystemExit):
    """Configured inputs absent from shots.csv: phase 5 must be rerun (exits with the message, like `load_shots`)."""


class TooFewLabels(ValueError):
    """Not enough labels, or labelled matches, for what was asked."""


# --- Data ------------------------------------------------------------------------------------


def match_ids(cfg: Config) -> list[str]:
    """Every match with a `shots.csv`, sorted."""
    return sorted(p.parent.name for p in Path(cfg.paths.cache_dir).glob("*/shots.csv"))


def load_shots(cfg: Config, matches: Iterable[str] | None = None,
               features: Sequence[str] | None = None) -> pd.DataFrame:
    """`shots.csv` of the chosen matches (all that have one by default), concatenated in match order.

    Every match's file must have the model inputs `features` (default
    `cfg.quality.features`): one written before a feature existed would
    otherwise come through as a match of NaN (`MissingFeatures`).
    """
    cache = Path(cfg.paths.cache_dir)
    ids = sorted(matches) if matches else match_ids(cfg)
    wanted = [c for c in (features if features is not None else cfg.quality.features) if c != "hitter_near"]
    frames, lacking = [], {}
    for m in ids:
        f = cache / m / "shots.csv"
        if not f.exists():
            raise SystemExit(f"no {f}; run `bda shots --match-id {m}` first")
        frames.append(pd.read_csv(f))
        gone = [c for c in wanted if c not in frames[-1].columns]
        if gone:
            lacking[m] = gone
    if not frames:
        raise SystemExit(f"no shots.csv under {cache}; run `bda shots` first")
    if lacking:
        cols = list(dict.fromkeys(c for gone in lacking.values() for c in gone))
        raise MissingFeatures(f"quality.features not in shots.csv of {len(lacking)} of {len(ids)} match(es) "
                              f"({', '.join(lacking)}): {cols}; rerun `bda shots` for them (phase 5), or drop "
                              "the columns from quality.features")
    shots = pd.concat(frames, ignore_index=True)
    shots["match_id"] = shots["match_id"].astype(str)
    shots["frame"] = shots["frame"].astype(int)
    dup = shots.duplicated(["match_id", "frame"])
    if dup.any():  # labels join on this key: a duplicate would copy a label onto another shot
        raise ValueError(f"{int(dup.sum())} shots share a (match_id, frame) key with another")
    return shots


def allowed_input(column: str) -> bool:
    """May `column` be a model input? Known at contact (`CONTACT_INPUTS`), or `early_*` but not `EARLY_QA`."""
    return column in CONTACT_INPUTS or (column.startswith("early_") and column not in EARLY_QA)


def _far_unmeasured(cfg: Config) -> list[str]:
    """`quality.far_unmeasured`: the inputs left unmeasured (NaN) for a far player's shot."""
    return [str(c) for c in (cfg.quality.get("far_unmeasured") or [])]


def feature_matrix(shots: pd.DataFrame, cfg: Config, features: Sequence[str] | None = None,
                   far_unmeasured: Sequence[str] | None = None) -> pd.DataFrame:
    """The model inputs, float, NaN where unmeasured: `features` (default `cfg.quality.features`), in order.

    Only inputs on the allow-list (`allowed_input`); anything else is a
    `LeakageError`. `far_unmeasured` (default `quality.far_unmeasured`; a
    saved model passes the list it was trained with): inputs blanked for far
    shots.
    """
    cols = list(features if features is not None else cfg.quality.features)
    leaked = [c for c in cols if c in LEAKY]
    if leaked:
        raise LeakageError(f"model inputs include {leaked}: measured up to the reply or the floor — the flight's "
                           "and the contact's QA columns too — (they give away whether the shot was returned, which "
                           "the labeller never sees) or unreliable; use the early_* columns")
    other = [c for c in cols if not allowed_input(c)]
    if other:
        raise LeakageError(f"model inputs include {other}, not on the allow-list: only what is known at contact "
                           f"({', '.join(CONTACT_INPUTS)}) and the early_* columns, except the early fit's QA "
                           f"({', '.join(EARLY_QA)}), may be inputs — anything else may tell whether the shot was "
                           "returned. Drop them from quality.features (or, if one really is known at contact, add it "
                           "to quality.CONTACT_INPUTS)")
    if len(set(cols)) != len(cols):
        raise ValueError(f"quality.features lists a column twice: {cols}")
    missing = [c for c in cols if c != "hitter_near" and c not in shots.columns]
    if missing:
        raise MissingFeatures(f"quality.features not in shots.csv: {missing}; rerun `bda shots` (phase 5) to add "
                              "them, or drop them from quality.features")
    # The early flight can't place a far player's shot in depth (it comes
    # straight at the camera; measured: worse than a constant guess): those
    # columns are left unmeasured for far shots (`quality.far_unmeasured`).
    far = (shots["hitter_side"] == "far").to_numpy() if "hitter_side" in shots.columns else np.zeros(len(shots), bool)
    blank_far = set(_far_unmeasured(cfg) if far_unmeasured is None else (str(c) for c in far_unmeasured))
    data = {}
    for c in cols:
        if c == "hitter_near":
            side = shots["hitter_side"]
            data[c] = np.where(side.isna(), np.nan, (side == "near").to_numpy(float))
        else:
            data[c] = pd.to_numeric(shots[c], errors="coerce").to_numpy(float)
            if c in blank_far:
                data[c] = np.where(far, np.nan, data[c])
    return pd.DataFrame(data, index=shots.index, columns=cols)


def load_rallies(cfg: Config, matches: Iterable[str]) -> pd.DataFrame:
    """`rallies.csv` of each match that has one, with `match_id`; a match without one is noted, not an error.

    `score_exact` (nullable boolean) is <NA> for a file written before it
    existed (noted).
    """
    cache = Path(cfg.paths.cache_dir)
    frames, missing, no_exact = [], [], []
    ids = sorted(matches)
    for m in ids:
        f = cache / m / "rallies.csv"
        if not f.exists():
            missing.append(m)
            continue
        r = pd.read_csv(f)
        lacking = [c for c in RALLY_COLUMNS if c not in r.columns]
        if lacking:
            raise ValueError(f"{f}: no column(s) {lacking}")
        if r["rally_id"].duplicated().any():
            raise ValueError(f"{f}: a rally_id appears twice")
        r = r.drop(columns="match_id", errors="ignore")  # the directory names the match
        r.insert(0, "match_id", m)
        if "score_exact" not in r.columns:
            no_exact.append(m)
            r["score_exact"] = pd.NA
        r["score_exact"] = _as_boolean(r["score_exact"])
        frames.append(r)
    if missing:
        print(f"quality: no rallies.csv for {len(missing)} of {len(ids)} matches ({', '.join(missing)}): "
              "no baseline labels or game state there")
    if no_exact:
        print(f"quality: rallies.csv without score_exact for {len(no_exact)} match(es) ({', '.join(no_exact)}): "
              "left blank; rerun `bda outcomes`")
    if not frames:
        return pd.DataFrame({"match_id": pd.Series(dtype=str), "rally_id": pd.Series(dtype=int),
                             **{c: pd.Series(dtype=float) for c in RALLY_COLUMNS[1:]},
                             "score_exact": pd.Series(dtype="boolean")})
    out = pd.concat(frames, ignore_index=True)
    out["match_id"] = out["match_id"].astype(str)
    out["rally_id"] = out["rally_id"].astype(int)
    for c in RALLY_COLUMNS[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out["score_exact"] = _as_boolean(out["score_exact"])
    return out


def load_landings(cfg: Config, matches: Iterable[str]) -> dict[tuple[str, int], Landing]:
    """`(match_id, rally_id) -> Landing`: where and when each rally's shuttle first came to rest.

    Where: `outcomes.landing_points` (court metres, near half y < 0). When:
    the rally's landing frame in `contacts.csv` and the match's fps in
    `poses.meta.json`. A match without the files it needs (contacts.csv,
    shuttle.csv, homography.json) adds nothing (noted): its baseline labels
    are not shifted (`missed_final_hits`); one without poses.meta.json adds
    landings with no time (noted), which shift nothing either.
    """
    from outcomes import landing_points  # cv2 and the video reader: only when needed

    out, missing, no_fps = {}, [], []
    ids = sorted({str(m) for m in matches})
    for m in ids:
        try:
            pts = landing_points(cfg, m)
        except FileNotFoundError:
            missing.append(m)
            continue
        d = Path(cfg.paths.cache_dir) / m
        c = pd.read_csv(d / "contacts.csv", usecols=["rally_id", "kind", "frame"])
        frames = c[c["kind"] == "landing"].groupby("rally_id")["frame"].min()
        frames.index = frames.index.astype(int)
        try:
            fps = float(json.loads((d / "poses.meta.json").read_text())["fps"])
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            fps = math.nan
        if not fps > 0:
            fps = math.nan
            no_fps.append(m)
        for r, xy in pts.items():
            out[(m, int(r))] = Landing(float(xy[0]), float(xy[1]), float(frames.get(int(r), math.nan)), fps)
    if missing:
        print(f"quality: no landing points for {len(missing)} of {len(ids)} matches ({', '.join(missing)}; no "
              "contacts.csv / shuttle.csv / homography.json): baseline labels there not checked for a missed final hit")
    if no_fps:
        print(f"quality: no fps for {len(no_fps)} of {len(ids)} matches ({', '.join(no_fps)}; no poses.meta.json): "
              "landings there have no time, so no missed final hit is inferred")
    return out


def min_missed_gap_s(cfg: Config) -> float:
    """`quality.baseline.min_missed_gap_s` (default `MIN_MISSED_GAP_S`), checked."""
    t = float(cfg.quality.baseline.get("min_missed_gap_s", MIN_MISSED_GAP_S))
    if not t >= 0:
        raise ValueError(f"quality.baseline.min_missed_gap_s must be at least 0, got {t}")
    return t


# --- Labels ----------------------------------------------------------------------------------


def _shots_after(s: pd.DataFrame) -> np.ndarray:
    """Per row, how many detected shots of its rally come after it (the last detected one: 0)."""
    return (s.groupby(["match_id", "rally_id"])["shot_index"].rank(method="first", ascending=False) - 1).to_numpy()


def missed_final_hits(shots: pd.DataFrame, landings: dict[tuple[str, int], Landing] | None,
                      cfg: Config) -> pd.Series:
    """Per shot (bool, aligned with `shots`): its rally's true final hit was not detected.

    The shuttle came to rest on the last detected hitter's own side (near:
    y < 0) at least `outcomes.landing_net_gap_m` from the net, and at least
    `quality.baseline.min_missed_gap_s` after that hit — time for their
    flight and the missed reply — so someone hit it back after them. Nearer
    the net it may be a net error falling on the hitter's own side; sooner,
    a net or short error (a shuttle hanging in the net a metre up projects
    deep into the far half through the floor homography): both left alone.
    No landing point, no landing time (a bare `(x, y)`), or unknown side:
    False.
    """
    out = np.zeros(len(shots), bool)
    if not landings or not len(shots):
        return pd.Series(out, index=shots.index)
    gap, min_s = float(cfg.outcomes.landing_net_gap_m), min_missed_gap_s(cfg)
    s = shots[["match_id", "rally_id", "shot_index", "frame", "hitter_side"]].reset_index(drop=True)
    s["match_id"] = s["match_id"].astype(str)
    k = _shots_after(s)
    missed = set()
    for i in np.flatnonzero(k == 0):
        m, rid, side = s.at[i, "match_id"], s.at[i, "rally_id"], s.at[i, "hitter_side"]
        if pd.isna(rid) or side not in ("near", "far"):
            continue
        p = landings.get((m, int(rid)))
        if p is None:
            continue
        p = Landing(*p)
        if not np.isfinite(p.y) or abs(p.y) < gap or (p.y < 0) != (side == "near"):
            continue
        after_s = (p.frame - float(s.at[i, "frame"])) / p.fps if p.fps > 0 else math.nan
        if after_s >= min_s:  # NaN (no time) is no evidence
            missed.add((m, rid))
    if missed:
        out = np.fromiter(((m, r) in missed for m, r in zip(s["match_id"], s["rally_id"])), bool, len(s))
    return pd.Series(out, index=shots.index)


def baseline_labels(shots: pd.DataFrame, rallies: pd.DataFrame, cfg: Config,
                    landings: dict[tuple[str, int], Landing] | None = None) -> pd.Series:
    """Free labels 1..5 (Int64, <NA> = unknown) from rally outcomes, aligned with `shots`.

    Per rally, k = shots after this one (the last shot k = 0); credit =
    (+1 if the hitter won the rally, else -1) x discount**k. |credit| at
    least cuts[0] gives 1 or 5 by its sign, at least cuts[1] 2 or 4, else 3.
    Unknown winner (or no rallies.csv) or unknown hitter: <NA>.

    `landings` (`(match_id, rally_id) -> Landing`, `load_landings`): a rally
    whose final hit was missed (`missed_final_hits`) counts k from that
    missing hit, one more for each of its shots — the last detected shot, by
    the player who then failed to return it or won with it, is not the end.
    """
    b = cfg.quality.baseline
    discount = float(b.discount)
    hi, lo = (float(c) for c in b.cuts)
    if not 0 <= discount <= 1:
        raise ValueError(f"quality.baseline.discount must be in [0, 1], got {discount}")
    if not hi >= lo > 0:
        raise ValueError(f"quality.baseline.cuts must be [high, low] with high >= low > 0, got {list(b.cuts)}")
    r = rallies
    if "match_id" not in r.columns:  # one match's rallies.csv as read
        if shots["match_id"].nunique() > 1:
            raise ValueError("rallies has no match_id but shots span several matches")
        r = r.assign(match_id=shots["match_id"].iloc[0] if len(shots) else "")
    s = shots[["match_id", "rally_id", "shot_index", "hitter_id"]].reset_index(drop=True)
    winner = s[["match_id", "rally_id"]].merge(r[["match_id", "rally_id", "winner_id"]], how="left",
                                               on=["match_id", "rally_id"], validate="many_to_one")["winner_id"]
    winner = pd.to_numeric(winner, errors="coerce").to_numpy(float)
    hitter = pd.to_numeric(s["hitter_id"], errors="coerce").to_numpy(float)
    k = _shots_after(s) + missed_final_hits(shots, landings, cfg).to_numpy(float)
    credit = np.where(hitter == winner, 1.0, -1.0) * discount ** k
    a = np.abs(credit) + 1e-12  # 0.9**2 lands a hair under a cut of 0.81
    cls = np.where(a >= hi, np.where(credit > 0, 1, 5), np.where(a >= lo, np.where(credit > 0, 2, 4), 3))
    out = pd.Series(cls, index=shots.index).astype("Int64")
    out[~(np.isfinite(winner) & np.isfinite(hitter) & np.isfinite(k))] = pd.NA
    return out


def hand_labels(cfg: Config) -> pd.DataFrame:
    """Hand labels 1..5 as ints (`match_id, frame, label`); `x` (not a shot / can't judge) dropped. Empty if none yet."""
    raw = load_labels(Path(cfg.paths.labels_dir) / "shot_labels.csv")
    keep = raw[raw["label"].isin([str(c) for c in CLASSES])]
    return pd.DataFrame({"match_id": keep["match_id"].astype(str).to_numpy(),
                         "frame": keep["frame"].astype(int).to_numpy(),
                         "label": keep["label"].astype(int).to_numpy()})


def labelled_shots(cfg: Config, source: str, shots: pd.DataFrame | None = None,
                   rallies: pd.DataFrame | None = None,
                   landings: dict[tuple[str, int], Landing] | None = None) -> pd.DataFrame:
    """The `shots` rows that have a `source` label, with it as `label` (int 1..5).

    The index is `shots`' own, so every row (and every feature-matrix row
    built from it) maps back to its shot. Baseline labels read `rallies`
    and `landings` (each loaded for the shots' matches when None).
    """
    _check_source(source)
    shots = load_shots(cfg) if shots is None else shots
    if source == "hand":
        lab = hand_labels(cfg)
        lab = lab[lab["match_id"].isin(set(shots["match_id"]))]
        row = "__shot_row__"
        df = shots.assign(**{row: shots.index.to_numpy()}).merge(lab, on=["match_id", "frame"], how="inner",
                                                                  validate="one_to_one")
        df = df.set_index(row).rename_axis(shots.index.name)
        if len(df) < len(lab):
            print(f"quality: {len(lab) - len(df)} hand labels match no shot in shots.csv "
                  "(shots rebuilt since labelling?); left out")
    else:
        if rallies is None:
            rallies = load_rallies(cfg, shots["match_id"].unique())
        if landings is None:
            landings = load_landings(cfg, shots["match_id"].unique())
        lab = baseline_labels(shots, rallies, cfg, landings)
        keep = lab.notna().to_numpy()
        df = shots[keep].copy()
        df["label"] = lab[keep].astype(int).to_numpy()
    return df


def support(y: Sequence[int] | np.ndarray) -> dict[str, int]:
    """Count per class `{"1": n, ..., "5": n}` from class indices 0..4."""
    n = np.bincount(np.asarray(y, int), minlength=N_CLASSES)
    return {str(c): int(n[i]) for i, c in enumerate(CLASSES)}


def proba_floor(cfg: Config) -> float:
    """`quality.proba_floor`, checked: the share of a uniform distribution blended into every prediction."""
    a = float(cfg.quality.proba_floor)
    if not 0 <= a < 1:
        raise ValueError(f"quality.proba_floor must be in [0, 1), got {a}")
    return a


def floor_proba(P: np.ndarray, floor: float) -> np.ndarray:
    """(1 - floor) P + floor / 5: every class keeps at least floor / 5, rows still sum to 1."""
    return (1.0 - floor) * np.asarray(P, float) + floor / N_CLASSES


def class_prior(y: np.ndarray, floor: float) -> np.ndarray:
    """Class frequencies (indices 0..4), floored like the model's (`floor_proba`): the no-features reference."""
    y = np.asarray(y, int)
    prior = np.bincount(y, minlength=N_CLASSES) / len(y) if len(y) else np.full(N_CLASSES, 1 / N_CLASSES)
    return floor_proba(prior, floor)


def shift_prior(P: np.ndarray, source: Sequence[float], target: Sequence[float], floor: float) -> np.ndarray:
    """Floored distributions `P` made under class shares `source`, moved to class shares `target`, floored again.

    Bayes' rule for a new prior, same evidence: the floor is taken out, then
    p_k x target_k / source_k (0 for a class absent from `source`, which the
    model never predicts), renormalised, and floored (`floor_proba`).
    `source` and `target` are counts or shares over classes 0..4; a row with
    nothing left becomes `target`.
    """
    src, tgt = np.asarray(source, float), np.asarray(target, float)
    if src.shape != (N_CLASSES,) or tgt.shape != (N_CLASSES,) or src.sum() <= 0 or tgt.sum() <= 0:
        raise ValueError(f"source and target must be {N_CLASSES} non-negative class shares, got {src} and {tgt}")
    src, tgt = src / src.sum(), tgt / tgt.sum()
    P0 = np.clip((np.asarray(P, float) - floor / N_CLASSES) / (1.0 - floor), 0.0, None)
    Q = P0 * np.divide(tgt, src, out=np.zeros(N_CLASSES), where=src > 0)
    total = Q.sum(axis=1, keepdims=True)
    Q = np.where(total > 0, Q / np.where(total > 0, total, 1.0), tgt)
    return floor_proba(Q, floor)


# --- The model -------------------------------------------------------------------------------


@dataclass
class QualityModel:
    """A fitted Booster and the inputs it takes; `booster` is what phase 8's SHAP explains.

    `train_counts`: rows per class (indices 0..4) the trees were grown on;
    with `class_weight` balanced, `predict_proba` maps the raw outputs back
    to those shares. `proba_floor`: `quality.proba_floor`, blended in last.
    `far_unmeasured`: the inputs blanked for far shots when it was trained
    (`quality.far_unmeasured` then): its inputs must be built the same way
    (`feature_matrix(..., far_unmeasured=model.far_unmeasured)`).
    (SHAP on `booster` explains the raw, balanced-prior scores; the prior
    correction adds the same log-share per class to every row.)
    """

    booster: lgb.Booster
    features: list[str]
    best_iteration: int
    train_counts: np.ndarray
    class_weight: str | None
    proba_floor: float
    far_unmeasured: list[str] = field(default_factory=list)

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        """The Booster's own (n, 5) softmax: under balanced weights, a uniform-prior distribution."""
        if len(X) == 0:
            return np.zeros((0, N_CLASSES))
        P = np.asarray(self.booster.predict(X[self.features].to_numpy(float), num_iteration=self.best_iteration), float)
        P = P.reshape(len(X), N_CLASSES)
        return P / P.sum(axis=1, keepdims=True)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 5) outcome distribution under the training prior, floored; column k is class k+1. Rows sum to 1."""
        P = self.raw_proba(X)
        if len(P) and _balanced(self.class_weight):
            # Balanced weights train under a uniform prior over the classes present:
            # p_k proportional to raw p_k x training share of k puts the training prior back.
            P = P * (np.asarray(self.train_counts, float) / max(float(np.sum(self.train_counts)), 1.0))
            P = P / P.sum(axis=1, keepdims=True)
        return floor_proba(P, self.proba_floor)


def lgb_params(cfg: Config) -> dict:
    """`lightgbm.train` parameters from `cfg.quality`."""
    m = cfg.quality.model
    threads = cfg.get("quality.model.num_threads")
    if threads is None:
        # LightGBM's default (every core) was ~10x slower than half of them
        # here (WSL, 10 cores): its threads spin-wait on each other.
        threads = max(1, (os.cpu_count() or 2) // 2)
    return {"objective": "multiclass", "num_class": N_CLASSES, "metric": "multi_logloss",
            "learning_rate": float(m.learning_rate), "num_leaves": int(m.num_leaves),
            "min_data_in_leaf": int(m.min_child_samples), "seed": int(cfg.quality.seed),
            "deterministic": True, "force_row_wise": True, "num_threads": int(threads), "verbosity": -1}


def _balanced(mode: str | None) -> bool:
    """`quality.model.class_weight` read: True for balanced, False for none / null; anything else is an error."""
    if mode is None or str(mode).lower() == "none":
        return False
    if mode != "balanced":
        raise ValueError(f"quality.model.class_weight must be balanced or null, got {mode!r}")
    return True


def sample_weights(y: np.ndarray, mode: str | None) -> np.ndarray | None:
    """Per-row weights for `quality.model.class_weight`: `balanced` = n / (k x count) over the k classes present."""
    if not _balanced(mode):
        return None
    y = np.asarray(y, int)
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts)
    return w[np.searchsorted(classes, y)]


def fit_model(X: pd.DataFrame, y: np.ndarray, cfg: Config, X_es: pd.DataFrame | None = None,
              y_es: np.ndarray | None = None, n_estimators: int | None = None) -> QualityModel:
    """Fit on class indices `y` (0..4); early stopping on (`X_es`, `y_es`) when given and non-empty.

    Without an early-stopping set it runs `n_estimators` rounds (default
    `quality.model.n_estimators`). The model keeps `y`'s class counts (its
    prior correction), `quality.proba_floor`, and `quality.far_unmeasured`
    (`X` is taken to be `feature_matrix(..., cfg)`'s).
    """
    y = np.asarray(y, int)
    if len(y) == 0:
        raise TooFewLabels("no training rows")
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError(f"class indices must be 0..{N_CLASSES - 1}, got {sorted(set(y.tolist()))}")
    m = cfg.quality.model
    mode = m.class_weight
    rounds = int(n_estimators if n_estimators is not None else m.n_estimators)
    train = lgb.Dataset(X.to_numpy(float), label=y, weight=sample_weights(y, mode), feature_name=list(X.columns))
    kwargs: dict = {}
    if X_es is not None and len(X_es):
        y_es = np.asarray(y_es, int)
        val = lgb.Dataset(X_es[list(X.columns)].to_numpy(float), label=y_es, weight=sample_weights(y_es, mode),
                          reference=train)
        kwargs = {"valid_sets": [val], "valid_names": ["early_stopping"],
                  "callbacks": [lgb.early_stopping(int(m.early_stopping_rounds), first_metric_only=True, verbose=False)]}
    booster = lgb.train(lgb_params(cfg), train, num_boost_round=rounds, **kwargs)
    best = int(booster.best_iteration) or int(booster.current_iteration())
    return QualityModel(booster, list(X.columns), best, np.bincount(y, minlength=N_CLASSES),
                        None if not _balanced(mode) else "balanced", proba_floor(cfg), _far_unmeasured(cfg))


# --- Splits and scores -----------------------------------------------------------------------


@dataclass
class Fold:
    """One split by whole matches: trees grow on `fit`, stop on `es`, are scored on `test`."""

    test: list[str]
    fit: list[str]
    es: list[str]


def _es_frac(cfg: Config) -> float:
    return float(cfg.get("quality.early_stopping_frac", ES_FRAC))


def split_early_stopping(matches: Iterable[str], frac: float, rng: np.random.Generator) -> tuple[list[str], list[str]]:
    """`(fit, es)`: ~`frac` of the matches (at least one) held out whole for early stopping.

    A single match cannot spare one: `es` is empty and the caller trains
    without early stopping.
    """
    ms = sorted({str(m) for m in matches})
    if len(ms) < 2:
        return ms, []
    k = min(len(ms) - 1, max(1, int(round(frac * len(ms)))))
    es = sorted(str(m) for m in rng.choice(ms, size=k, replace=False))
    return [m for m in ms if m not in es], es


def match_folds(groups: Sequence[str], cfg: Config, what: str = "labels") -> list[Fold]:
    """GroupKFold over the matches of `groups` (one match id per labelled row), min(cv_folds, matches) folds."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups).astype(str)
    ids = np.unique(groups)
    if len(ids) < 2:
        raise TooFewLabels(f"{what} in {len(ids)} match(es): a split by match needs at least 2")
    n_splits = min(int(cfg.quality.cv_folds), len(ids))
    seed, frac = int(cfg.quality.seed), _es_frac(cfg)
    folds = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)):
        fit, es = split_early_stopping(groups[tr], frac, np.random.default_rng([seed, i]))
        folds.append(Fold(sorted(set(groups[te].tolist())), fit, es))
    return folds


def class_scores(cfg: Config) -> np.ndarray:
    """`quality.class_scores` as an array over class indices 0..4."""
    raw = cfg.quality.class_scores
    raw = raw.to_dict() if isinstance(raw, Config) else dict(raw)
    scores = {int(k): float(v) for k, v in raw.items()}
    if set(scores) != set(CLASSES):
        raise ValueError(f"quality.class_scores must give classes {list(CLASSES)}, got {sorted(scores)}")
    return np.array([scores[c] for c in CLASSES])


def evaluate(y: np.ndarray, proba: np.ndarray, cfg: Config) -> dict:
    """Scores of `proba` (n, 5) against class indices `y` (0..4); JSON-ready.

    Log loss first (the whole distribution); the predicted class (argmax)
    for the rest. Macro-F1 averages the classes in the truth or the
    predictions; balanced accuracy the classes in the truth.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                                 precision_recall_fscore_support)

    y, P = np.asarray(y, int), np.asarray(proba, float)
    if len(y) == 0:
        raise TooFewLabels("nothing to score")
    if P.shape != (len(y), N_CLASSES) or not np.isfinite(P).all():
        raise ValueError(f"probabilities must be finite, shape ({len(y)}, {N_CLASSES}); got {P.shape}")
    labels = list(range(N_CLASSES))
    pred = P.argmax(axis=1)
    eps = np.finfo(float).eps
    nll = -np.log(np.clip(P[np.arange(len(y)), y], eps, 1))
    scores = class_scores(cfg)
    with warnings.catch_warnings():  # constant inputs, classes predicted but absent: handled below
        warnings.simplefilter("ignore")
        prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        rho = spearmanr(P @ scores, scores[y]).statistic if len(y) > 1 else np.nan
        out = {
            "n": int(len(y)),
            "log_loss": float(log_loss(y, P, labels=labels)),
            "balanced_log_loss": float(np.mean([nll[y == c].mean() for c in np.unique(y)])),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "accuracy": float(accuracy_score(y, pred)),
            "spearman": None if not np.isfinite(rho) else float(rho),
            "support": support(y),
            "per_class": {str(c): {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                                   "support": int(sup[i])} for i, c in enumerate(CLASSES)},
            # Rows: true class 1..5; columns: predicted class 1..5.
            "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        }
    return out


@dataclass
class CVResult:
    """`summary`: JSON-ready scores and folds. `oof`: one out-of-fold distribution per labelled shot."""

    summary: dict
    oof: pd.DataFrame


def _masks(ids: np.ndarray, fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.isin(ids, fold.test), np.isin(ids, fold.fit), np.isin(ids, fold.es)


def _cross_validate(cfg: Config, source: str, data: pd.DataFrame) -> CVResult:
    folds = match_folds(data["match_id"], cfg, f"{source} labels")
    X, y = feature_matrix(data, cfg), data["label"].to_numpy(int) - 1
    ids = data["match_id"].to_numpy().astype(str)
    P, prior = np.full((len(y), N_CLASSES), np.nan), np.full((len(y), N_CLASSES), np.nan)
    fold_of = np.full(len(y), -1)
    info = []
    for i, f in enumerate(folds):
        te, fit, es = _masks(ids, f)
        model = fit_model(X[fit], y[fit], cfg, X[es], y[es])
        P[te] = model.predict_proba(X[te])
        prior[te] = class_prior(y[fit | es], proba_floor(cfg))
        fold_of[te] = i
        info.append({"test": f.test, "early_stopping": f.es, "fit": f.fit, "n_fit": int(fit.sum()),
                     "n_early_stopping": int(es.sum()), "n_test": int(te.sum()), "best_iteration": model.best_iteration})
    best = [d["best_iteration"] for d in info]
    summary = {"source": source, "n": int(len(y)), "n_matches": int(len(np.unique(ids))), "n_folds": len(folds),
               "support": support(y), "best_iterations": best,
               "median_best_iteration": max(1, int(round(float(np.median(best))))),
               "metrics": evaluate(y, P, cfg), "prior": evaluate(y, prior, cfg), "folds": info}
    oof = data[["match_id", "rally_id", "shot_index", "frame", "label"]].copy()
    oof["fold"] = fold_of
    for k, c in enumerate(PROBA_COLUMNS):
        oof[c] = P[:, k]
    return CVResult(_jsonable(summary), oof)


def cross_validate(cfg: Config, source: str = "hand", shots: pd.DataFrame | None = None,
                   rallies: pd.DataFrame | None = None,
                   landings: dict[tuple[str, int], Landing] | None = None) -> CVResult:
    """Out-of-fold scores of the `source` model ("hand" or "baseline" labels as the target), grouped by match."""
    data = labelled_shots(cfg, source, shots, rallies, landings)
    return _cross_validate(cfg, source, data)


def _baseline_oof(cfg: Config, hand: pd.DataFrame, base: pd.DataFrame, folds: list[Fold], X_hand: pd.DataFrame,
                  y_hand: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, list[dict]]:
    """Per fold, a model on the baseline labels of the training matches, predicting the held-out hand-labelled shots.

    `(P, P_recalibrated, info)`. `P`: as the model gives it, under its own
    training labels' class shares (mostly class 3), so on hand labels it
    loses on calibration alone. `P_recalibrated`: moved to the fold's
    training hand labels' class shares (`shift_prior`; `y_hand`, class
    indices of `hand`) — what the baseline's features say, under the hand
    prior. Both None when some fold has no baseline label to train on (no
    rallies.csv).
    """
    P = np.full((len(hand), N_CLASSES), np.nan)
    R = np.full((len(hand), N_CLASSES), np.nan)
    Xb, yb = feature_matrix(base, cfg), base["label"].to_numpy(int) - 1
    hand_ids, base_ids = hand["match_id"].to_numpy().astype(str), base["match_id"].to_numpy().astype(str)
    y_hand, floor = np.asarray(y_hand, int), proba_floor(cfg)
    info = []
    for f in folds:
        te = np.isin(hand_ids, f.test)
        _, fit, es = _masks(base_ids, f)
        if not fit.any():  # the fit matches lack rallies.csv: train on the early-stopping ones, unstopped
            fit, es = es, np.zeros_like(es)
        if not fit.any():
            info.append({"n_fit": 0, "n_early_stopping": 0, "best_iteration": None})
            continue
        model = fit_model(Xb[fit], yb[fit], cfg, Xb[es], yb[es])
        P[te] = model.predict_proba(X_hand[te])
        hand_counts = np.bincount(y_hand[np.isin(hand_ids, f.fit + f.es)], minlength=N_CLASSES)
        R[te] = shift_prior(P[te], model.train_counts, hand_counts, floor)
        info.append({"n_fit": int(fit.sum()), "n_early_stopping": int(es.sum()), "best_iteration": model.best_iteration})
    if np.isnan(P).any():
        return None, None, info
    return P, R, info


def compare(cfg: Config, shots: pd.DataFrame | None = None, rallies: pd.DataFrame | None = None,
            landings: dict[tuple[str, int], Landing] | None = None) -> dict:
    """The plan's acceptance check, out of fold by match, all scored on the same held-out hand labels.

    (a) hand: trained on the training folds' hand labels; (b) baseline:
    trained on the baseline labels of every shot in the training folds'
    matches (never a held-out match), as it comes — under the baseline
    labels' class shares; (c) baseline_recalibrated: the same, moved to the
    training folds' hand-label class shares (`shift_prior`); (d) prior: the
    training folds' hand label frequencies.

    `beats_baseline`, the acceptance verdict: the hand model's log loss is
    under all of (b), (c) and (d) (None when the baseline cannot be trained:
    no rallies.csv). (b) alone is not enough: it pays for the baseline
    labels' prior (mostly class 3), so a hand model that only learned the
    hand labels' prior would beat it with features that say nothing; (c)
    and (d) leave only what the features add. Each comparison is reported
    too (`beats_baseline_raw`, `beats_baseline_recalibrated`, `beats_prior`).
    """
    shots = load_shots(cfg) if shots is None else shots
    if rallies is None:
        rallies = load_rallies(cfg, shots["match_id"].unique())
    hand = labelled_shots(cfg, "hand", shots)
    if hand.empty:
        raise TooFewLabels("no hand labels yet: label shots with `bda label`")
    base = labelled_shots(cfg, "baseline", shots, rallies, landings)
    folds = match_folds(hand["match_id"], cfg, "hand labels")
    X, y = feature_matrix(hand, cfg), hand["label"].to_numpy(int) - 1
    ids = hand["match_id"].to_numpy().astype(str)
    P_hand, P_prior = np.full((len(y), N_CLASSES), np.nan), np.full((len(y), N_CLASSES), np.nan)
    info = []
    for f in folds:
        te, fit, es = _masks(ids, f)
        model = fit_model(X[fit], y[fit], cfg, X[es], y[es])
        P_hand[te] = model.predict_proba(X[te])
        P_prior[te] = class_prior(y[fit | es], proba_floor(cfg))
        info.append({"test": f.test, "early_stopping": f.es, "n_test": int(te.sum()), "n_fit_hand": int(fit.sum()),
                     "n_early_stopping_hand": int(es.sum()), "best_iteration_hand": model.best_iteration})
    P_base, P_recal, base_info = _baseline_oof(cfg, hand, base, folds, X, y)
    for d, b in zip(info, base_info):
        d.update({"n_fit_baseline": b["n_fit"], "n_early_stopping_baseline": b["n_early_stopping"],
                  "best_iteration_baseline": b["best_iteration"]})
    hand_m, prior_m = evaluate(y, P_hand, cfg), evaluate(y, P_prior, cfg)
    base_m = None if P_base is None else evaluate(y, P_base, cfg)
    recal_m = None if P_recal is None else evaluate(y, P_recal, cfg)
    beats_prior = hand_m["log_loss"] < prior_m["log_loss"]
    beats_raw = None if base_m is None else hand_m["log_loss"] < base_m["log_loss"]
    beats_recal = None if recal_m is None else hand_m["log_loss"] < recal_m["log_loss"]
    notes = []
    if base_m is None:
        notes.append("baseline model unavailable: some training fold has no baseline labels (no rallies.csv with "
                     "known winners)")
    return _jsonable({
        "n_labels": len(y), "n_matches": len(np.unique(ids)), "n_folds": len(folds), "support": support(y),
        "hand": hand_m, "baseline": base_m, "baseline_recalibrated": recal_m, "prior": prior_m,
        "beats_baseline": None if base_m is None else bool(beats_raw and beats_recal and beats_prior),
        "beats_baseline_raw": beats_raw, "beats_baseline_recalibrated": beats_recal, "beats_prior": beats_prior,
        "baseline_training_labels": len(base), "folds": info, "notes": notes})


# --- Learning curve --------------------------------------------------------------------------


def learning_curve(cfg: Config, source: str = "hand", shots: pd.DataFrame | None = None,
                   rallies: pd.DataFrame | None = None, reference: dict | None = None,
                   out_dir: str | Path | None = None,
                   landings: dict[tuple[str, int], Landing] | None = None) -> dict:
    """Held-out score against the number of training labels; writes a CSV and a PNG.

    Within each outer fold (by match), `learning_curve_repeats` seeded random
    subsamples of each size from the training folds' labels; the size counts
    every label used, early-stopping match included (what N labels buy).
    Sizes over the smallest training fold are dropped and printed, and that
    fold's full size is added as the last point. Per size and repeat the
    out-of-fold predictions are pooled and scored; the curve is the mean and
    sd over repeats. `reference` (`{"name", "log_loss", "macro_f1"}`) is the
    horizontal line; by default the baseline model recalibrated to the
    hand-label prior for hand labels (as in `compare`), the class prior for
    baseline labels.
    """
    _check_source(source)
    shots = load_shots(cfg) if shots is None else shots
    if (source == "baseline" or reference is None) and rallies is None:
        rallies = load_rallies(cfg, shots["match_id"].unique())
    if (source == "baseline" or reference is None) and landings is None:
        landings = load_landings(cfg, shots["match_id"].unique())
    data = labelled_shots(cfg, source, shots, rallies, landings)
    folds = match_folds(data["match_id"], cfg, f"{source} labels")
    X, y = feature_matrix(data, cfg), data["label"].to_numpy(int) - 1
    ids = data["match_id"].to_numpy().astype(str)
    tests = [np.isin(ids, f.test) for f in folds]
    pools = [np.flatnonzero(np.isin(ids, f.fit + f.es)) for f in folds]
    available = min(len(p) for p in pools)
    wanted = sorted({int(s) for s in cfg.quality.learning_curve})
    sizes = [s for s in wanted if s <= available]
    dropped = [s for s in wanted if s > available]
    if not sizes:
        raise TooFewLabels(f"{source} labels: the smallest training fold has {available}, under every "
                           f"learning-curve size {wanted}")
    capped = bool(dropped) and available > sizes[-1]
    if capped:
        sizes.append(available)
    if dropped:
        print(f"quality: learning curve ({source}): the smallest training fold has {available} labels; sizes "
              f"{dropped} dropped" + (f", {available} (all of them) added as the last point" if capped else ""))
    repeats, seed, frac = int(cfg.quality.learning_curve_repeats), int(cfg.quality.seed), _es_frac(cfg)
    if repeats < 1:
        raise ValueError(f"quality.learning_curve_repeats must be at least 1, got {repeats}")
    runs = []
    for size in sizes:
        for r in range(repeats):
            P, best = np.full((len(y), N_CLASSES), np.nan), []
            for i, f in enumerate(folds):
                rng = np.random.default_rng([seed, i, size, r])
                take = np.sort(rng.choice(pools[i], size=size, replace=False))
                _, es_m = split_early_stopping(ids[take], frac, rng)
                in_es = np.isin(ids[take], es_m)
                model = fit_model(X.iloc[take[~in_es]], y[take[~in_es]], cfg, X.iloc[take[in_es]], y[take[in_es]])
                P[tests[i]] = model.predict_proba(X[tests[i]])
                best.append(model.best_iteration)
            m = evaluate(y, P, cfg)
            runs.append({"size": size, "repeat": r, "log_loss": m["log_loss"], "macro_f1": m["macro_f1"],
                         "balanced_accuracy": m["balanced_accuracy"], "best_iterations": best})
    if reference is None:
        if source == "hand":
            base = labelled_shots(cfg, "baseline", shots, rallies, landings)
            _, P_ref, _ = _baseline_oof(cfg, data, base, folds, X, y)
            name = REFERENCE_BASELINE
        else:
            P_ref = np.full((len(y), N_CLASSES), np.nan)
            for f, te in zip(folds, tests):
                P_ref[te] = class_prior(y[np.isin(ids, f.fit + f.es)], proba_floor(cfg))
            name = "class prior"
        if P_ref is not None:
            m = evaluate(y, P_ref, cfg)
            reference = {"name": name, "log_loss": m["log_loss"], "macro_f1": m["macro_f1"]}
    runs_df = pd.DataFrame(runs)
    points = runs_df.groupby("size").agg(
        n_repeats=("repeat", "size"), log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean")).reset_index().fillna({"log_loss_sd": 0.0, "macro_f1_sd": 0.0})
    out = Path(out_dir) if out_dir else Path(cfg.paths.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path, png_path = out / f"quality_{source}_learning_curve.csv", out / f"quality_{source}_learning_curve.png"
    points.to_csv(csv_path, index=False)
    plot_learning_curve(points, reference, png_path,
                        f"Learning curve: {source} labels, {len(folds)} folds by match, {repeats} subsamples a size")
    return _jsonable({"source": source, "n_labels": len(y), "n_folds": len(folds), "available": available,
                      "sizes": sizes, "requested": wanted, "dropped": dropped, "capped": capped, "repeats": repeats,
                      "points": points.to_dict("records"), "runs": runs, "reference": reference,
                      "csv": csv_path, "png": png_path})


def plot_learning_curve(points: pd.DataFrame, reference: dict | None, path: Path, title: str) -> None:
    """Log loss and macro-F1 against training labels (mean +/- sd), the reference as a dashed line."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ink, muted, series, rule, surface = "#0b0b0b", "#52514e", "#2a78d6", "#dcdbd6", "#fcfcfb"
    fig = Figure(figsize=(10, 3.8), dpi=120, facecolor=surface)
    FigureCanvasAgg(fig)  # Agg, without pyplot's global state
    axes = fig.subplots(1, 2)
    for ax, (key, label) in zip(axes, [("log_loss", "Log loss, held-out matches (lower is better)"),
                                       ("macro_f1", "Macro-F1, held-out matches (higher is better)")]):
        ax.set_facecolor(surface)
        ax.errorbar(points["size"], points[f"{key}_mean"], yerr=points[f"{key}_sd"], color=series, lw=2,
                    marker="o", ms=6, capsize=3, elinewidth=1.5)
        if reference and reference.get(key) is not None:
            ax.axhline(reference[key], color=muted, lw=1.5, ls="--")
            ax.annotate(f"{reference['name']} {reference[key]:.3f}", xy=(1, reference[key]),
                        xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                        ha="right", va="bottom", color=muted, fontsize=8)
        ax.set_title(label, color=ink, fontsize=9, loc="left")
        ax.set_xlabel("training labels", color=muted, fontsize=8)
        ax.grid(axis="y", color=rule, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(rule)
        ax.tick_params(colors=muted, labelsize=8)
    fig.suptitle(title, color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)


# --- Derived quantities ----------------------------------------------------------------------


def apply_game_state(verdicts: pd.Series, state: pd.DataFrame | None, cfg: Config) -> pd.Series:
    """The game-state layer over the verdicts. No rules yet: they pass through unchanged."""
    rules = cfg.quality.game_state_rules or []
    if rules:
        raise ValueError("quality.game_state_rules is not empty, but the rule format has not been decided yet; "
                         "leave it [] until it is")
    return verdicts


def derive(proba: np.ndarray | pd.DataFrame, cfg: Config, state: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expected quality, risk and verdict per row of an (n, 5) distribution (columns classes 1..5).

    expected_quality = sum p_k x class_scores[k]; risk = p1 x (p4 + p5).
    Verdict, first match wins: risky (risk >= risky_min, whatever the
    quality), good (quality >= good_min), bad (quality <= bad_max), else
    neutral. Then the game-state layer (`apply_game_state`).
    """
    index = proba.index if isinstance(proba, pd.DataFrame) else None
    P = np.asarray(proba[PROBA_COLUMNS] if isinstance(proba, pd.DataFrame) else proba, float)
    if P.ndim != 2 or P.shape[1] != N_CLASSES:
        raise ValueError(f"need an (n, {N_CLASSES}) distribution, got shape {P.shape}")
    if len(P) and not np.allclose(P.sum(axis=1), 1, atol=1e-6):
        raise ValueError("each row of the distribution must sum to 1")
    v = cfg.quality.verdict
    risky_min, good_min, bad_max = float(v.risky_min), float(v.good_min), float(v.bad_max)
    if not bad_max < good_min:
        raise ValueError(f"quality.verdict.bad_max ({bad_max}) must be under good_min ({good_min})")
    quality = P @ class_scores(cfg)
    risk = P[:, 0] * (P[:, 3] + P[:, 4])
    verdict = np.select([risk >= risky_min, quality >= good_min, quality <= bad_max], ["risky", "good", "bad"],
                        default="neutral")
    verdict = apply_game_state(pd.Series(verdict, index=index), state, cfg)
    return pd.DataFrame({"expected_quality": quality, "risk": risk, "verdict": verdict.to_numpy()}, index=index)


def game_state(shots: pd.DataFrame, rallies: pd.DataFrame) -> pd.DataFrame:
    """Score before each shot's rally from the hitter's side (`STATE_COLUMNS`); NaN when unknown.

    `score_exact` is the rally's (nullable boolean): False when a rally
    earlier in the game wasn't read, so the score may be off; <NA> unknown.
    """
    cols = ["game", "points_0", "points_1", "games_0", "games_1"]
    if "score_exact" not in rallies.columns:
        rallies = rallies.assign(score_exact=pd.NA)
    r = shots[["match_id", "rally_id"]].reset_index(drop=True).merge(
        rallies[["match_id", "rally_id", *cols, "score_exact"]], how="left", on=["match_id", "rally_id"],
        validate="many_to_one")
    h = pd.to_numeric(shots["hitter_id"], errors="coerce").to_numpy(float)

    def side(prefix: str, hitter: bool) -> np.ndarray:
        a0, a1 = r[f"{prefix}_0"].to_numpy(float), r[f"{prefix}_1"].to_numpy(float)
        out = np.full(len(h), np.nan)
        out[h == 0] = (a0 if hitter else a1)[h == 0]
        out[h == 1] = (a1 if hitter else a0)[h == 1]
        return out

    return pd.DataFrame({"game": r["game"].to_numpy(float),
                         "hitter_points": side("points", True), "opponent_points": side("points", False),
                         "hitter_games": side("games", True), "opponent_games": side("games", False),
                         "score_exact": _as_boolean(r["score_exact"]).array},
                        index=shots.index)


# --- Final model and prediction --------------------------------------------------------------


def _model_paths(cfg: Config, source: str) -> tuple[Path, Path]:
    out = Path(cfg.paths.models_dir)
    return out / f"quality_{source}.txt", out / f"quality_{source}.json"


def train_final(cfg: Config, source: str = "hand", matches: list[str] | None = None) -> dict:
    """Fit on every `source` label and save `<models_dir>/quality_<source>.txt` (+ `.json` meta).

    Rounds: the median best iteration of a grouped cross-validation (whose
    scores go in the meta); `quality.model.n_estimators` when the labels are
    in fewer than two matches.
    """
    _check_source(source)
    shots = load_shots(cfg, matches)
    data = labelled_shots(cfg, source, shots)
    if data.empty:
        raise TooFewLabels(f"no {source} labels to train on")
    try:
        cv = _cross_validate(cfg, source, data).summary
        rounds = cv["median_best_iteration"]
        cv_meta = {k: cv[k] for k in ("n_folds", "best_iterations", "median_best_iteration", "metrics")}
        cv_meta["prior_log_loss"] = cv["prior"]["log_loss"]
    except TooFewLabels as exc:
        rounds = int(cfg.quality.model.n_estimators)
        cv_meta = {"status": "too_few", "reason": str(exc)}
        print(f"quality: no cross-validation ({exc}); training {rounds} rounds (quality.model.n_estimators)")
    y = data["label"].to_numpy(int) - 1
    model = fit_model(feature_matrix(data, cfg, far_unmeasured=_far_unmeasured(cfg)), y, cfg, n_estimators=rounds)
    model_file, meta_file = _model_paths(cfg, source)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(model_file))
    meta = _jsonable({
        "source": source, "created_utc": _now(), "model_file": model_file.name, "features": model.features,
        # Inputs blanked for far shots in training: `load_model` / `predict` blank the same.
        "far_unmeasured": model.far_unmeasured,
        "classes": list(CLASSES), "proba_columns": PROBA_COLUMNS, "label_counts": support(y),
        # What `QualityModel.predict_proba` maps balanced outputs back to (`load_model` reads it).
        "train_class_counts": model.train_counts, "class_weight": model.class_weight,
        "proba_floor_at_training": model.proba_floor,
        "n_train": len(y), "matches": sorted(data["match_id"].unique()), "best_iteration": model.best_iteration,
        "params": lgb_params(cfg), "cv": cv_meta, "lightgbm_version": lgb.__version__})
    meta_file.write_text(json.dumps(meta, indent=1))
    print(f"quality: trained the {source} model on {len(y)} labels from {len(meta['matches'])} matches, "
          f"{model.best_iteration} rounds -> {model_file}")
    return meta


def load_model(cfg: Config, source: str = "hand") -> tuple[QualityModel, dict]:
    """The saved `source` model and its meta.

    The prior correction comes from the meta (the class weighting and class
    counts it was trained with), and so do the inputs blanked for far shots
    (`far_unmeasured`: noted when `quality.far_unmeasured` now differs; a
    meta from before it was saved gets today's list, noted); the floor is
    today's `quality.proba_floor`.
    """
    _check_source(source)
    model_file, meta_file = _model_paths(cfg, source)
    if not model_file.exists() or not meta_file.exists():
        raise SystemExit(f"no trained {source} model at {model_file}; train one first")
    meta = json.loads(meta_file.read_text())
    booster = lgb.Booster(model_file=str(model_file))
    if booster.num_model_per_iteration() != N_CLASSES:
        raise ValueError(f"{model_file}: {booster.num_model_per_iteration()} classes, expected {N_CLASSES}")
    counts = meta.get("train_class_counts")
    if counts is None:  # a meta from before the prior correction: its label counts are the training counts
        counts = [meta["label_counts"][str(c)] for c in CLASSES]
    counts = np.asarray(counts, float)
    if counts.shape != (N_CLASSES,):
        raise ValueError(f"{meta_file}: train_class_counts must give {N_CLASSES} classes, got {counts.tolist()}")
    weight = meta.get("class_weight")
    now = _far_unmeasured(cfg)
    far = meta.get("far_unmeasured")
    if far is None:
        print(f"quality: {meta_file.name} does not say which inputs were blanked for far shots (trained before that "
              f"was saved); assuming quality.far_unmeasured {now} — retrain to record it")
        far = now
    elif set(far) != set(now):
        print(f"quality: the {source} model was trained with far_unmeasured {list(far)}, quality.far_unmeasured is "
              f"now {now}; using the model's — retrain to use the new list")
    model = QualityModel(booster, list(meta["features"]), booster.current_iteration(), counts,
                         "balanced" if _balanced(weight) else None, proba_floor(cfg), [str(c) for c in far])
    return model, meta


def predict(cfg: Config, source: str = "hand", matches: list[str] | None = None) -> dict[str, str]:
    """Write `quality.csv` per match (`QUALITY_COLUMNS`) from the saved `source` model; returns {match: path}."""
    model, meta = load_model(cfg, source)
    if list(meta["features"]) != list(cfg.quality.features):
        print("quality: the model was trained on other inputs than quality.features; using the model's — retrain "
              "to use the new list")
    shots = load_shots(cfg, matches, model.features)
    rallies = load_rallies(cfg, shots["match_id"].unique())
    P = model.predict_proba(feature_matrix(shots, cfg, model.features, model.far_unmeasured))
    state = game_state(shots, rallies)
    derived = derive(P, cfg, state)
    out = pd.concat([shots[ID_COLUMNS].reset_index(drop=True), pd.DataFrame(P, columns=PROBA_COLUMNS),
                     derived.reset_index(drop=True), state.reset_index(drop=True)], axis=1)[QUALITY_COLUMNS]
    written = {}
    for m, g in out.groupby("match_id", sort=True):
        path = cache_dir(cfg, str(m)) / "quality.csv"
        g.to_csv(path, index=False)
        written[str(m)] = str(path)
        n = g["verdict"].value_counts()
        print(f"quality: wrote {path} — {len(g)} shots; " + ", ".join(f"{v} {int(n.get(v, 0))}" for v in VERDICTS))
    return written


# --- Leak check ------------------------------------------------------------------------------


def _grouped_auc(X: pd.DataFrame, y: np.ndarray, ids: np.ndarray, folds: list[Fold], cfg: Config) -> float:
    """Pooled out-of-fold ROC AUC of a LightGBM binary model of `y` on `X`, folds by match (as the quality model's)."""
    from sklearn.metrics import roc_auc_score

    m = cfg.quality.model
    params = {k: v for k, v in lgb_params(cfg).items() if k != "num_class"}
    params.update(objective="binary", metric="binary_logloss")
    p = np.full(len(y), np.nan)
    for f in folds:
        te, fit, es = _masks(ids, f)
        train = lgb.Dataset(X[fit].to_numpy(float), label=y[fit], feature_name=list(X.columns))
        kwargs: dict = {}
        if es.any():
            val = lgb.Dataset(X[es].to_numpy(float), label=y[es], reference=train)
            kwargs = {"valid_sets": [val], "valid_names": ["early_stopping"],
                      "callbacks": [lgb.early_stopping(int(m.early_stopping_rounds), first_metric_only=True,
                                                       verbose=False)]}
        booster = lgb.train(params, train, num_boost_round=int(m.n_estimators), **kwargs)
        p[te] = booster.predict(X[te].to_numpy(float), num_iteration=int(booster.best_iteration) or None)
    return float(roc_auc_score(y, p))


def leak_check(shots: pd.DataFrame, cfg: Config) -> dict:
    """Does whether the shot came back (`ended`) show through the model inputs? JSON-ready.

    Grouped-CV AUC (folds by match, `quality.cv_folds`, early stopping on
    held-out matches) of a LightGBM model of `ended` from (i) which of
    `feature_matrix`'s inputs are missing — the check: above
    `quality.max_missing_auc` is a LEAK (when an input exists depends on what
    happened after the shot) — and (ii) all the inputs, for information:
    good shots do end rallies, so some signal is expected. Per input, the AUC
    of its missingness alone (either direction). Rows with an unknown
    `ended` are left out; too little data is a skip with the reason, not an
    error.
    """
    from sklearn.metrics import roc_auc_score

    limit = float(cfg.quality.max_missing_auc)
    res: dict = {"max_missing_auc": limit}

    def skip(reason: str) -> dict:
        return {"status": "skipped", "reason": reason, **res}

    if "ended" not in shots.columns:
        return skip("shots.csv has no `ended` column")
    ended = pd.to_numeric(shots["ended"], errors="coerce")
    keep = ended.notna().to_numpy()
    data = shots[keep]
    y = (ended[keep].to_numpy(float) > 0).astype(int)
    res.update({"n": int(len(y)), "n_matches": int(data["match_id"].nunique()),
                "ended_share": float(y.mean()) if len(y) else None})
    if len(np.unique(y)) < 2:
        return skip(f"`ended` is known for {len(y)} shots and takes {len(np.unique(y))} value(s): nothing to "
                    "tell apart")
    try:
        folds = match_folds(data["match_id"], cfg, "shots with a known `ended`")
    except TooFewLabels as exc:
        return skip(str(exc))
    ids = data["match_id"].to_numpy().astype(str)
    X = feature_matrix(data, cfg)
    missing = X.isna().astype(float)
    by_feature = {}
    for c in X.columns:
        share = float(missing[c].mean())
        if 0 < share < 1:
            a = float(roc_auc_score(y, missing[c].to_numpy()))
            by_feature[c] = {"missing_share": share, "auc": max(a, 1 - a)}
    by_feature = dict(sorted(by_feature.items(), key=lambda kv: -kv[1]["auc"]))
    missing_auc = _grouped_auc(missing.add_suffix("_missing"), y, ids, folds, cfg)
    full_auc = _grouped_auc(X, y, ids, folds, cfg)
    leak = missing_auc > limit
    res.update({"status": "ok", "n_folds": len(folds), "missing_auc": missing_auc, "full_auc": full_auc,
                "leak": leak, "verdict": "LEAK" if leak else "ok", "missing_by_feature": by_feature})
    return _jsonable(res)


# --- Report ----------------------------------------------------------------------------------


def report(cfg: Config, out_dir: str | Path | None = None, matches: list[str] | None = None) -> dict:
    """Everything that can be evaluated now, as `quality_report.json` / `.txt` and learning-curve PNGs.

    First the leak check (`leak_check`); always the baseline-label CV and
    learning curve (when rallies.csv exist); the acceptance check and the
    hand learning curve when there are hand labels in at least two matches —
    otherwise the report says why not. Writes into `out_dir` (default
    `paths.models_dir`).
    """
    out = Path(out_dir) if out_dir else Path(cfg.paths.models_dir)
    out.mkdir(parents=True, exist_ok=True)
    shots = load_shots(cfg, matches)
    feature_matrix(shots, cfg)  # leaky or absent inputs stop the report here, before any training
    ids = sorted(shots["match_id"].unique())
    rallies = load_rallies(cfg, ids)
    landings = load_landings(cfg, ids)
    raw = load_labels(Path(cfg.paths.labels_dir) / "shot_labels.csv")
    raw = raw[raw["match_id"].isin(ids)]
    base = labelled_shots(cfg, "baseline", shots, rallies, landings)
    missed = missed_final_hits(shots, landings, cfg).to_numpy()
    res: dict = {
        "created_utc": _now(), "n_shots": len(shots), "n_matches": len(ids),
        "matches_without_rallies": sorted(set(ids) - set(rallies["match_id"])),
        "hand_label_counts": {k: int((raw["label"] == k).sum()) for k in LABEL_CLASSES},
        "baseline_labels": {"n": len(base), "n_matches": int(base["match_id"].nunique()),
                            "support": support(base["label"].to_numpy(int) - 1),
                            "n_rallies": int(len(shots[["match_id", "rally_id"]].drop_duplicates())),
                            "n_rallies_with_landing": len(landings),
                            "min_missed_gap_s": min_missed_gap_s(cfg),
                            "n_rallies_final_hit_missed": int(len(
                                shots.loc[missed, ["match_id", "rally_id"]].drop_duplicates()))},
        "quality_config": cfg.quality.to_dict()}
    try:
        res["leak_check"] = leak_check(shots, cfg)
    except (TooFewLabels, ValueError, lgb.basic.LightGBMError) as exc:
        res["leak_check"] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}",
                             "max_missing_auc": float(cfg.quality.max_missing_auc)}
    for key, run in (("baseline_cv", lambda: _cross_validate(cfg, "baseline", base).summary),
                     ("baseline_learning_curve", lambda: learning_curve(cfg, "baseline", shots, rallies, out_dir=out,
                                                                        landings=landings))):
        try:
            res[key] = run()
        except TooFewLabels as exc:
            res[key] = {"status": "too_few", "reason": str(exc)}
    try:
        res["compare"] = compare(cfg, shots, rallies, landings)
    except TooFewLabels as exc:
        res["compare"] = {"status": "too_few", "reason": str(exc)}
    cmp = res["compare"]
    if cmp.get("status") == "too_few":
        res["hand_learning_curve"] = {"status": "too_few", "reason": cmp["reason"]}
    else:
        rb = cmp["baseline_recalibrated"]
        ref = None if rb is None else {"name": REFERENCE_BASELINE, "log_loss": rb["log_loss"],
                                       "macro_f1": rb["macro_f1"]}
        try:
            res["hand_learning_curve"] = learning_curve(cfg, "hand", shots, rallies, reference=ref, out_dir=out,
                                                        landings=landings)
        except TooFewLabels as exc:
            res["hand_learning_curve"] = {"status": "too_few", "reason": str(exc)}
    res = _jsonable(res)
    (out / "quality_report.json").write_text(json.dumps(res, indent=1))
    text = summary_text(res)
    (out / "quality_report.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"quality: report -> {out / 'quality_report.json'}, {out / 'quality_report.txt'}")
    return res


def summary_text(res: dict) -> str:
    """The report as a few readable lines."""
    lines = [f"Shot quality, phase 7 — {res['created_utc']}", "",
             f"{res['n_shots']:,} shots in {res['n_matches']} matches."]
    if res["matches_without_rallies"]:
        lines.append(f"No rallies.csv for {len(res['matches_without_rallies'])}: "
                     f"{', '.join(res['matches_without_rallies'])} (no baseline labels or game state there).")
    lines.append("Classes: " + "; ".join(f"{c} {LABEL_CLASSES[str(c)].lower()}" for c in CLASSES) + ".")
    lines += _leak_lines(res.get("leak_check"))
    b = res["baseline_labels"]
    lines += ["", f"Baseline labels (rally outcome credited backwards): {b['n']:,} shots in {b['n_matches']} matches",
              f"  support  {_support_line(b['support'])}"]
    if "n_rallies_final_hit_missed" in b:
        after = f", {b['min_missed_gap_s']:g} s or more after that hit" if b.get("min_missed_gap_s") is not None else ""
        lines.append(f"  final hit missed (landed on the last detected hitter's side, away from the net{after}): "
                     f"{b['n_rallies_final_hit_missed']:,} of {b['n_rallies']:,} rallies "
                     f"({b['n_rallies_with_landing']:,} with a landing point); credit counted from the missing hit")
    cv = res["baseline_cv"]
    if cv.get("status") == "too_few":
        lines.append(f"  cross-validation: too few — {cv['reason']}")
    else:
        lines.append(f"  cross-validation, {cv['n_folds']} folds by match: {_metrics_line(cv['metrics'])}")
        lines.append(f"  class prior: log loss {cv['prior']['log_loss']:.3f}")
        lines += _class_table(cv["metrics"], "  ")
    lines += _curve_lines(res.get("baseline_learning_curve"), "  ")
    counts = res["hand_label_counts"]
    n_hand = sum(v for k, v in counts.items() if k != "x")
    lines += ["", f"Hand labels: {n_hand:,} (and {counts.get('x', 0)} x, not used)"]
    cmp = res["compare"]
    if cmp.get("status") == "too_few":
        lines.append(f"  too few for the acceptance check: {cmp['reason']}")
    else:
        lines.append(f"  class 1 (outright winners): {cmp['support']['1']} labels — the risk score leans on it most")
        lines.append(f"  support  {_support_line(cmp['support'])}")
        lines.append(f"  acceptance check, out of fold by match ({cmp['n_folds']} folds, {cmp['n_labels']} labels):")
        lines.append(f"    hand model             {_metrics_line(cmp['hand'])}")
        if cmp["baseline"] is None:
            lines.append("    baseline model         unavailable (no rallies.csv with known winners)")
        else:
            lines.append(f"    baseline model         {_metrics_line(cmp['baseline'])}")
            lines.append("      (as trained: under the baseline labels' class shares)")
            lines.append(f"    baseline, recalibrated {_metrics_line(cmp['baseline_recalibrated'])}")
            lines.append("      (moved to the training folds' hand-label class shares)")
        lines.append(f"    class prior            log loss {cmp['prior']['log_loss']:.3f}")
        if cmp["beats_baseline"] is None:
            lines.append("  acceptance: not decided — no baseline model")
        elif cmp["beats_baseline"]:
            lines.append("  acceptance: PASS — the hand-labelled model beats the rally-outcome baseline, as trained "
                         "and recalibrated to the hand-label prior, and the hand-label prior (log loss)")
        else:
            lost = [name for key, name in (("beats_baseline_raw", "the baseline as trained"),
                                           ("beats_baseline_recalibrated", "the recalibrated baseline"),
                                           ("beats_prior", "the hand-label prior")) if cmp.get(key) is False]
            lines.append(f"  acceptance: FAIL — the hand-labelled model does not beat {' or '.join(lost)} "
                         "(log loss): the features are wrong, or there are too few labels yet")
        lines += _class_table(cmp["hand"], "  ")
    lines += _curve_lines(res.get("hand_learning_curve"), "  ")
    return "\n".join(lines) + "\n"


def _leak_lines(lc: dict | None) -> list[str]:
    if lc is None:
        return []
    lines = ["", "Leak check: does whether the shot came back (`ended`) show through the inputs?"]
    if lc.get("status") != "ok":
        return lines + [f"  skipped — {lc.get('reason')}"]
    lines.append(f"  grouped CV by match ({lc['n_folds']} folds, {lc['n']:,} shots, {lc['ended_share']:.0%} ended):")
    if lc["leak"]:
        lines.append(f"  LEAK — which inputs are missing predicts `ended`: AUC {lc['missing_auc']:.3f} > "
                     f"{lc['max_missing_auc']:.2f} (quality.max_missing_auc); the model can tell whether the shot came "
                     "back. Every score below is suspect")
    else:
        lines.append(f"  ok — which inputs are missing: AUC {lc['missing_auc']:.3f} "
                     f"(at most {lc['max_missing_auc']:.2f}, quality.max_missing_auc)")
    lines.append(f"  all the inputs: AUC {lc['full_auc']:.3f} (for information: good shots do end rallies)")
    worst = list(lc["missing_by_feature"].items())[:3]
    if worst:
        lines.append("  most telling missing inputs: " + ", ".join(
            f"{c} (missing {d['missing_share']:.0%}, AUC {d['auc']:.2f})" for c, d in worst))
    return lines


def _support_line(sup: dict) -> str:
    return "  ".join(f"{c}: {sup[str(c)]:,}" for c in CLASSES)


def _fmt(x: float | None, spec: str = ".3f") -> str:
    return "n/a" if x is None else format(x, spec)


def _metrics_line(m: dict) -> str:
    return (f"log loss {m['log_loss']:.3f} (balanced {m['balanced_log_loss']:.3f}), macro-F1 {m['macro_f1']:.3f}, "
            f"balanced accuracy {m['balanced_accuracy']:.3f}, accuracy {m['accuracy']:.3f}, "
            f"Spearman {_fmt(m['spearman'])}")


def _class_table(m: dict, indent: str) -> list[str]:
    lines = [f"{indent}class  precision  recall    F1  support"]
    for c in CLASSES:
        p = m["per_class"][str(c)]
        lines.append(f"{indent}{c:>5}  {p['precision']:9.2f}  {p['recall']:6.2f}  {p['f1']:4.2f}  {p['support']:7,}")
    lines.append(f"{indent}confusion, rows true 1-5, columns predicted 1-5:")
    lines += [f"{indent}  {c}: " + " ".join(f"{v:6,}" for v in row) for c, row in zip(CLASSES, m["confusion"])]
    return lines


def _curve_lines(lc: dict | None, indent: str) -> list[str]:
    if lc is None:
        return []
    if lc.get("status") == "too_few":
        return [f"{indent}learning curve: too few — {lc['reason']}"]
    lines = [f"{indent}learning curve ({lc['repeats']} subsamples a size, {lc['n_folds']} folds by match) -> {lc['png']}"]
    if lc["dropped"]:
        lines.append(f"{indent}  sizes {lc['dropped']} dropped: the smallest training fold has {lc['available']} labels")
    for p in lc["points"]:
        lines.append(f"{indent}  {p['size']:>6,} labels: log loss {p['log_loss_mean']:.3f} +/- {p['log_loss_sd']:.3f}, "
                     f"macro-F1 {p['macro_f1_mean']:.3f} +/- {p['macro_f1_sd']:.3f}")
    ref = lc.get("reference")
    if ref:
        lines.append(f"{indent}  reference, {ref['name']}: log loss {ref['log_loss']:.3f}, macro-F1 {ref['macro_f1']:.3f}")
    return lines


# --- Helpers ---------------------------------------------------------------------------------


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_boolean(values: pd.Series) -> pd.Series:
    """A nullable boolean Series from booleans, 0/1 or "True"/"False" text; anything else <NA>."""
    truth = {True: True, False: False, 1: True, 0: False, "true": True, "false": False, "1": True, "0": False}

    def one(v):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
            return pd.NA
        key = v.strip().lower() if isinstance(v, str) else v
        try:
            return truth.get(key, pd.NA)
        except TypeError:  # unhashable
            return pd.NA

    return pd.Series([one(v) for v in values], index=values.index, dtype="boolean")


def _jsonable(obj):
    """Plain JSON types: numpy scalars and arrays unwrapped, paths as strings, NaN/inf as null."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj
