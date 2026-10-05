import json

import numpy as np
import pandas as pd
import pytest

import quality as q
from config import load_config
from features import SHOT_COLUMNS
from labeler import LABEL_COLUMNS, LabelStore

# Small, quick models: the tests check the plumbing, not LightGBM.
FAST = ["quality.model.n_estimators=40", "quality.model.learning_rate=0.2", "quality.model.min_child_samples=5",
        "quality.model.early_stopping_rounds=10", "quality.model.num_threads=2", "quality.cv_folds=3",
        "quality.learning_curve=[20, 40]", "quality.learning_curve_repeats=2",
        "quality.far_unmeasured=[]"]  # tested on its own below: the synthetic labels lean on early depth


def test_far_shots_have_no_early_depth(tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    shots = pd.DataFrame({c: [1.0, 2.0] for c in cfg.quality.features if c != "hitter_near"})
    shots["hitter_side"] = ["near", "far"]
    X = q.feature_matrix(shots, cfg)
    for c in cfg.quality.features:
        if c in cfg.quality.far_unmeasured:
            assert X[c].iloc[0] == 1.0 and np.isnan(X[c].iloc[1]), c    # near kept, far left unmeasured
        elif c != "hitter_near":
            assert X[c].tolist() == [1.0, 2.0], c


def make_cfg(tmp_path, *extra):
    return load_config(overrides=[f"paths.cache_dir={tmp_path / 'cache'}", f"paths.labels_dir={tmp_path / 'labels'}",
                                  f"paths.models_dir={tmp_path / 'models'}", *FAST, *extra])


# shots.csv as phase 5 writes it, plus any configured input it doesn't have yet (the early_* columns).
COLUMNS = list(dict.fromkeys([*SHOT_COLUMNS, *(c for c in load_config().quality.features if c != "hitter_near")]))


def true_class(shots):
    """The synthetic hand label: two inputs decide it (high early landing depth + net clearance = class 1)."""
    s = shots["early_landing_depth"].to_numpy() + shots["early_net_clearance"].to_numpy()
    return 5 - np.digitize(s, [-1.2, -0.4, 0.4, 1.2])


def write_match(cache, mid, rng, n_rallies=15, rallies=True):
    """A match of random features; rally winners random, so baseline labels say nothing about the features."""
    rows = []
    for r in range(n_rallies):
        n, first = int(rng.integers(3, 8)), int(rng.integers(0, 2))
        for i in range(n):
            hid = (first + i) % 2
            row = {c: float(rng.normal()) for c in COLUMNS}
            row.update({"match_id": mid, "rally_id": r, "shot_index": i + 1, "frame": 1000 * r + 20 * i,
                        "time_s": (1000 * r + 20 * i) / 25, "hitter_id": float(hid),
                        "hitter_side": "near" if hid == 0 else "far", "is_serve": int(i == 0), "ended": int(i == n - 1),
                        "landing_src": "floor" if i == n - 1 else "receiver", "opponent_id": float(1 - hid)})
            rows.append(row)
    shots = pd.DataFrame(rows, columns=COLUMNS)
    out = cache / mid
    out.mkdir(parents=True)
    shots.to_csv(out / "shots.csv", index=False)
    if rallies:
        pd.DataFrame({"rally_id": range(n_rallies), "winner_id": rng.integers(0, 2, n_rallies).astype(float),
                      "how": "score", "game": 1.0, "points_0": np.arange(n_rallies) // 2, "points_1": (np.arange(n_rallies) + 1) // 2,
                      "games_0": 0.0, "games_1": 0.0, "score_exact": True,
                      # outcomes' own columns, which quality ignores
                      "points_r1": 0, "points_r2": 0, "games_r1": 0, "games_r2": 0}
                     ).to_csv(out / "rallies.csv", index=False)
    return shots


def write_labels(labels_dir, shots, n_x=0, labels=None):
    labels_dir.mkdir(parents=True, exist_ok=True)
    lab = shots.assign(label=np.asarray(true_class(shots) if labels is None else labels).astype(str))
    if n_x:
        lab.loc[lab.index[:n_x], "label"] = "x"
    rows = lab.assign(time_utc="2026-10-05T00:00:00+00:00", seconds=1.0, labeler="test", tool_version=1)
    rows[LABEL_COLUMNS].to_csv(labels_dir / "shot_labels.csv", index=False)


@pytest.fixture
def synthetic(tmp_path):
    """Six matches with rallies.csv; 80 % of the shots hand-labelled by `true_class`, 10 more as x."""
    rng = np.random.default_rng(1)
    shots = pd.concat([write_match(tmp_path / "cache", f"m{i}", rng) for i in range(6)], ignore_index=True)
    labelled = shots.sample(frac=0.8, random_state=0)
    write_labels(tmp_path / "labels", labelled, n_x=10)
    return {"cfg": make_cfg(tmp_path), "shots": shots, "n_hand": len(labelled) - 10, "tmp": tmp_path}


# --- Labels ----------------------------------------------------------------------------------


def toy_rallies():
    # (rally, shot_index, hitter); rows deliberately out of order. Player 0 is at the near end.
    shots = [(0, 3, 0), (0, 1, 0), (0, 5, 0), (0, 2, 1), (0, 4, 1),
             (1, 1, 1), (1, 2, 0), (1, 3, 1),
             (2, 1, 0), (2, 2, 1),
             (3, 1, np.nan), (3, 2, 0),
             (4, 1, 0),
             (5, 2, 0), (5, 1, 1)]
    s = pd.DataFrame([{"match_id": "m", "rally_id": r, "shot_index": i, "frame": 100 * r + i, "hitter_id": h,
                       "hitter_side": {0: "near", 1: "far"}.get(h, np.nan)} for r, i, h in shots])
    rallies = pd.DataFrame({"match_id": "m", "rally_id": [0, 1, 2, 3, 5], "winner_id": [0.0, 0.0, np.nan, 1.0, 1.0]})
    return s, rallies


def labels_by_shot(s, lab):
    return {(int(r), int(i)): (None if pd.isna(v) else int(v)) for r, i, v in zip(s["rally_id"], s["shot_index"], lab)}


def test_baseline_labels_credit_backwards(tmp_path):
    s, rallies = toy_rallies()
    got = labels_by_shot(s, q.baseline_labels(s, rallies, make_cfg(tmp_path)))
    # Rally 0, won by 0: the last shot (0's) 1, the one before (1's) 4, earlier 3.
    assert [got[(0, i)] for i in range(1, 6)] == [3, 3, 3, 4, 1]
    # Rally 1, won by 0, last shot by 1: 5; before it 0's: 2.
    assert [got[(1, i)] for i in range(1, 4)] == [3, 2, 5]
    assert got[(2, 1)] is None and got[(2, 2)] is None  # unknown winner
    assert got[(3, 1)] is None and got[(3, 2)] == 5  # unknown hitter; last shot by the loser
    assert got[(4, 1)] is None  # rally missing from rallies.csv
    assert [got[(5, 1)], got[(5, 2)]] == [2, 5]  # won by 1; 0 hit last


def test_baseline_labels_shift_when_the_final_hit_was_missed(tmp_path):
    s, rallies = toy_rallies()
    cfg = make_cfg(tmp_path)
    gap = float(cfg.outcomes.landing_net_gap_m)
    assert q.min_missed_gap_s(cfg) == pytest.approx(0.8)  # default: two flights

    def at(rally, x, y, after_s, fps=25.0):  # landing `after_s` after the rally's last detected hit
        last = s.loc[s["rally_id"] == rally, "frame"].max()
        return q.Landing(x, y, last + round(after_s * fps), fps)

    landings = {("m", 0): at(0, 0.5, -(gap + 2), 1.0),   # last detected hitter 0 (near), near half, 1 s on: shifted
                ("m", 1): at(1, 0.2, gap / 2, 1.6),      # last hitter 1 (far), far half but by the net: not
                # Last hitter 1 (far), deep in the far half, but 0.4 s on: a net error hanging in the net, a metre
                # up, projects deep through the floor homography — no time for a missed reply: not shifted.
                ("m", 2): at(2, 0.0, gap + 4, 0.4),
                ("m", 3): at(3, 0.0, gap + 3, 1.6),      # last hitter 0 (near), far half: the expected end, not shifted
                ("m", 4): (1.0, -(gap + 1)),             # near hitter, near half, but no landing time: no evidence
                ("m", 5): at(5, 1.0, -(gap + 3.5), 0.8)}  # last hitter 0 (near), near half, 0.8 s on: shifted
    missed = q.missed_final_hits(s, landings, cfg)
    assert missed.index.equals(s.index)
    assert set(s.loc[missed, "rally_id"]) == {0, 5}
    assert missed[s["rally_id"].isin([0, 5])].all()
    quick = make_cfg(tmp_path, "quality.baseline.min_missed_gap_s=0.3")
    assert set(s.loc[q.missed_final_hits(s, landings, quick), "rally_id"]) == {0, 2, 5}
    slow = make_cfg(tmp_path, "quality.baseline.min_missed_gap_s=1.2")
    assert set(s.loc[q.missed_final_hits(s, landings, slow), "rally_id"]) == set()
    # A landing before the last detected hit (a pick-up read as a hit) is no evidence either.
    assert not q.missed_final_hits(s, {("m", 0): at(0, 0.5, -(gap + 2), -1.0)}, cfg).any()
    with pytest.raises(ValueError, match="min_missed_gap_s"):
        q.missed_final_hits(s, landings, make_cfg(tmp_path, "quality.baseline.min_missed_gap_s=-1"))
    got = labels_by_shot(s, q.baseline_labels(s, rallies, cfg, landings))
    # Rally 0, won by 0: k counts from the missed hit (1's), so 0's last detected shot is credited +0.5.
    assert [got[(0, i)] for i in range(1, 6)] == [3, 3, 3, 3, 2]
    assert [got[(1, i)] for i in range(1, 4)] == [3, 2, 5]  # by the net: as without landings
    assert got[(3, 2)] == 5
    # Rally 5, won by 1: 0's last detected shot, by the loser, is 4 (1 hit it back, 0 failed), not 5.
    assert [got[(5, 1)], got[(5, 2)]] == [3, 4]
    assert q.baseline_labels(s, rallies, cfg, {}).equals(q.baseline_labels(s, rallies, cfg))


def test_baseline_labels_follow_config(tmp_path):
    s, rallies = toy_rallies()
    cfg = make_cfg(tmp_path, "quality.baseline.discount=0.8", "quality.baseline.cuts=[0.9, 0.5]")
    got = labels_by_shot(s, q.baseline_labels(s, rallies, cfg))
    # credits +1, -0.8, +0.64, -0.512, +0.4096 from the end
    assert [got[(0, i)] for i in range(1, 6)] == [3, 4, 2, 4, 1]
    with pytest.raises(ValueError):
        q.baseline_labels(s, rallies, make_cfg(tmp_path, "quality.baseline.cuts=[0.3, 0.75]"))


def test_hand_labels(tmp_path):
    cfg = make_cfg(tmp_path)
    assert q.hand_labels(cfg).empty  # no label file yet
    store = LabelStore(tmp_path / "labels" / "shot_labels.csv")
    for frame, label in [(10, "3"), (20, "x"), (30, "1"), (10, "5"), (40, "2"), (40, "clear")]:
        store.append({"match_id": "m", "frame": frame, "label": label})
    lab = q.hand_labels(cfg).sort_values("frame")
    assert lab["frame"].tolist() == [10, 30] and lab["label"].tolist() == [5, 1]  # latest wins, x and cleared out
    assert lab["label"].dtype.kind == "i"


# --- Inputs ----------------------------------------------------------------------------------


def test_feature_matrix_has_no_leakage(synthetic):
    cfg, shots = synthetic["cfg"], synthetic["shots"]
    X = q.feature_matrix(shots, cfg)
    assert list(X.columns) == list(cfg.quality.features)
    assert not set(q.LEAKY) & set(X.columns)
    assert (X["hitter_near"] == (shots["hitter_side"] == "near")).all()
    assert all(X[c].dtype == float for c in X.columns)
    # Everything measured up to the reply or the floor (the flight's and the contact's QA too), and the unreliable two.
    assert {"ended", "landing_src", "landing_depth", "landing_lateral", "dist_from_lines", "flight_time", "shot_length",
            "avg_speed", "cross_court", "net_clearance", "shuttle_speed", "opponent_dist", "opponent_toward",
            "fit_rms_px", "fit_n_obs", "contact_reach", "contact_gap",
            "hitter_recovery_time", "is_serve"} <= set(q.LEAKY)
    for leak in q.LEAKY:
        bad = make_cfg(synthetic["tmp"], f"quality.features=[shot_index, {leak}]")
        with pytest.raises(q.LeakageError, match=leak):
            q.feature_matrix(shots, bad)
    # An allow-list, not just a deny-list: every shots.csv column off it is refused, LEAKY or not.
    allowed = [c for c in [*COLUMNS, "hitter_near"] if q.allowed_input(c)]
    assert set(allowed) == set(q.CONTACT_INPUTS) | {c for c in COLUMNS if c.startswith("early_")} - set(q.EARLY_QA)
    assert set(cfg.quality.features) <= set(allowed)
    assert list(q.feature_matrix(shots, cfg, allowed).columns) == allowed
    for col in ["early_fit_rms_px", "early_fit_n_obs", "time_s", "frame", "rally_id", "hitter_id", "opponent_id",
                "match_id", "hitter_side", "a_new_column"]:
        bad = make_cfg(synthetic["tmp"], f"quality.features=[shot_index, {col}]")
        with pytest.raises(q.LeakageError, match=rf"\['{col}'\], not on the allow-list"):
            q.feature_matrix(shots.assign(a_new_column=1.0), bad)
    for col in COLUMNS:
        if not q.allowed_input(col):
            with pytest.raises(q.LeakageError):
                q.feature_matrix(shots, cfg, ["shot_index", col])


def test_missing_features_say_rerun_shots(synthetic):
    cfg, shots = synthetic["cfg"], synthetic["shots"]
    with pytest.raises(q.MissingFeatures, match=r"early_flight_time.*rerun `bda shots`"):
        q.feature_matrix(shots.drop(columns="early_flight_time"), cfg)
    # One match's shots.csv from before the early_* columns: named, not read as a match of NaN.
    f = synthetic["tmp"] / "cache" / "m2" / "shots.csv"
    pd.read_csv(f).drop(columns=["early_flight_time", "early_avg_speed"]).to_csv(f, index=False)
    with pytest.raises(q.MissingFeatures, match=r"1 of 6 match\(es\) \(m2\).*early_flight_time.*rerun `bda shots`"):
        q.load_shots(cfg)
    assert len(q.load_shots(cfg, ["m0", "m1"])) > 0
    assert len(q.load_shots(cfg, features=["shot_index"])) == len(shots)


def leak_shots(tmp_path, n_matches=6):
    rng = np.random.default_rng(7)
    return pd.concat([write_match(tmp_path / "cache", f"m{i}", rng) for i in range(n_matches)], ignore_index=True)


def test_leak_check_flags_missingness_that_gives_ended_away(tmp_path):
    cfg = make_cfg(tmp_path)
    shots = leak_shots(tmp_path)
    shots.loc[shots.index[::7], "early_opponent_dist"] = np.nan  # missing at random: harmless
    shots.loc[shots.index[:5], "ended"] = np.nan  # unknown: left out
    clean = q.leak_check(shots, cfg)
    json.dumps(clean)
    assert clean["status"] == "ok" and clean["n"] == len(shots) - 5 and clean["n_folds"] == 3
    assert clean["leak"] is False and clean["verdict"] == "ok"
    assert clean["missing_auc"] <= cfg.quality.max_missing_auc
    assert set(clean["missing_by_feature"]) == {"early_opponent_dist"}

    leaky = shots.copy()
    leaky.loc[leaky["ended"] == 1, "early_flight_time"] = np.nan  # measured only when the shot came back
    res = q.leak_check(leaky, cfg)
    assert res["leak"] is True and res["verdict"] == "LEAK"
    assert res["missing_auc"] > 0.95 and res["full_auc"] > 0.95
    assert next(iter(res["missing_by_feature"])) == "early_flight_time"

    one = q.leak_check(shots[shots["match_id"] == "m0"], cfg)
    assert one["status"] == "skipped" and "match" in one["reason"]
    assert q.leak_check(shots.drop(columns="ended"), cfg)["status"] == "skipped"
    assert q.leak_check(shots.assign(ended=np.nan), cfg)["status"] == "skipped"


# --- Model -----------------------------------------------------------------------------------


def test_probabilities_always_five_columns(tmp_path):
    cfg = make_cfg(tmp_path)
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = rng.choice([0, 1, 2, 4], size=200)  # class 4 (index 3) never seen
    y_es = np.r_[y[:40], [3, 3]]  # ...though the early-stopping set has it
    model = q.fit_model(X, y, cfg, X.iloc[:42], y_es)
    P = model.predict_proba(X)
    assert P.shape == (200, 5)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[:, 3].max() < 1e-2 and P[:, [0, 1, 2, 4]].min() > 0
    one = q.fit_model(X, np.zeros(200, int), cfg).predict_proba(X.iloc[:3])  # a single class
    assert one.shape == (3, 5) and np.allclose(one.sum(axis=1), 1)
    assert q.fit_model(X, y, cfg).predict_proba(X.iloc[:0]).shape == (0, 5)


def test_absent_class_keeps_the_floor(tmp_path):
    cfg = make_cfg(tmp_path)
    a = float(cfg.quality.proba_floor)
    assert a > 0
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = rng.choice([0, 1, 2, 4], size=200)  # class 4 (index 3) never seen
    P = q.fit_model(X, y, cfg, X.iloc[:40], y[:40]).predict_proba(X)
    assert (P >= a / 5 * (1 - 1e-12)).all() and np.allclose(P[:, 3], a / 5)
    assert np.allclose(P.sum(axis=1), 1)
    # A test shot of the absent class costs ln(5 / a), not ~35 nats.
    assert q.evaluate(np.full(200, 3), P, cfg)["log_loss"] == pytest.approx(np.log(5 / a))
    prior = q.class_prior(y, a)
    assert prior[3] == pytest.approx(a / 5) and prior.sum() == pytest.approx(1)
    assert np.allclose(prior, (1 - a) * np.bincount(y, minlength=5) / len(y) + a / 5)
    unfloored = q.fit_model(X, y, make_cfg(tmp_path, "quality.proba_floor=0"), X.iloc[:40], y[:40]).predict_proba(X)
    assert (unfloored[:, 3] == 0).all()
    with pytest.raises(ValueError, match="proba_floor"):
        q.fit_model(X, y, make_cfg(tmp_path, "quality.proba_floor=1.5"))


def test_balanced_model_follows_the_training_prior(tmp_path):
    """Uninformative inputs: the outputs are the training class shares, so verdicts are not all 'risky'."""
    cfg = make_cfg(tmp_path)
    rng = np.random.default_rng(5)
    shares = [0.06, 0.14, 0.45, 0.25, 0.10]
    X = pd.DataFrame(rng.normal(size=(3000, 4)), columns=list("abcd"))
    y = rng.choice(5, size=3000, p=shares)
    model = q.fit_model(X.iloc[:2400], y[:2400], cfg, X.iloc[2400:], y[2400:])
    assert model.class_weight == "balanced" and model.train_counts.tolist() == np.bincount(y[:2400]).tolist()
    X_new = pd.DataFrame(rng.normal(size=(1000, 4)), columns=list("abcd"))
    P = model.predict_proba(X_new)
    assert np.allclose(P.mean(axis=0), np.bincount(y[:2400]) / 2400, atol=0.03)
    raw = model.raw_proba(X_new)  # balanced training alone: ~0.2 a class, risk ~0.08
    assert np.allclose(raw.mean(axis=0), 0.2, atol=0.05)
    assert (q.derive(raw, cfg)["verdict"] == "risky").mean() > 0.5
    verdicts = q.derive(P, cfg)["verdict"]
    assert (verdicts == "risky").mean() < 0.1
    # Unweighted training needs no correction: its outputs already follow the training prior.
    plain = q.fit_model(X.iloc[:2400], y[:2400], make_cfg(tmp_path, "quality.model.class_weight=null"),
                        X.iloc[2400:], y[2400:])
    assert plain.class_weight is None
    assert np.allclose(plain.predict_proba(X_new).mean(axis=0), np.bincount(y[:2400]) / 2400, atol=0.03)


def test_grouped_cv_never_trains_on_a_held_out_match(synthetic, monkeypatch):
    """Every model fit (and early-stopped) only on matches other than the ones it then predicts."""
    cfg = synthetic["cfg"]
    shots = q.load_shots(cfg)
    match_of = shots["match_id"]
    full = q.feature_matrix(shots, cfg)
    real_fit, calls = q.fit_model, []

    def rows(X):
        if X is None or not len(X):
            return set()
        # the index maps each row back to its shot: check it really is that shot's inputs
        assert np.allclose(X.to_numpy(float), full.loc[X.index, list(X.columns)].to_numpy(float), equal_nan=True)
        return set(match_of.loc[X.index])

    def spy(X, y, cfg, X_es=None, y_es=None, n_estimators=None):
        model = real_fit(X, y, cfg, X_es, y_es, n_estimators)
        call = {"fit": rows(X), "es": rows(X_es), "test": set()}
        predict = model.predict_proba

        def recorded(X_test):
            call["test"] |= rows(X_test)
            return predict(X_test)

        model.predict_proba = recorded
        calls.append(call)
        return model

    monkeypatch.setattr(q, "fit_model", spy)

    def check(fold_tests, per_fold):
        assert len(calls) == per_fold * len(fold_tests)
        for c in calls:
            assert c["fit"] and c["test"]
            assert not c["test"] & (c["fit"] | c["es"])
            assert c["test"] in fold_tests  # predicted one fold's held-out matches...
        for t in fold_tests:  # ...and every fold was predicted, by the expected number of models
            assert sum(c["test"] == t for c in calls) == per_fold
        calls.clear()

    cv = q.cross_validate(cfg, "baseline", shots)
    check([set(f["test"]) for f in cv.summary["folds"]], 1)
    cv = q.cross_validate(cfg, "hand", shots)
    check([set(f["test"]) for f in cv.summary["folds"]], 1)
    res = q.compare(cfg, shots)  # per fold: the hand model and the baseline OOF model
    check([set(f["test"]) for f in res["folds"]], 2)
    lc = q.learning_curve(cfg, "hand", shots, reference={"name": "x", "log_loss": 1.0, "macro_f1": 0.0},
                          out_dir=synthetic["tmp"] / "lc")
    check([set(f.test) for f in q.match_folds(q.labelled_shots(cfg, "hand", shots)["match_id"], cfg)],
          len(lc["sizes"]) * lc["repeats"])


def test_folds_never_share_a_match(synthetic):
    cfg = synthetic["cfg"]
    cv = q.cross_validate(cfg, "baseline")
    folds = cv.summary["folds"]
    assert cv.summary["n_folds"] == 3 and len(folds) == 3
    tests = [set(f["test"]) for f in folds]
    for f in folds:
        test, fit, es = set(f["test"]), set(f["fit"]), set(f["early_stopping"])
        assert es and fit  # >= 1 whole training match held out for early stopping
        assert not test & fit and not test & es and not fit & es
        assert test | fit | es == {f"m{i}" for i in range(6)}
    assert set().union(*tests) == {f"m{i}" for i in range(6)} and sum(map(len, tests)) == 6
    oof = cv.oof
    assert not oof[q.PROBA_COLUMNS].isna().any().any()
    assert np.allclose(oof[q.PROBA_COLUMNS].sum(axis=1), 1)
    for i, f in enumerate(folds):  # each shot scored by the fold that held its match out
        assert set(oof.loc[oof["fold"] == i, "match_id"]) == set(f["test"])
    for f in q.match_folds(np.repeat([f"m{i}" for i in range(6)], 10), cfg):
        assert not set(f.test) & set(f.fit + f.es)
    with pytest.raises(q.TooFewLabels):
        q.match_folds(["m0"] * 10, cfg)


# --- Derived quantities ----------------------------------------------------------------------


def test_derive_quality_risk_and_verdict(tmp_path):
    cfg = make_cfg(tmp_path)
    P = np.array([[0.5, 0.0, 0.0, 0.0, 0.5],   # quality 0, risk 0.25: risky
                  [0.4, 0.4, 0.0, 0.2, 0.0],   # quality 0.5 (good) but risk 0.08: risky wins
                  [0.1, 0.6, 0.3, 0.0, 0.0],   # quality 0.4, no risk: good
                  [0.0, 0.0, 0.0, 0.2, 0.8],   # quality -0.9: bad
                  [0.0, 0.0, 1.0, 0.0, 0.0],   # neutral
                  [0.0, 0.0, 0.4, 0.6, 0.0]])  # quality -0.3: bad (<= bad_max)
    d = q.derive(P, cfg)
    assert np.allclose(d["expected_quality"], [0.0, 0.5, 0.4, -0.9, 0.0, -0.3])
    assert np.allclose(d["risk"], [0.25, 0.08, 0.0, 0.0, 0.0, 0.0])
    assert d["verdict"].tolist() == ["risky", "risky", "good", "bad", "neutral", "bad"]


def test_derive_reads_config(tmp_path):
    P = np.array([[0.4, 0.4, 0.0, 0.2, 0.0], [0.1, 0.6, 0.3, 0.0, 0.0]])
    cfg = make_cfg(tmp_path, "quality.verdict.risky_min=0.1", "quality.verdict.good_min=0.45")
    assert q.derive(P, cfg)["verdict"].tolist() == ["good", "neutral"]
    cfg = make_cfg(tmp_path, "quality.class_scores={1: 2.0, 2: 1.0, 3: 0.0, 4: -1.0, 5: -2.0}")
    assert np.allclose(q.derive(P, cfg)["expected_quality"], [1.0, 0.8])
    with pytest.raises(ValueError, match="not been decided"):
        q.derive(P, make_cfg(tmp_path, "quality.game_state_rules=[{when: behind}]"))
    with pytest.raises(ValueError):
        q.derive(np.full((2, 5), 0.3), cfg)  # rows not summing to 1


# --- Evaluation ------------------------------------------------------------------------------


def test_compare_hand_beats_uninformative_baseline(synthetic):
    res = q.compare(synthetic["cfg"])
    json.dumps(res)  # JSON-ready
    assert res["n_labels"] == synthetic["n_hand"] and res["n_matches"] == 6 and res["n_folds"] == 3
    assert res["beats_baseline"] is True and res["beats_prior"] is True
    assert res["beats_baseline_raw"] is True and res["beats_baseline_recalibrated"] is True
    assert res["hand"]["log_loss"] < min(res["baseline"]["log_loss"], res["baseline_recalibrated"]["log_loss"])
    for m in (res["hand"], res["baseline"], res["baseline_recalibrated"], res["prior"]):
        assert m["n"] == res["n_labels"]
        assert sum(m["support"].values()) == res["n_labels"]
        assert set(m["per_class"]) == {"1", "2", "3", "4", "5"}
        assert all(set(p) == {"precision", "recall", "f1", "support"} for p in m["per_class"].values())
        assert np.array(m["confusion"]).shape == (5, 5) and np.array(m["confusion"]).sum() == res["n_labels"]
    assert res["hand"]["spearman"] > 0.5
    for f in res["folds"]:
        # The baseline trains on every shot of the training matches, not just the hand-labelled ones.
        assert f["n_fit_baseline"] > f["n_fit_hand"]
        assert not set(f["test"]) & set(f["early_stopping"])


def test_shift_prior(tmp_path):
    a = 0.01
    base, hand = np.array([10, 15, 50, 15, 10.0]), np.array([30, 30, 5, 20, 15.0])
    # A model that only knows its training prior, moved to another prior, says that prior.
    P = np.tile(q.floor_proba(base / base.sum(), a), (3, 1))
    assert np.allclose(q.shift_prior(P, base, hand, a), q.floor_proba(hand / hand.sum(), a))
    # Evidence is kept: the likelihood ratio between classes changes only by the prior ratio.
    rng = np.random.default_rng(0)
    P = q.floor_proba(rng.dirichlet(np.ones(5), size=50), a)
    Q = q.shift_prior(P, base, hand, a)
    assert np.allclose(Q.sum(axis=1), 1) and (Q >= a / 5 * (1 - 1e-12)).all()
    unfloor = lambda M: (M - a / 5) / (1 - a)  # noqa: E731
    ratio = unfloor(Q) / unfloor(P) * (base / base.sum()) / (hand / hand.sum())
    assert np.allclose(ratio / ratio[:, :1], 1)
    assert np.allclose(q.shift_prior(Q, hand, base, a), P)  # and back
    # A class absent from the target (or the source) gets just the floor.
    Q0 = q.shift_prior(P, base, [30, 30, 0, 20, 15], a)
    assert np.allclose(Q0[:, 2], a / 5)
    with pytest.raises(ValueError):
        q.shift_prior(P, base, [0, 0, 0, 0, 0], a)


def label_by(shots, kind, rng):
    """Hand labels with the same skewed prior (class 3 rare), unlike the baseline labels' (mostly 3).

    "signal": by early_landing_depth + early_net_clearance (high: class 1);
    "noise": drawn at random, whatever the inputs.
    """
    shares = [0.30, 0.30, 0.05, 0.20, 0.15]
    if kind == "noise":
        return rng.choice(np.arange(1, 6), size=len(shots), p=shares)
    s = (shots["early_landing_depth"] + shots["early_net_clearance"]).to_numpy()
    cuts = np.quantile(s, np.cumsum(shares[::-1])[:-1])  # bottom 15 % class 5, ... top 30 % class 1
    return 5 - np.digitize(s, cuts)


@pytest.mark.parametrize("kind", ["noise", "signal"])
def test_acceptance_needs_feature_signal(tmp_path, kind):
    """Labels that only match the hand prior better than the baseline's must not pass; real signal must."""
    rng = np.random.default_rng(11)
    shots = pd.concat([write_match(tmp_path / "cache", f"m{i}", rng) for i in range(6)], ignore_index=True)
    labelled = shots.sample(frac=0.8, random_state=0)
    write_labels(tmp_path / "labels", labelled, labels=label_by(labelled, kind, rng))
    res = q.compare(make_cfg(tmp_path))
    json.dumps(res)
    ll = {k: res[k]["log_loss"] for k in ("hand", "baseline", "baseline_recalibrated", "prior")}
    # The baseline as trained pays for its prior (mostly class 3) on these labels; recalibrated, it doesn't.
    assert ll["baseline_recalibrated"] < ll["baseline"] - 0.3
    assert res["beats_baseline_raw"] is True  # the old check: passes either way
    if kind == "noise":
        assert res["beats_baseline"] is False, ll
        assert res["beats_prior"] is False, ll
    else:
        assert res["beats_baseline"] is True, ll
        assert res["beats_baseline_recalibrated"] is True and res["beats_prior"] is True, ll
    q.report(make_cfg(tmp_path))
    text = (tmp_path / "models" / "quality_report.txt").read_text(encoding="utf-8")
    assert ("acceptance: PASS" in text) == (kind == "signal") and "baseline, recalibrated" in text


def test_compare_needs_hand_labels(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        write_match(tmp_path / "cache", f"m{i}", rng)
    with pytest.raises(q.TooFewLabels, match="no hand labels"):
        q.compare(make_cfg(tmp_path))


def test_learning_curve_caps_sizes(synthetic, capsys):
    cfg = make_cfg(synthetic["tmp"], "quality.learning_curve=[20, 40, 5000]")
    lc = q.learning_curve(cfg, "hand")
    assert lc["dropped"] == [5000] and lc["capped"]
    assert lc["sizes"] == [20, 40, lc["available"]] and lc["available"] < len(synthetic["shots"])
    assert "dropped" in capsys.readouterr().out
    assert [p["size"] for p in lc["points"]] == lc["sizes"] and all(p["n_repeats"] == 2 for p in lc["points"])
    assert lc["reference"]["name"] == q.REFERENCE_BASELINE  # the acceptance check's: recalibrated
    curve = pd.read_csv(lc["csv"])
    assert curve["size"].tolist() == lc["sizes"]
    with open(lc["png"], "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(q.TooFewLabels):
        q.learning_curve(make_cfg(synthetic["tmp"], "quality.learning_curve=[5000]"), "hand")


# --- Final model and prediction --------------------------------------------------------------


def test_train_and_predict_with_game_state(tmp_path):
    rng = np.random.default_rng(3)
    cache = tmp_path / "cache"
    m0 = write_match(cache, "m0", rng, n_rallies=6)
    write_match(cache, "m1", rng, n_rallies=4, rallies=False)
    # Rally 0 of m0: game 2, player 0 on 3 points and 1 game, player 1 on 5 and 0.
    r = pd.read_csv(cache / "m0" / "rallies.csv")
    r.loc[0, ["game", "points_0", "points_1", "games_0", "games_1"]] = [2, 3, 5, 1, 0]
    r["score_exact"] = r["rally_id"] != 0  # rally 0: an earlier rally of the game unread
    r.to_csv(cache / "m0" / "rallies.csv", index=False)
    m0.loc[m0.index[1], "hitter_id"] = np.nan  # rally 0's second shot: hitter unknown
    m0.to_csv(cache / "m0" / "shots.csv", index=False)
    cfg = make_cfg(tmp_path)

    meta = q.train_final(cfg, "baseline")  # labels in one match: no CV, n_estimators rounds
    assert meta["cv"]["status"] == "too_few" and 0 < meta["best_iteration"] <= 40
    assert (tmp_path / "models" / "quality_baseline.txt").exists()
    saved = json.loads((tmp_path / "models" / "quality_baseline.json").read_text())
    assert saved["features"] == list(cfg.quality.features)
    assert saved["class_weight"] == "balanced" and saved["train_class_counts"] == list(saved["label_counts"].values())
    assert sum(saved["train_class_counts"]) == saved["n_train"]
    model, _ = q.load_model(cfg, "baseline")
    assert model.booster.num_model_per_iteration() == 5  # what SHAP will read
    # The prior correction and the floor come back with the model.
    assert model.class_weight == "balanced" and model.train_counts.tolist() == saved["train_class_counts"]
    assert model.proba_floor == cfg.quality.proba_floor
    X = q.feature_matrix(q.load_shots(cfg, ["m0"]), cfg)
    raw = model.raw_proba(X) * np.asarray(saved["train_class_counts"], float)
    expected = q.floor_proba(raw / raw.sum(axis=1, keepdims=True), cfg.quality.proba_floor)
    assert np.allclose(model.predict_proba(X), expected)

    written = q.predict(cfg, "baseline")
    assert set(written) == {"m0", "m1"}
    out = pd.read_csv(cache / "m0" / "quality.csv")
    assert list(out.columns) == q.QUALITY_COLUMNS and len(out) == len(m0)
    assert np.allclose(out[q.PROBA_COLUMNS].sum(axis=1), 1)
    assert set(out["verdict"]) <= set(q.VERDICTS)
    rally0 = out[out["rally_id"] == 0].set_index("shot_index")
    assert (rally0["score_exact"] == False).all()  # noqa: E712 (read back from csv)
    assert (out.loc[out["rally_id"] != 0, "score_exact"] == True).all()  # noqa: E712
    for idx, row in rally0.iterrows():
        assert row["game"] == 2
        if pd.isna(row["hitter_id"]):
            assert row[["hitter_points", "opponent_points", "hitter_games", "opponent_games"]].isna().all()
        elif row["hitter_id"] == 0:
            assert row[["hitter_points", "opponent_points", "hitter_games", "opponent_games"]].tolist() == [3, 5, 1, 0]
        else:
            assert row[["hitter_points", "opponent_points", "hitter_games", "opponent_games"]].tolist() == [5, 3, 0, 1]
    assert rally0["hitter_id"].isna().sum() == 1 and rally0["hitter_id"].notna().sum() >= 2
    no_rallies = pd.read_csv(cache / "m1" / "quality.csv")
    assert no_rallies[q.STATE_COLUMNS].isna().all().all()


def test_saved_model_keeps_its_far_unmeasured(tmp_path, capsys):
    """The inputs blanked for far shots in training are saved, and prediction blanks the same, whatever the config."""
    rng = np.random.default_rng(3)
    write_match(tmp_path / "cache", "m0", rng, n_rallies=12)
    far = ["early_landing_depth", "early_net_clearance"]
    trained = make_cfg(tmp_path, f"quality.far_unmeasured=[{', '.join(far)}]")
    meta = q.train_final(trained, "baseline")
    assert meta["far_unmeasured"] == far
    assert json.loads((tmp_path / "models" / "quality_baseline.json").read_text())["far_unmeasured"] == far
    model, _ = q.load_model(trained, "baseline")
    assert model.far_unmeasured == far
    assert "far_unmeasured" not in capsys.readouterr().out  # same list: nothing to say

    now = make_cfg(tmp_path)  # quality.far_unmeasured: [] since
    model, _ = q.load_model(now, "baseline")
    assert model.far_unmeasured == far
    assert "using the model's" in capsys.readouterr().out
    q.predict(now, "baseline")
    out = pd.read_csv(tmp_path / "cache" / "m0" / "quality.csv")
    shots = q.load_shots(now)
    X_trained = q.feature_matrix(shots, now, model.features, far_unmeasured=far)
    assert X_trained.loc[shots["hitter_side"] == "far", far].isna().all().all()
    assert np.allclose(out[q.PROBA_COLUMNS].to_numpy(), model.predict_proba(X_trained))
    assert np.allclose(X_trained.to_numpy(), q.feature_matrix(shots, trained).to_numpy(), equal_nan=True)
    # ...which matters: built with today's (empty) list, the far shots' predictions would differ.
    assert not np.allclose(out[q.PROBA_COLUMNS].to_numpy(), model.predict_proba(q.feature_matrix(shots, now)))

    # A meta from before the list was saved: today's list, noted.
    f = tmp_path / "models" / "quality_baseline.json"
    f.write_text(json.dumps({k: v for k, v in json.loads(f.read_text()).items() if k != "far_unmeasured"}))
    model, _ = q.load_model(trained, "baseline")
    assert model.far_unmeasured == far and "assuming quality.far_unmeasured" in capsys.readouterr().out


def test_report_without_hand_labels(tmp_path, capsys):
    rng = np.random.default_rng(4)
    for i in range(4):
        write_match(tmp_path / "cache", f"m{i}", rng)
    cfg = make_cfg(tmp_path)
    res = q.report(cfg)
    models = tmp_path / "models"
    saved = json.loads((models / "quality_report.json").read_text())
    assert saved["n_matches"] == 4
    assert res["baseline_cv"]["n_folds"] == 3
    assert res["compare"]["status"] == "too_few" and res["hand_learning_curve"]["status"] == "too_few"
    assert (models / "quality_baseline_learning_curve.png").exists()
    # The leak check: no input ever missing here, so nothing to give away.
    lc = saved["leak_check"]
    assert lc["status"] == "ok" and lc["leak"] is False and lc["verdict"] == "ok"
    assert lc["missing_auc"] is not None and lc["full_auc"] is not None
    # No contacts.csv / shuttle.csv here: no landing points, nothing shifted (noted).
    assert saved["baseline_labels"]["n_rallies_final_hit_missed"] == 0
    assert "no landing points for 4 of 4 matches" in capsys.readouterr().out
    text = (models / "quality_report.txt").read_text(encoding="utf-8")
    assert "too few for the acceptance check" in text and "no hand labels" in text
    assert "Leak check" in text and "ok — which inputs are missing: AUC" in text


def test_report_survives_a_leak_check_it_cannot_run(tmp_path):
    rng = np.random.default_rng(4)
    for i in range(3):
        write_match(tmp_path / "cache", f"m{i}", rng)
    for i in range(3):  # `ended` unknown everywhere
        f = tmp_path / "cache" / f"m{i}" / "shots.csv"
        pd.read_csv(f).assign(ended=np.nan).to_csv(f, index=False)
    res = q.report(make_cfg(tmp_path))
    assert res["leak_check"]["status"] == "skipped" and res["leak_check"]["reason"]
    assert "skipped" in (tmp_path / "models" / "quality_report.txt").read_text(encoding="utf-8")
