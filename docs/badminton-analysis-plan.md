# Badminton Match Analysis — Build Plan

A pipeline that takes BWF broadcast footage and produces per-shot quality
assessments with readable feedback.

Hand this to Claude Code one phase at a time. Do not start a phase until the
previous phase's acceptance check passes.

---

## Context

- **Input for now:** BWF broadcast match video (1080p, 30 or 60fps).
- **Later:** fixed-tripod footage of my own matches. Design for this but don't
  build it yet.
- **Environment:** WSL2 Ubuntu, CUDA, RTX 5070 (12GB). Python.
- **Existing experience:** YOLO-pose, temporal smoothing of noisy keypoints,
  torso-normalised measurements. Reuse those patterns.

## Goal

For each shot in a match, produce:
1. A predicted distribution over rally-outcome classes.
2. A derived good / bad / risky classification.
3. A short natural-language explanation of *why*.

## Non-goals (do not build these)

- Do **not** modify or retrain TrackNet. It is a frozen dependency.
- Do **not** build realtime processing. Everything is offline batch.
- Do **not** build a web app, dashboard, or deployment. CLI + CSV only.
- Do **not** handle doubles. Singles only.
- Do **not** use a neural network for shot quality. Gradient-boosted trees.

---

## Repo structure

```
badminton-analysis/
├── external/
│   └── TrackNetV3/          # cloned, never edited, gitignored
├── src/
│   ├── video.py             # decode, frame iteration, clip extraction
│   ├── segment.py           # rally / camera-cut segmentation
│   ├── court.py             # court detection, homography
│   ├── players.py           # detection, tracking, pose
│   ├── shuttle.py           # TrackNet wrapper
│   ├── contacts.py          # hit-moment detection
│   ├── features.py          # per-shot feature extraction
│   ├── quality.py           # LightGBM model
│   └── feedback.py          # SHAP -> text
├── tools/
│   └── labeler/             # local labeling UI
├── data/
│   ├── raw/                 # video (gitignored)
│   ├── cache/               # per-match intermediate artifacts
│   └── labels/              # my labels (committed, they're precious)
├── tests/
└── configs/
```

**Caching rule:** every stage writes its output to `data/cache/<match_id>/`
and skips work if the output already exists. Video decode and TrackNet
inference are expensive; nothing downstream should ever trigger a re-run
unless explicitly forced with `--force`.

---

## Phase 1 — Scaffolding and TrackNet wrapper

**Goal:** get a shuttle trajectory out of a video file.

- Set up the repo structure, `pyproject.toml`, config loading.
- Clone TrackNetV3 into `external/`. Download the pretrained checkpoints
  (check the README — the weights may be hosted off-repo).
- Write `src/shuttle.py`: a wrapper that takes a video path and a frame range,
  runs their `predict.py` logic, and returns a DataFrame with columns
  `frame, x, y, visible, confidence`.
- Write the trajectory back onto the video as an overlay for visual checking.

**Acceptance check:** run on a 60-second clip from a BWF match that is *not*
in the Shuttlecock Trajectory Dataset (check their match list first). Watch the
overlay. The shuttle should be tracked through most rallies. Note the failure
modes you see — do not try to fix them yet.

---

## Phase 2 — Rally segmentation

**Goal:** find the segments of the broadcast that are actual play.

Broadcast has cuts to replays, crowd, close-ups, and graphics. Everything
downstream must only see play segments.

- Camera-cut detection via frame-to-frame colour histogram difference.
- Classify each segment as play-view or not. Start with a heuristic: the play
  view has high court-line coverage and a consistent green/blue court region.
  Only build a classifier if the heuristic fails.
- Within play segments, detect rally boundaries (serve to point).

**Output:** `data/cache/<match_id>/segments.csv` with
`segment_id, start_frame, end_frame, is_play, rally_id`.

**Acceptance check:** on a 10-minute chunk, manually verify the segment
boundaries against the video. Report precision and recall of play-segment
detection. Should be well above 90% before moving on.

---

## Phase 3 — Court homography

**Goal:** map image pixels to real court coordinates in metres.

- Court model: BWF singles court, 13.4m long × 5.18m wide (6.1m for doubles
  sidelines), net at the midpoint. Define this once as a constant.
- Detect court lines: white-pixel mask, line detection, then fit to the court
  model with RANSAC.
- Solve homography per play segment (the broadcast camera pans and zooms, so
  it cannot be solved once for the whole video).
- Provide a manual fallback: a script that lets me click four corners for a
  segment where auto-detection fails.

**Output:** `data/cache/<match_id>/homography.json`, one 3×3 matrix per segment.

**Acceptance check:** project the four court corners back into the image and
overlay them. They must sit on the actual lines. Then verify that the computed
court length is 13.4m ± 0.2m. Write this as an assertion in the code, not just
a one-time check.

---

## Phase 4 — Players

**Goal:** track both players and get their pose in court coordinates.

- YOLO-pose for detection and keypoints in one pass. Use the largest model
  that fits — this is offline, there is no framerate pressure.
- ByteTrack via `model.track(persist=True)` for identity across frames.
- Assign players to near/far court half. Reject detections outside the court
  polygon (umpire, line judges, crowd).
- Players are small in frame at distance. If keypoint quality is poor, crop
  each player box and upscale before the pose pass.
- Convert foot position (midpoint of ankle keypoints) to court coordinates
  via the homography.
- Temporal smoothing on keypoints — reuse the approach from CPR Trainer, but
  note that the smoothing window there caused a measurement bug, so keep the
  window small and verify against ground truth.

**Output:** `data/cache/<match_id>/players.parquet` with
`frame, player_id, court_x, court_y, keypoints[17][3]`.

**Acceptance check:** a player must never move more than 4 m/s in court
coordinates. Assert this. Violations mean a bad homography or a tracking
identity swap.

---

## Phase 5 — Contact detection and feature extraction

**Goal:** the per-shot feature table. This is the most important phase.

**Contact detection:** combine two signals —
- Shuttle trajectory direction reversal (the shuttle changes direction at a hit).
- Player pose: wrist velocity peak, arm extension.

Neither alone is reliable. Fuse them.

**Features per shot.** These need domain judgment — propose a set, but I will
review and revise before you build on them. Starting candidates:

| Feature | Notes |
|---|---|
| `landing_x`, `landing_y` | court coords where the shuttle next lands or is hit |
| `net_clearance` | height above net at crossing, in metres |
| `shuttle_speed` | peak speed after contact |
| `dist_from_lines` | shortest distance from landing point to a boundary |
| `opponent_dist` | opponent's distance from the landing point at contact |
| `opponent_velocity` | opponent's speed and direction at contact |
| `hitter_court_pos` | where the hitter was when they hit |
| `hitter_recovery_time` | time to return to base position after the shot |
| `hitter_off_balance` | proxy from pose — torso lean, base of support |
| `rally_shot_index` | how deep into the rally |
| `time_since_prev_contact` | tempo |

**Output:** `data/cache/<match_id>/shots.csv`, one row per shot. This file is
the entire interface to everything downstream. Nothing after this phase reads
video.

**Acceptance check:** extract shots from one full match. Manually verify 20
random shots against the video — is the contact frame right, is the landing
point right? Report accuracy. Also assert: shuttle speed < 500 km/h, all court
coordinates within court bounds + 2m margin.

---

## Phase 6 — Labeling tool

**Goal:** let me label rally outcomes fast.

- Local web page. Reads `shots.csv` and the source video.
- Shows a short clip around one shot. Cut the clip at contact + ~10 frames so
  I cannot see the opponent's reply — I must judge the shot, not the outcome.
- Keyboard shortcuts only, no mouse. Keys 1–5 for the outcome classes:
  1. Outright winner or opponent error
  2. Opponent forced into a weak reply
  3. Neutral, rally continues even
  4. Hitter now on defense
  5. Hitter erred (out or into net)
- Space to replay, arrow keys to navigate, auto-advance after a label.
- Saves incrementally to `data/labels/` after every single label. Never lose
  work on a crash.
- Show progress and a running count per class.

**Acceptance check:** I can label 50 shots in under 10 minutes without
touching the mouse.

---

## Phase 7 — Quality model

**Goal:** predict the outcome distribution, then derive good / bad / risky.

- **One model: LightGBM multiclass over the 5 outcome classes.**
  `LGBMClassifier(objective='multiclass', num_class=5)`. Do not fit quantile
  regression — the target is categorical, and `predict_proba` gives the full
  outcome distribution, which is strictly more information than a quantile
  spread.

- **Suggested hyperparameters** for ~1000 labeled rows:
  ```python
  LGBMClassifier(
      objective='multiclass', num_class=5,
      n_estimators=500, learning_rate=0.05,
      num_leaves=15,            # default 31 overfits at this size
      min_child_samples=20,
      class_weight='balanced',  # outright winners are rare
  )
  ```
  Early stopping against the held-out match.

- **Derived quantities**, both computed from `predict_proba`:
  - *Expected quality* — map classes to scores (winner +1.0, forced weak +0.5,
    neutral 0, on defense −0.5, error −1.0) and take the weighted sum. The
    score mapping lives in config, not in code.
  - *Risk* — `p(class 1) × p(class 4 or 5)`. This is high only when both
    extremes carry mass, which is what "risky" actually means. A variance or
    quantile-spread measure would not distinguish genuinely bimodal shots from
    merely uncertain ones.

- **Derived classes:** good = high expected quality, low risk. Bad = low
  expected quality. Risky = high risk regardless of expected quality.
  Thresholds tunable in config, not hardcoded.

- **Watch class 1.** Outright winners may be only ~5% of shots, so 1000 labels
  gives ~50 examples — and the risk score depends on that class most. Report
  per-class support and confusion, not just overall accuracy.

- **Do not output a verdict, output the distribution.** Whether a risky shot is
  the *right* choice depends on the score (down 18–20 you take the gamble; up
  20–18 you don't). Keep the good/bad/risky judgment in a thin layer above the
  model that can factor in game state.
- **Free baseline labels:** propagate rally outcomes backwards with a discount
  factor so the final shot gets most of the credit. Train on these first. The
  hand-labeled model must beat this baseline or the features are wrong.
- **Split by match, never by shot.** Shots within a rally are highly
  correlated; random shot-level splits will inflate the score badly.
- Produce a learning curve: train on 200/400/600/800/1000 labels and plot
  validation score against training size. This tells me when to stop labeling.

**Acceptance check:** the hand-labeled model beats the rally-outcome baseline
on a held-out match. Report the learning curve, and per-class precision/recall.

*Deferred:* if five buckets later prove too coarse (e.g. ranking two shots that
both land in "neutral"), the alternative is pairwise comparison labeling with a
Bradley-Terry model to recover a continuous score. Quantile regression becomes
appropriate at that point, since the target would no longer be categorical.
Not now.

---

## Phase 8 — Feedback

**Goal:** explain why a shot was bad.

- SHAP values per shot from the LightGBM model. `pip install shap`.
- **Templates first.** Map the top contributing feature to a sentence.
  "Net clearance 1.4m — high enough for a comfortable attack." Build 15–20 of
  these covering common cases. Deterministic and correct.
- **Aggregate before narrating.** Per-shot notes across a match is 800 comments
  nobody reads. Group by (shot type × court zone), surface the three worst
  cells with counts.
- **Counterfactuals**, if the model is good enough: hold court position and
  game state fixed, vary the shot-type features, rank the alternatives. Only
  suggest alternatives that (a) appear in training data with reasonable
  density and (b) are physically available from that body position.
- **LLM layer is optional and last.** If templates read badly, pass the feature
  row and SHAP output as JSON to a local model via Ollama. Instruct it to
  restate the given attributions only — no independent badminton reasoning, no
  invented causes. Low temperature, JSON output so fields can be validated.

**Acceptance check:** for 10 shots the model calls bad, I read the explanation
and agree with it after watching the clip.

---

## Invariants to assert throughout

These catch the silent failures — output that looks fine and is wrong.

- Court length from homography = 13.4m ± 0.2m
- Player speed in court coords < 4 m/s (excluding a few frames of jump landing)
- Shuttle speed < 500 km/h
- All court coordinates within bounds + 2m margin
- Contact frames strictly increasing within a rally
- Rally shot count between 1 and 60
- No two contacts within 100ms of each other

Write these as real assertions, enabled by default. Coordinate-system bugs and
frame-index off-by-ones are the expected failure class here.

---

## What I decide, not you

Ask me, don't guess:

- The feature set for shot quality, and what counts as "opponent advantage."
- The definition of a contact moment, if the fusion heuristic is ambiguous.
- Anything about the labeling rubric or the outcome classes.
- Whether to accept a tracking accuracy number as good enough.

## Working notes

- Do not start Phase N+1 before Phase N's acceptance check passes.
- Prefer small, testable modules over an integrated script.
- Log timings per stage. TrackNet runs at roughly 25 FPS and will dominate.
- Commit `data/labels/` to git. Everything else in `data/` is gitignored.
