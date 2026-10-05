# Phase 5 — per-shot features

**Decided 2026-10-04:** the set below as proposed; build 3D now (camera from
the net tape, each shot's flight fitted to shuttle physics) for
`net_clearance` and `shuttle_speed`; recovery base = the player's own median
position at the moments their opponent hits.

One row per contact (hit) in `shots.csv`. Positions are court metres from
phase 3/4, **seen from the hitter's end**: `depth` = distance from the net
(0 at the net, 6.70 at the baseline), `lateral` = across the court as the
hitter faces the net (− left, + right). That way a shot from the near and the
far end mean the same thing.

What the pipeline actually measures today: the shuttle as a **2D image point**
(TrackNet, 85% of rally frames), both players' feet on the court and their 17
keypoints (phase 4). The shuttle's **height is not known** — see "3D" below.

## Identity and context

| feature | meaning | how |
|---|---|---|
| `match_id, rally_id, shot_index` | which shot; `shot_index` 1 = serve | contact detection |
| `frame, time_s` | contact frame | contact detection |
| `hitter_id, hitter_side` | player 0/1 and near/far | phase 4 |
| `is_serve` | first shot of the rally | |
| `time_since_prev_contact` | tempo, s (NaN for the serve) | |

## Hitter

| feature | meaning | how |
|---|---|---|
| `hitter_depth, hitter_lateral` | where they hit from | feet at contact |
| `hitter_speed` | m/s at contact | phase 4 `speed` |
| `contact_height` | shuttle relative to the body at contact: 0 = feet, 1 = top of head, >1 overhead | shuttle image y against the hitter's ankle and head keypoints — no 3D needed |
| `hitter_lean` | torso angle from vertical, degrees | shoulder-mid → hip-mid in the image |
| `stance_width` | metres between the ankles | ankles through the homography |
| `hitter_recovery_time` | s from contact until back within 1 m of their base | **base: to decide** (below) |

## Shuttle

| feature | meaning | how |
|---|---|---|
| `landing_depth, landing_lateral` | where it lands (receiver's end) | floor: the shuttle's image point once it stops (it is on the floor, so the homography is exact). Returned: the receiver's feet at their contact (a proxy; the shuttle is ~1 m from them) |
| `ended` | the shot ended the rally (landed, not returned) | no next contact |
| `dist_from_lines` | m from the landing point to the nearest singles boundary, − if out | landing point |
| `flight_time` | s from contact to the next contact or landing | contacts |
| `shot_length` | m from hitter to landing | |
| `avg_speed` | `shot_length / flight_time`, m/s — a 2D stand-in for shuttle speed | |
| `cross_court` | angle of the shot off straight, degrees | hitter → landing |
| `net_clearance` | height over the net, m | **needs 3D** |
| `shuttle_speed` | peak speed after contact | **needs 3D** |

## Opponent (at the moment of contact)

| feature | meaning | how |
|---|---|---|
| `opponent_depth, opponent_lateral` | where the receiver stands | feet |
| `opponent_dist` | m from the receiver to the landing point | |
| `opponent_speed` | m/s | phase 4 |
| `opponent_toward` | speed component towards the landing point, m/s (− = moving away: wrong-footed) | |

## As built (2026-10-04 night)

- Units: metres, seconds, m/s (`shuttle_speed` too — ×3.6 for km/h).
- `landing_*`: `landing_src` says which — `floor` (the shuttle at rest after
  the shot; exact through the homography), `shuttle3d` (where the fitted 3D
  flight met the receiver), `receiver` (their feet, when no fit).
- `shuttle_speed`: the fitted speed off the racket, only when the track saw
  the shuttle at least 3 times in the first 0.2 s (drag takes most of the
  speed by then; extrapolated, it swung 69-205 km/h with the drag setting).
- `net_clearance`: fitted height at the net plane minus the net there
  (1.524 m centre, 1.55 m at the posts), only when the track saw both sides
  of the crossing within 0.3 s. Negative = into the net.
- `contact_height`: (ankles − shuttle) / (ankles − nose) in the image.
- `hitter_lean`: degrees from vertical, hip-mid → shoulder-mid in the image.
- `hitter_recovery_time`: until within 1 m of the player's base, before their
  next hit; NaN if they never get there.
- `time_since_prev_contact`, `flight_time`: real seconds (repeated frames in
  the 30 fps broadcasts not counted).
- Also kept, for checking: `fit_rms_px`, `fit_n_obs` (the 3D fit),
  `contact_reach`, `contact_gap` (the hit detection).

## Early-flight features (phase 7 inputs)

**Why.** Every flight column above is measured up to the shot's *next
event* — the reply or the floor — and how it is measured switches on which
it was: the landing point is the floor point when the rally ended, the
receiver's feet or the 3D fit's end at the receiver when it came back; the
3D fit's end anchor and its over-the-net constraint switch on the same
condition; `flight_time` runs to the next event; `opponent_dist` and
`opponent_toward` use that landing point; `hitter_recovery_time`'s window
differs for the rally's last shot. From those columns LightGBM predicted
whether a shot ended the rally with grouped-CV **AUC 0.99** — the outcome,
not the quality. The labeller sees the clip only up to 10 frames after the
contact (`labeler.after_frames`), never the reply. So the `early_*` columns
measure the same things from what the clip shows, the same way for every
shot, returned or not. The columns above stay as they were (they describe
what actually happened; not for phase 7).

| column | meaning |
|---|---|
| `early_landing_depth` | where the early flight, flown on, comes down: m from the net into the receiver's half, receiver's frame like `landing_depth` — **signed**: negative = it comes down short of the net, on the hitter's own side |
| `early_landing_lateral` | across the court there, receiver's frame (+ = their right) |
| `early_dist_from_lines` | m from that point to the receiver's singles court, − outside (wide, long, or short of the net) |
| `early_flight_time` | s from the contact until it reaches the floor (not until the reply: the reply is unknown) |
| `early_shot_length`, `early_avg_speed`, `early_cross_court` | launch point → that landing; length / flight time; degrees off straight |
| `early_net_clearance` | height over the tape where the predicted path crosses the net plane, m; − = into the net. **NaN when it comes down before the net** (the landing depth is then negative) |
| `early_shuttle_speed` | speed off the racket, m/s — only with 3+ track points in the first 0.2 s, the rule `shuttle_speed` uses |
| `early_opponent_dist`, `early_opponent_toward` | the receiver's feet and velocity at the contact (as `opponent_*`) against the predicted landing |
| `early_fit_rms_px`, `early_fit_n_obs` | checking only: the early fit's pixel error and track points used |

**How** (`flight.fit_early_flight`, `features.early_flight_features`). The
window is the frames the clip shows after the contact: contact+1 …
contact+`features.early_frames` (10), timed with the `FrameClock` (the "30
fps" broadcasts' repeated frames count no time). Fitted under gravity and
drag: the launch position and velocity, and the launch moment (below), to
the start anchor (within reach of the hitter, pulled towards their feet, as
the main fit has it) and the track points in the window. **Everything comes
from the clip's own frames** — contact − 12 … the window's end:

- *The start pixel* is the shuttle where the clip shows it at the contact
  frame, else the frame before (`features.start_pixel`; 93% / 4% of hits);
  seen at neither (3%), there is no pixel term and the feet and the
  fitted launch moment place the hit. Not `contacts.csv`'s pixel: that is
  where the incoming and outgoing curves meet, and the outgoing curve is
  fitted over the whole flight, up to the reply or the landing — so it
  carried the flight after the clip into every `early_*` value (in the
  test below, a 1.4 px shift from a reply 4 frames later moved the drive's
  landing 0.05-0.1 m). Never the frame after the contact either: that is the
  outgoing flight a frame on (80 px from the hit, median). The clip's pixel
  is the shuttle at a frame, up to half a frame of flight from the launch,
  so it counts 0.3 of a track point (`flight.EARLY_ANCHOR_WEIGHT`, would be
  `flight.early_anchor_weight`; the main fit's anchor counts 3) — chosen on
  the refits below.
- *The track is cleaned on those frames alone* (`features.clip_track`,
  from `features.EARLY_CLEAN_BACK` = 12 frames before the contact; would be
  `features.early_clean_back_frames`): `clean_track` judges a point by both
  its neighbours, so cleaned with the whole match a lone point on the
  window's last frame was kept or dropped by the frame after it.

The contact *frame* is still contacts.csv's (found where the two curves
meet too, so rounded with the whole flight's help): it is what the clip is
cut on, so it is what the labeller sees. Tested: two rallies identical up to the window's end, the reply 4
frames apart after it, give identical `early_*` values though their
contacts.csv pixels differ. **Not used:** the next contact's pixel or
time, the landing, the receiver's position as an anchor, the over-the-net
constraint, whether a next event exists, any track after the window. Two
priors, the same for every shot, neither knowing the outcome:

- *Strokes go towards the net* (`flight.FORWARD_SIGMA_MPS`, a soft wall at
  0.5 m/s on going back towards the hitter's own baseline; which way is
  "towards" comes from the hitter's side, near or far — not from the sign
  of their feet's court y, which flipped for 6 net shots whose feet
  projected across the net: kv_ind2023_f_axelsen's at frame 131224 came
  down 8.6 m behind the hitter, and now 4.0 m into the far court (it
  landed at 6.6). Without it a flight coming at the
  camera has a twin going up and away that its first frames can't tell
  apart (a noise-free test case: the far player's clear fits its clip to
  0.15 px either way, landing 13 m apart).
- *The shuttle leaves the racket at the contact, give or take a frame*: the
  launch moment is fitted, with a prior sd of 0.02 s about the contact frame
  (`flight.LAUNCH_SIGMA_S`), and before the first track point. The contact
  frame is the rounded moment the incoming and outgoing flights meet;
  fixing the launch on it left the first track point ~18 px off on a
  quarter of the real shots (fit error 4.0 → 2.8 px median with it), and on
  a synthetic drive launched 0.4 frame late it puts the landing 1.7 m off
  (with the start pixel at weight 3; at the clip pixel's 0.3 the launch
  moment matters less, but is still fitted).

The fit starts from guesses along the last track point's line of sight
(the depth the camera can't see), runs from the best 6 and keeps the best;
the fitted path is then flown on until z = 0. Fewer than 4 track points in
the window (`features.EARLY_MIN_OBS`): no fit. Fit error over 20 px (as the
main fit): not used, counted (`early_unfit` in `shots.meta.json`). The
plan's invariants, applied to the fit as a whole:

- *A broken fit keeps nothing* (only `early_fit_rms_px`, `early_fit_n_obs`):
  launched at 500 km/h or more, or stopped at the solver's 160 m/s cap
  (`flight.V_CAP_MPS`; a step there is cut, not fitted), or reaching its
  landing at an average of 500 km/h or more — `blanked.early_speed`; or on
  the floor at or before the contact frame, with the clip still to come —
  `blanked.early_time`. That holds whatever the track saw in the first
  0.2 s. (Before, a fit over 500 km/h lost only `early_shuttle_speed`, and
  only with 3+ points there: on km_sgp2019_f_ginting 28 of 585 accepted
  fits launched at 500+ km/h, many pinned at the cap, and kept their
  landing, flight time, average speed and net clearance; 9 shots had a
  flight time ≤ 0 and averages reached 1,467 km/h.)
- `early_shuttle_speed` itself still needs 3+ track points in the first
  0.2 s, as `shuttle_speed` does.
- A predicted landing outside the doubles court + 2 m blanks every
  landing-derived column (`blanked.early_position`); the speed and net
  clearance stay.

All three are counted against their own systematic limit, 30% of a match's
shots (`features.EARLY_MAX_BLANKED_FRAC`, would be
`features.max_early_blanked_frac`): on the 32 matches the early values broke
an invariant on 3.6-17% of shots (landings 2-13%, the far player's mostly;
broken fits 1-5%, 832 in all; on the floor too soon: 7), against 0.2-1.5%
for the measured flights — counted into the 15% rule they would fail
matches whose measured columns are fine. The 15% rule counts the measured
flight values only, as it always did.

**The one exception: fast exchanges.** When the next event — the reply or
the landing — comes within the window, the window stops
`labeler.next_event_margin` (2) frames short of it, exactly as the clip
does (`labeler.clip_range`): the reply's own flight must never be fitted as
this shot. Only the next event's *frame* is used, never what it was, so a
reply and a landing at the same frame give the same early flight (tested).
On the 32 matches the next event came within the 10 frames on 770 shots
(2.5%) and within the 12 that the margin needs on 1,724 (5.7%: 1,650 by a
reply, 74 by the landing); 1,484 of those still had the 4 track points for
a fit, 1,227 a landing.

**Measured accuracy** (all 32 matches, 30,266 shots; an early landing on
25,031 = 83%). Truth for depth and lateral: the 1,641 shots that ended on
the floor with both, at the floor point (exact through the homography);
for returned shots, a rough proxy — the receiver's feet at their contact
(the shuttle is ~1 m from them and in the air). Median absolute error, m
(90th percentile); in brackets, before the start pixel came from the clip
(contacts.csv's, weight 3) and the side and broken-fit fixes:

| | near player's shots | far player's shots | all |
|---|---|---|---|
| floor: landing depth | 1.19 (4.5), corr 0.53 [1.29 (4.9), 0.45] | **2.42 (7.9), corr 0.29** [2.65 (8.2), 0.19] | 1.57 (6.3), corr 0.42 [1.70 (6.8), 0.32] — a constant guess: 2.21 |
| floor: landing lateral | 0.28 (0.8), corr 0.96 [0.29] | 0.28 (1.0), corr 0.96 [0.30] | 0.28 (0.9) [0.29] |
| floor: time to the floor, s | 0.11 (0.37), corr 0.88 [0.12, 0.86] | 0.19 (0.63), corr 0.66 [0.19, 0.63] | |
| returned shots extrapolated short of the net | 6% [5%] | 22% [23%] | |
| all shots: `early_shuttle_speed` vs `shuttle_speed`, m/s | 4.8, corr 0.80 [4.4, 0.82] | 10.8, corr 0.62 [10.7, 0.59] | |
| all shots: `early_net_clearance` vs `net_clearance` | 0.16, corr 0.78 [0.77] | 0.23, corr 0.83 [0.83] | |
| all shots: `early_cross_court` vs `cross_court`, ° | 1.7, corr 0.60 [0.60] | 2.5, corr 0.43 [0.45] | |

How each step changed the depth (refits of the 1,933 floor-ended shots and
3,000 random returned ones, without the invariant blanking, so the tails are
longer than above; returned = the early path at the reply's time against
the receiver's feet, where a constant guess scores 1.04 m):

| early fit | floor depth, median (p90) | near / far, 25 fps | near / far, 30 fps | floor lateral | returned depth | rank corr. near / far (returned) |
|---|---|---|---|---|---|---|
| start anchor + track only | 3.71 (13.3) | 1.63 / 7.97 | 2.30 / 7.64 | 0.37 | 4.14 | 0.45 / −0.17 |
| + strokes go towards the net | 2.83 (8.8) | 1.49 / 3.74 | 2.21 / 4.64 | 0.32 | 2.61 | 0.44 / 0.09 |
| + launch moment fitted | 2.00 (9.1) | 1.20 / 3.03 | 1.43 / 3.99 | 0.31 | 1.66 | 0.65 / 0.09 |
| + best of 6 starts | 1.90 (7.9) | 1.20 / 2.56 | 1.43 / 3.38 | 0.30 | 1.58 | 0.65 / 0.27 |

Then the start pixel, which until then was contacts.csv's (built with the
flight after the clip). Refits of the 1,933 floor-ended shots and 1,400
random returned ones (the side and the clip-only cleaning already fixed;
fits over 500 km/h dropped, no court blanking); landing depth, median
absolute error, m (p90), corr; returned = the early path at the reply's
time against the receiver's feet:

| start pixel | floor, near | floor, far | returned, near / far | floor lateral |
|---|---|---|---|---|
| contacts.csv's, weight 3 (leaks) | 1.29 (4.4), 0.52 | 3.01 (9.5), 0.13 | 1.23 / 2.16 | 0.295 |
| refit to the window, where the curves meet at the contact frame, weight 3 | 1.40 (4.9), 0.49 | 3.05 (9.9), 0.15 | 1.30 / 2.26 | 0.290 |
| the clip's (contact frame, else the one before), weight 3 | 1.27 (4.7), 0.50 | 2.98 (10.2), 0.15 | 1.19 / 2.14 | 0.297 |
| the clip's, weight 1 | 1.24 (4.6), 0.52 | 2.87 (9.8), 0.19 | 1.18 / 2.01 | 0.290 |
| **the clip's, weight 0.3 (as built)** | 1.23 (4.6), 0.53 | 2.73 (9.0), 0.25 | 1.22 / 2.01 | 0.285 |
| none (the feet and the launch moment only) | 1.37 (4.6), 0.52 | 2.70 (8.5), 0.27 | 1.36 / 2.12 | 0.286 |

Against contacts.csv's pixel, weight 0.3 is better on the far player's
floor landings by 0.31 m (bootstrap 95%: 0.06-0.49) and their returned
shots by 0.20 m, the near player's within ±0.1 m. Taking the clip's pixel
as a track point at the contact frame instead (no launch anchor) did best
on returned shots (1.11 / 1.73) but worse on the floor (1.35 / 2.73).

Tried on the second row's fit, no gain: a launch-height prior from the
pose (`contact_height` × the match's standing nose height, sd 0.3 m): 2.83
→ 2.82 m; a robust (Cauchy) track loss: within 0.15 m (8 matches). Moving
the start 0.6 m along the court moves the landing ~0.55 m, so the start is
not what is loose. The depth hangs on the drag rate: the terminal velocity
at 6.0 or 7.8 m/s instead of 6.8 moved the far player's median landing by
−2.7 / +5.1 m (8 matches). Even 20 or 40 frames of track (diagnosis only,
8 matches) left the far player's depth 3.1 / 2.5 m off.

**What this means for phase 7.** Across the court (`early_landing_lateral`,
`early_cross_court`), the time to the floor, the net clearance and the
near player's speed are measured well. Depth is rough for the near
player's shots and carries **little information for the far player's**
(corr 0.29 with the floor point; its median error, 2.42 m, is still worse
than a constant guess's 2.21) — a shot coming at the camera: drag and the approach pull the image
speed opposite ways, so a third of a second of track can't tell a fast deep
shot from a slow short one. Everything built on the landing depth inherits
that (`early_landing_depth`, `early_dist_from_lines`, `early_shot_length`,
`early_avg_speed`, `early_opponent_dist`, `early_opponent_toward`); a model
with `hitter_near` can learn to discount them, or they can be dropped for
the far player's shots — your call. Only a second view would really fix it.

Cost: the early fit (seven unknowns, six starts) about doubles the `shots`
step — all 32 matches took ~30 min with 9 in parallel.

**The leak is gone** (grouped 8-fold CV, LightGBM, `ended` from the
columns alone, 29,132 shots): the measured flight columns 0.989; the
`early_*` columns 0.79 (0.80 on shots whose window was not cut: a good shot
ends rallies, so some of this is real, but it can't come from the reply);
which `early_*` columns are missing, alone: 0.58 (phase 7's limit 0.65).
Re-measured after the start pixel, the side and the broken-fit fixes
(2026-10-05): 0.791, 0.801 and 0.577 — unchanged; what the old start
pixel carried from after the clip was too little to show here, but it was
there. Missing values now differ more with the outcome (no early landing on
23% of shots that ended the rally against 16% of returned ones, was
19% / 16%; no net clearance 37% / 22%, was 31% / 20%): more of the shots
that ended the rally have a broken fit (6.6% against 2.4%; they are hit
higher — `contact_height` 1.07 against 0.87 — the smashes) or come down
before the net (18% / 11%), which their clips show; the broken fits used
to keep their landings. Together they still say 0.58. The early columns are checked to be identical for a shot that came
back and the same shot landing, and for two rallies that differ only after
the window (`tests/test_features.py`). The measured columns are unchanged,
value for value, on all 32 matches.

## Decisions for you

1. **The set** — add, drop, rename. The plan's candidates are all here except
   the two that need 3D, plus `contact_height`, `flight_time`, `avg_speed`,
   `cross_court`, `opponent_toward`, which come cheaply.
2. **3D (`net_clearance`, peak `shuttle_speed`).** The single-plane camera
   recovery is degenerate for a camera looking straight down the court
   (focal length 461 vs 3043 px from the two constraints on the same match).
   Real 3D needs the net tape (1.55 m) as a second reference to calibrate,
   then fitting each shot's flight to shuttle physics through the 2D track —
   research-grade (cf. MonoTrack), a few days, accuracy unknown until tried.
3. **Recovery base.** Where a player "recovers to": their own median position
   when the opponent hits (data-driven, per player), or a fixed point (centre
   of the half, ~3.5 m from the net), or drop the feature.

## Contact detection (proposed; the definition is yours if this is ambiguous)

A contact is a frame where (a) the shuttle's image track turns sharply or
starts moving, (b) the shuttle is within racket reach of one player's wrist
(distance ≤ ~0.7 of their body height in the image), and (c) that player's
wrist is moving fast. Within a rally, hitters alternate near/far, contacts are
at least 100 ms apart, and a rally has 1-60 of them — chosen jointly over the
rally so one missed or doubtful frame doesn't break the alternation.
