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
