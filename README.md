# badminton-analysis

Per-shot quality assessment from badminton match video. See
`docs/badminton-analysis-plan.md` for the full build plan.

**Status: phases 1-4 run on 32 broadcasts — rallies within 1.4% of the real
point count overall; every court 13.40 x 6.10 m; both players found in
95-100% of play frames, identity right across every change of ends, the
movement check passing on 31 (World Champs 2023 fails: phase 2 takes breaks
for rallies there). Overlays not yet signed off. Progress notes:
`docs/progress.md`.**

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# TrackNetV3 — frozen dependency, never edited, gitignored
git clone https://github.com/qaz812345/TrackNetV3.git external/TrackNetV3
uv run gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O /tmp/tnv3_ckpts.zip
python -m zipfile -e /tmp/tnv3_ckpts.zip /tmp/tnv3_ckpts
mkdir -p external/TrackNetV3/ckpts
cp /tmp/tnv3_ckpts/ckpts/*.pt external/TrackNetV3/ckpts/

# Phase 4: YOLO26x-pose weights (126 MB), and torchvision from the same CUDA
# index as torch — PyPI's default build can mismatch torch's CUDA ops.
uv pip install torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
mkdir -p external/models
curl -L -o external/models/yolo26x-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt
```

`ultralytics` is AGPL-3.0. Fine for running this locally; it matters only if
the pipeline is ever distributed or served.

**WSL note:** if the repo lives under `/mnt/c`, put the venv on the Linux
filesystem and symlink it, or every `import torch` pays the 9p filesystem tax:

```bash
uv venv --python 3.12 ~/.venvs/badminton-analysis
ln -sfn ~/.venvs/badminton-analysis .venv
```

The PyPI `torch` wheel (2.14, cu130) already includes `sm_120`, so the RTX
5070 needs no special index.

## Use

```bash
bda info  --video data/raw/match.mp4
bda track --video data/raw/match.mp4 --match-id msia_open_f --seconds 60 --overlay
bda overlay --video data/raw/match.mp4 --match-id msia_open_f   # re-render from cache
bda segment --video data/raw/match.mp4 --match-id msia_open_f --sheet
bda segment-eval --match-id msia_open_f --truth data/labels/msia_open_f.play.csv
bda court   --video data/raw/match.mp4 --match-id msia_open_f        # needs `segment`
bda court-click --video data/raw/match.mp4 --match-id msia_open_f    # manual fallback
bda track   --video data/raw/match.mp4 --match-id msia_open_f --play-only   # whole match, play spans only
bda players --video data/raw/match.mp4 --match-id msia_open_f        # needs `court`; ~20 min a match
bda players-overlay --video data/raw/match.mp4 --match-id msia_open_f --start 30000 --seconds 30
bda shots   --match-id msia_open_f                                   # needs `players`
bda outcomes --match msia_open_f                                     # rally winners; needs `shots`
bda label                                                            # phase 6
bda quality report                                                   # phase 7
```

A whole match, in order: `segment`, `court`, `track --play-only`, `segment`
again (rallies come from the trajectory), `players`, `shots`, `outcomes`.

Outputs land in `data/cache/<match_id>/`:

| File | Contents |
|---|---|
| `shuttle.csv` | `frame, x, y, visible, confidence` in source-video pixels |
| `shuttle.meta.json` | video path and frame range that produced it |
| `overlay.mp4` | trajectory drawn onto the video, for eyeballing |
| `line_masks.npz` | per-frame white-pixel masks at 320 px, bit-packed (the only decode pass phase 2 makes; ~6 MB per 10 min) |
| `view_scores.parquet` | per-frame `recall, precision, score` against the learned court-line template |
| `line_template.png` | the learned template: white = court lines, red = overlay pixels (score graphic) excluded |
| `view_segments.csv` | play / not-play spans with their median score, before rally splitting |
| `segments.csv` | `segment_id, start_frame, end_frame, is_play, rally_id` — tiles the range exactly once |
| `segments.meta.json` | range, `segment:` config and `shuttle.csv` stamp that produced `segments.csv` |
| `segments.png` | contact sheet: one thumbnail per span, green = play, red = not |
| `homography.json` | per play span: `image_to_court` 3x3 (source pixels -> court metres), how it was found, fit cost, measured length/width |
| `court_overlay.png` | every court line projected back onto the empty court, corners circled — the phase 3 eyeball check |
| `court_background.png` | median of play-view frames: the empty court the fit runs on |
| `court_lines.png` | line pixels found on that background |
| `poses.parquet` | every tracked person near the court, every play frame: box, ByteTrack id, feet, shirt/shorts colour, `keypoints[17][3]` |
| `poses.meta.json` | play spans, model and `players.detect` settings that produced it; fps; run time |
| `players.parquet` | the two players: `frame, player_id, side, court_x, court_y, speed, keypoints[17][3]`, box, feet |
| `players.meta.json` | found rate per side, changes of ends, rally speed percentiles, speed violations |
| `players_overlay.mp4` | boxes, skeletons, ids, speeds and a top-down court map — the phase 4 eyeball check |
| `contacts.csv` | `rally_id, shot_index, kind (serve/first/hit/landing), frame, player_id, side, x, y` (contact pixel), reach, gap |
| `camera.json` / `camera_check.png` | focal length, per-span pose; net tape, posts and a 1.8 m figure drawn back on the court |
| `repeats.npz` | repeated (duplicate) frames over the rallies, for `FrameClock` |
| `shots.csv` | one row per hit: the features of `docs/features.md` — **the interface to everything downstream** |
| `review/` | phase 5 acceptance sheets and `review.csv` |
| `rallies.csv` | per rally: winning row of the score graphic and winning player id, how it was read, game, score before it (`outcomes.meta.json`, `score_graphic.png`: what was read, and where) |
| `quality.csv` | phase 7, per shot: `p1`..`p5`, `expected_quality`, `risk`, `verdict`, and the score from the hitter's side |
| `data/labels/shot_labels.csv` | phase 6: every label key press (latest per shot wins) — committed |
| `data/models/` | phase 7: `quality_<source>.txt` (LightGBM) + `.json`, `quality_report.{txt,json}`, learning curves |

Every stage caches and skips work when its output exists. `--force`
recomputes. Exceptions to "exists means reuse": `segments.csv` is redone
(from the cached masks, in seconds) whenever the range, the `segment:` config
or `shuttle.csv` changes; and the shuttle cache is keyed by match id alone, so
a different `--start/--seconds` needs a new `--match-id` or `--force`.

## Config

`configs/default.yaml`. Override any key from the command line:

```bash
bda track ... -o device=cpu -o shuttle.batch_size=4 -o shuttle.eval_mode=nonoverlap
```

`device: auto` picks CUDA, then MPS, then CPU. Nothing hardcodes a device:
development runs on Apple Silicon, production on an RTX 5070. TrackNet output
on MPS matches CPU to ~1e-6.

Measured throughput in `weight` mode, 1080p source:

| Device | `shuttle.precision` | steady-state inference | notes |
|---|---|---|---|
| RTX 5070 | `fp32` | ~74 fps | exact |
| RTX 5070 | `fp16` | ~120 fps | heatmaps differ <1e-3; can flip a borderline pixel |
| M-series MPS | `fp32` | ~16 fps | measured before the preprocessing rewrite |

Plus a fixed ~4 s per `track` call for the background median. `nonoverlap` is
roughly 8x faster and less accurate. Frame rate does not depend on source
resolution — every frame is resized to 512x288 once, then stays on the device.
`shuttle.batch_size` 8 is the sweet spot on 12 GB; 32 overflows and crawls.

The wrapper does not use TrackNetV3's `Shuttlecock_Trajectory_Dataset` for
inference: it resized every frame once per sliding window and ran at ~5 fps
on the 5070. `tests/test_shuttle.py` asserts the wrapper's network inputs
match that dataset's output to 1e-6, so this is a faithful reimplementation,
not a change to the model.

## Layout

```
src/config.py    config loading, path resolution, cache directories
src/video.py     decode, frame iteration, clip extraction, overlay rendering
src/cli.py       command line entry point
src/shuttle.py   TrackNetV3 wrapper — the only module that imports external/
src/segment.py   play view by court-line template, rally boundaries from the trajectory
src/court.py     court model, line detection, image -> court homography, manual fallback
src/players.py   YOLO-pose + ByteTrack, the two players, identity across ends, speed check
src/contacts.py  hit moments: shuttle-track breaks + wrist reach, alternation, serve, landing
src/camera.py    full camera from the homography and the net tape
src/flight.py    a shot's 3D flight (gravity + drag) fitted to its image track
src/features.py  shots.csv: the per-shot features of docs/features.md
src/review.py    acceptance sheets: random shots against the video
src/labeler.py   the labelling server: queue, clips (JPEG frames), append-only label store
src/outcomes.py  who won each rally, read off the broadcast's score graphic; the score
src/quality.py   phase 7: LightGBM over the five outcome classes, CV by match, verdicts
tools/labeler/   the keyboard-only labelling page
```

`src/` is a flat module layout (`import shuttle`, not `import src.shuttle`),
so no module here may be named `model`, `dataset`, `test`, `train`, `predict`,
`preprocess`, or `utils` — TrackNetV3 imports those names from its own root.

## Phase 1 acceptance check

Not yet run: it needs a 60-second clip from a BWF match that is **not** in the
Shuttlecock Trajectory Dataset. The dataset's matches are anonymised as
`match1`–`match26` in the repo; the tournament identities are in the
[dataset page](https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw), which has to be
checked before choosing a clip.

```bash
bda track --video data/raw/<clip>.mp4 --match-id <id> --seconds 60 --overlay
open data/cache/<id>/overlay.mp4
```

Watch it. The shuttle should be tracked through most rallies. Note the failure
modes; do not fix them yet.

## Phase 2 — what counts as play

Play view is the broadcast's **main camera only**: the fixed high shot from
behind one baseline. Live play the director shows from another camera is
deliberately dropped — downstream homography and court positions are built
for one view, and a replay must never be counted as a new rally.

The main camera does not move, so its court lines land on the same pixels in
every play-view frame. `bda segment` learns that line layout from the range
itself (pixels white in a large share of frames, minus overlays such as the
score graphic, which are white in nearly every frame) and scores each frame
by how well its white pixels match it. No colours, no cut detection, nothing
per-venue to tune. On both test broadcasts the score is cleanly bimodal —
play view 0.85–1.0, everything else below 0.4, crossfades and wipes the only
frames in between — so `segment.play_score` sits in an empty gap.

The colour heuristic this replaced (court-hue coverage per histogram-cut
segment) called 85% of real footage play against a true ~40%: it missed
crossfade cuts, read blue shirts as blue court, and passed replays from the
overhead and side cameras. See `docs/progress.md`.

## Phase 2 acceptance check

On a 10-minute chunk — a new match id, so the phase 1 trajectory is not
reused:

```bash
bda track   --video data/raw/<clip>.mp4 --match-id <id>_10m --start <first play frame> --seconds 600
bda segment --video data/raw/<clip>.mp4 --match-id <id>_10m --start <first play frame> --seconds 600 --sheet
```

Open `data/cache/<id>_10m/segments.png` (green = play) and
`line_template.png` (should be the court lines and nothing else). The run
prints the share of frames that scored near `play_score`: well under 1% on a
static camera. A large share means the play camera pans or zooms, which this
method does not handle — those frames are dropped, not mislabelled.

Then mark the true play spans by hand in a CSV with `start_frame,end_frame`
rows and run `bda segment-eval`. Precision and recall should both be well
above 90% before phase 3.

Rally boundaries come from the cached `shuttle.csv`: a rally is a run of
consistently visible shuttle longer than `segment.rally_min_s`. Without a
trajectory in the cache, play spans are written whole with `rally_id = -1`.

Known gaps: a replay shot *from the main camera* would match the template and
be classed as play (none seen in either test broadcast; phase 5's
shuttle-speed invariant should catch slow motion). A range that is mostly
intro or interval has no dominant line layout, and `bda segment` stops with an
error rather than guess.

## Phase 3 — court homography

Court coordinates are metres on the floor: origin at the centre of the court
(under the net), x across it (+ to the right as seen from the main camera),
y along it (+ away from the camera). Near baseline y = -6.70, doubles
sidelines x = -/+3.05. The model is `court.py`'s constants, defined once.

`bda court` fits on an empty-court background (median of ~61 play-view
frames; players and shuttle vanish because the camera is static): line
pixels by brightness top-hat, Hough on their skeleton, every pairing of two
cross and two lengthwise candidates with model lines scored by how close
*all* projected model lines land to line pixels, then a least-squares refit
on every line intersection. Each play span is then checked on its own
frames; a span where the camera moved is refitted on its own background, and
one that still fails gets `image_to_court: null` and `needs_manual`.

`bda court-click` is the fallback: it opens the empty-court background (needs
a display — WSLg on Windows 11), you click the four doubles corners
near-left, near-right, far-right, far-left, Enter. The corners are refined
against the line pixels when that passes the checks. Applies to every span
that needs it, or `--segment N` (repeatable).

## Phase 3 acceptance check

```bash
bda court --video data/raw/<clip>.mp4 --match-id <id>_10m
```

Open `data/cache/<id>_10m/court_overlay.png`: every cyan line must sit on a
painted line, the red circles on the doubles corners. The length check is
built in: the court's length and width, measured from the fitted image
lines through the homography, must be 13.40 m and 6.10 m within
`court.length_tolerance_m` (0.2) or the fit raises `CourtCheckError`.

## Phase 4 — players

`bda players` runs two stages. **Detect** (`poses.parquet`, ~36 fps on the
5070, so ~20 min a match): YOLO26x-pose on every play-span frame at 1280 px,
ByteTrack over each play span (reset at every cut), keeping everyone whose
feet land within 7 x 11 m of the court centre. At 640 px the far player
(150-220 px tall at 1080p) was missed in some frames; at 1280 they were found
in every frame sampled, ankles confident in 99%+, so the plan's
crop-and-upscale pass is not needed. 1920 added only crowd.

**Select** (`players.parquet`, seconds, redone every run):

- Feet = midpoint of the ankles when both are confident, else the bottom
  centre of the box; a box cut by the bottom of the frame (near player
  behind the baseline) has no court position.
- A player stands within the doubles court plus 1 m at the sides and 2 m
  at the ends — line judges and the umpire sit further out. Per span and
  half, the track seen there most often is the player. Within 1 m of the
  net, a person's half is their track's usual half (feet by the post read
  a few cm over the line).
- **Hidden feet.** The near player often stands right in front of the far
  one, whose ankles are then guessed on the near player's legs or head —
  2-3 m too close. A far foot on the near player's head-and-body region is
  blanked (`foot_src = occluded`, no court position) unless both ankles
  are still seen at 0.9+ (those were as accurate as feet in the open).
  1.5-8% of far-player rally frames on the first five matches.
- **Identity across ends.** Ends change only in a break (a gap of 40 s+
  between play spans), so the spans between breaks form blocks. Every way
  of changing ends at up to 3 breaks is scored by how well two players'
  shirt and shorts colours, plus one far-end colour shift (the far end is
  lit differently and seen against the boards), explain each block. Player
  0 = near in the first play span. Then any chosen track whose clothes match
  neither player (a court cleaner while the player is at their bag) is
  dropped and the players re-chosen.
- **Speed and the check** (decided 2026-10-04, replacing the plan's 4 m/s
  cap). The `speed` column: positions median-filtered over 5 frames,
  centred difference over 0.2 s. The plan's cap does not separate errors
  from play: elite players run back to front at ~5 m/s, and a far-end jump
  reads 10-15 m/s for 0.2 s (ankles off the floor read as distance). The
  share of rally frames over 4 m/s is reported (4-12%). What is checked
  inside rallies is what one person cannot do — a step over 1.5 m between
  frames (swaps, hidden feet, camera bumps: 2-3 m), or over 6 m/s averaged
  over 1 s. Each spot found is blanked (`foot_src = flagged`, no court
  position, 0.2 s either side) and listed in `players.meta.json`; more than
  5 in a match is systematic and raises `PlayerCheckError`.

Keypoints are stored raw. `players.smooth_keypoints` is a centred
Savitzky-Golay fit (5 frames, quadratic): a wrist's speed peak keeps its
frame and most of its height, where a moving average of the same width
would flatten it — the smoothing bug the plan warns about. Phase 5 reads
through it.

## Phase 4 acceptance check

The movement check is asserted by `bda players` on every match; the changes
of ends should fall on game breaks (`players.meta.json`, `identity`). By eye:

```bash
bda players-overlay --video data/raw/<id>.mp4 --match-id <id> --start <frame> --seconds 30
```

Each player keeps one colour box and id all the way through; the red cross
sits between their feet; the dots on the map move like the players do.

## Phase 5 — hits, 3D flights, `shots.csv`

`bda shots --match-id <id>` runs three stages; the features themselves are
in `docs/features.md`.

**Hits** (`contacts.py` → `contacts.csv`). Each rally's shuttle track is cut
into the fewest smooth pieces (a cubic in time for x and y) that fit it — an
optimal partition, so a break needs real evidence. A break where the shuttle
is within racket reach of a wrist (0.75 body heights) is a candidate hit;
across a gap, the hit is where the neighbouring pieces' curves meet. Hits
alternate near/far (a Viterbi pass over the rally). The serve is the first
hit with both players placed for it — in the service courts, nearly still,
diagonal — and the shuttle seen in the server's hand just before; anything
earlier is the shuttle being tapped back. A flight that ends at rest is the
landing. On hand-labelled rallies (World Champs 2025 rallies 10, 40, 70;
World Champs 2019 rally 20) every hit was found within 1-3 frames.

**Camera** (`camera.py` → `camera.json`, `camera_check.png`). The court
homography leaves the focal length undetermined for a camera looking down
the court, so the net supplies it: the focal length whose projected tape
(1.55 m at the posts, 1.524 m in the middle) lands on white, court lines
erased first. Where several fit, the one where near and far players'
standing head heights agree wins. All 32 cameras: 20-38 m behind the court,
5-13 m up; standing nose height 1.34-1.52 m near, 1.24-1.41 m far.

**Flights and features** (`flight.py`, `features.py` → `shots.csv`). Each
shot's flight is fitted in 3D through the camera — six numbers, launch point
and velocity, under gravity and drag (terminal velocity 6.8 m/s) — to the
shuttle's image track, pinned at the contact pixel near the hitter and at
the next contact near the receiver (or the floor). Speed off the racket is
kept only when the track saw the first 0.2 s (drag takes most of it after
that); net clearance only when it saw both sides of the crossing. The
2025-26 "30 fps" broadcasts repeat every 6th frame (25 fps content): times
are counted in unique frames (`video.FrameClock`). Invariants: shuttle under
500 km/h, positions within the court + 2 m — breaking values are blanked,
more than 15% of a match's shots is an error.

## Phase 5 acceptance check

```bash
bda shots-review --video data/raw/<id>.mp4 --match-id <id> --n 20
```

writes `data/cache/<id>/review/`: per shot, five frames around the detected
contact (the contact outlined in red), the frame where the shot ends with the
landing point drawn on the court, and a top-down map; fill in
`review.csv` (`contact_ok`, `landing_ok`).

## Phase 6 — the labelling tool

```bash
bda label                       # all matches, one random order; then open http://127.0.0.1:8765
bda label --match kv_wc2025_f_shiyuqi --order rally   # one match, in play order
```

A local page (`tools/labeler/index.html`, served by `src/labeler.py` on
127.0.0.1 only), keyboard only:

| key | |
|---|---|
| `1`-`5` | label the shot (the plan's five outcome classes) and move on |
| `x` | not a real shot / can't judge (the hit detector was wrong, the clip is broken) |
| `Space` | replay; `s` or `Shift+Space` replays at ¼ speed |
| `→` `↓` / `←` `↑` | next / previous shot; `n` next unlabelled |
| `Backspace` | remove this shot's label |
| `h` | hide / show the players' boxes (hitter yellow, receiver cyan); `?` help |

Keys go by position, so they work on any keyboard layout (Thai included).

Each clip runs from 1.5 s before the contact to **10 frames after it, never
later**, and stops 2 frames short of the next hit or landing when that comes
sooner (a quick net reply) — the reply is never shown (`labeler.clip_range`
enforces it, a test checks it frame by frame) — and never across a camera
cut. The hitter is
boxed (yellow, magenta from the hit on); a tick on the bar under the picture
marks the contact. Frames are JPEGs played on a canvas: exact cut, instant
replay, slow motion, no codec needed. The next clips are rendered and
loaded while you watch the current one.

Labels go to **`data/labels/shot_labels.csv`** (committed — they're precious):
one row per key press, written and fsynced before the page shows it as
saved; unsaved labels wait in the browser and are retried, so a crash or a
server restart loses nothing. The latest row per shot wins (relabelling is
just labelling again). `labeler.load_labels()` gives the latest label per
shot, joined to `shots.csv` on `(match_id, frame)`. Columns: `time_utc,
match_id, frame, rally_id, shot_index, hitter_id, hitter_side, label,
seconds` (time spent on the shot), `labeler, tool_version`.

## Phase 6 acceptance check

`bda label`, then label 50 shots without the mouse. The top bar shows the
rate and, from the 50th label on, `last 50 in m:ss` — under 10:00 passes.

## Phase 7 — the quality model

```bash
bda outcomes                     # who won each rally, all matches (~40 s a match, reads the video)
bda outcomes --check             # games read vs the real final scores in docs/match_scores.csv
bda quality report               # CV by match, the acceptance check, learning curves -> data/models/
bda quality train   --source hand        # or --source baseline
bda quality predict --source hand        # quality.csv per match
```

**Rally outcomes** (`src/outcomes.py`). The baseline labels need each
rally's winner; phase 5 can't say (its serve detection only sees far-end
serves, and only 70% of final landings are seen). The broadcast's score
graphic can: between two rallies only the winner's score changes. No OCR
and no per-broadcast setup — the score is found by what a score does: two
digit cells one above the other, of which exactly one changes between
rallies while their column is in play (all 32 broadcasts put the graphic
top-left; 2018-19 and 2022-26 templates both read). A rally where the
graphic is covered (a banner, a cutaway) is bridged by comparing the rallies
either side; a game end seen only as a reset goes to whoever that point ends
the game for; the score is replayed rally by rally. Which row is which
player: the winner serves next, and at a rally's start the server stands
nearer the centre line (86-100% of rallies); clear in/out landings vote too.

**The model** (`src/quality.py`): the plan's LightGBM over the five classes
(`lightgbm.train`, `num_class=5`, balanced class weights, outputs mapped back
to the real class frequencies, a small probability floor).

**Inputs** (`quality.features`): what is known at contact, plus the
**early-flight features** — each shot's 3D flight fitted to only what the
labelling clip shows (10 frames after contact) and extrapolated to the floor,
the same for every shot whether it came back or not (`early_*`, phase 5;
`docs/features.md`). The other after-contact columns are measured up to the
reply or the floor and give away whether the shot came back (grouped-CV AUC
0.99 for `ended`): the model refuses them (`LeakageError`). One camera can't
place a far player's shot in depth, so those depth values are left out for
far shots (`quality.far_unmeasured`). `bda quality report` checks that which
inputs are missing doesn't predict `ended` (`quality.max_missing_auc`).

**Evaluation** is grouped by match, never by shot: GroupKFold, with an
early-stopping match held out inside each training fold. Primary score:
multiclass log loss; also macro-F1, balanced accuracy, per-class
precision/recall/support (watch class 1), the confusion matrix. From the
distribution: expected quality (class scores in config), risk = p1 x (p4 + p5),
and a verdict — risky, else good, else bad, else neutral (thresholds in config;
game-state rules go in `quality.game_state_rules`, empty until decided).
`quality.csv` carries the score from the hitter's side, with `score_exact`
false where a rally earlier in that game wasn't read.

**Free baseline labels**: each rally's result credited backwards from its last
shot (discount 0.5): the last shot 1 (its hitter won) or 5 (lost), the one
before 2 or 4, the rest 3. A rally whose winner wasn't read gives none. When
the shuttle came to rest on the last detected hitter's own side (away from
the net), the true final hit was missed: that rally's credit starts one shot
further back.

## Phase 7 acceptance check

Label shots with `bda label` (the learning curve wants 200 / 400 / ... / 1000),
then `bda quality report`: the hand-labelled model must beat the model trained
on the baseline labels, on held-out matches (log loss); the report gives the
learning curve and per-class precision / recall.

## Tests

```bash
uv run pytest
```

The tests build their own synthetic videos: one where a marker's x position
encodes the frame index (catches seek and off-by-one errors), and a fake
broadcast — play view, crowd, play view with a three-frame flash, a second
camera, a score graphic throughout — for the segmenter; and a court rendered
from a known homography with clutter and moving players, whose camera is
bumped mid-match, for the court fit. Phase 4's run a stand-in for the pose
network over a rendered match — two players who change ends, plus an umpire
— through detection, tracking, selection and identity; and feed the
movement check jumps, sprints, swaps and flicker. Phase 5's cut synthetic
shuttle tracks at known hits, play a whole rally (held serve, five hits, a
landing) past stand-in players, recover a focal length from a rendered net,
and fit 3D flights filmed by a known camera. Phase 7's read a synthetic
score graphic (both template styles, over a shifting crowd, with a banner and
a missing-graphic rally) and check every point and game boundary; match rows
to players from serve positions and landings; and train and score the model
on synthetic matches (grouped folds, absent classes, baseline labels, leakage
guard, verdicts, the learning curve). Phase 6's render clips from the
marker video and read the frame index back off every JPEG (start, contact,
last frame, and the early stop before a quick reply), tear the label file
mid-row and keep writing, and drive the server over HTTP: labels, refusals,
path tricks, a kept-alive connection. Tests that need the
TrackNet checkpoint or checkout skip when it is absent.
