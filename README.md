# badminton-analysis

Per-shot quality assessment from badminton match video. See
`docs/badminton-analysis-plan.md` for the full build plan.

**Status: phases 1-3 run on 32 broadcasts (rallies within 1.4% of the real
point count overall; every court 13.40 x 6.10 m); overlays not yet signed
off. Phase 4 (players) built; running on the 32. Progress notes:
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
```

A whole match, in order: `segment`, `court`, `track --play-only`, `segment`
again (rallies come from the trajectory), `players`.

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
  share of rally frames over 4 m/s is reported (5-10%); what raises
  `PlayerCheckError` inside rallies is what one person cannot do — a step
  over 1.5 m between frames (swaps, hidden feet, camera bumps: 2-3 m), or
  over 6 m/s averaged over 1 s.

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
movement check jumps, sprints, swaps and flicker. Tests that need the
TrackNet checkpoint or checkout skip when it is absent.
