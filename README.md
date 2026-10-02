# badminton-analysis

Per-shot quality assessment from badminton match video. See
`docs/badminton-analysis-plan.md` for the full build plan.

**Status: phase 1 built and unit-tested; run on a 60 s BWF clip, overlay not
yet signed off. Phase 2 (segmentation) rebuilt around a court-line template
after the colour heuristic failed on real footage; checked by eye on two
broadcasts, hand-marked precision/recall not yet run. Progress notes:
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
```

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
```

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

## Tests

```bash
uv run pytest
```

The tests build their own synthetic videos: one where a marker's x position
encodes the frame index (catches seek and off-by-one errors), and a fake
broadcast — play view, crowd, play view, with a three-frame flash — for the
segmenter. Tests that need the TrackNet checkpoint or checkout skip when it
is absent.
