# badminton-analysis

Per-shot quality assessment from badminton match video. See
`docs/badminton-analysis-plan.md` for the full build plan.

**Status: phase 1 (scaffolding + TrackNet wrapper).**

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# TrackNetV3 — frozen dependency, never edited, gitignored
git clone https://github.com/qaz812345/TrackNetV3.git external/TrackNetV3
uv run gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O /tmp/tnv3_ckpts.zip
unzip -q /tmp/tnv3_ckpts.zip -d /tmp/tnv3_ckpts
cp /tmp/tnv3_ckpts/ckpts/*.pt external/TrackNetV3/ckpts/
```

## Use

```bash
bda info  --video data/raw/match.mp4
bda track --video data/raw/match.mp4 --match-id msia_open_f --seconds 60 --overlay
bda overlay --video data/raw/match.mp4 --match-id msia_open_f   # re-render from cache
```

Outputs land in `data/cache/<match_id>/`:

| File | Contents |
|---|---|
| `shuttle.csv` | `frame, x, y, visible, confidence` in source-video pixels |
| `shuttle.meta.json` | video path and frame range that produced it |
| `overlay.mp4` | trajectory drawn onto the video, for eyeballing |

Every stage caches and skips work when its output exists. `--force`
recomputes.

## Config

`configs/default.yaml`. Override any key from the command line:

```bash
bda track ... -o device=cpu -o shuttle.batch_size=4 -o shuttle.eval_mode=nonoverlap
```

`device: auto` picks CUDA, then MPS, then CPU. Nothing hardcodes a device:
development runs on Apple Silicon, production on an RTX 5070. TrackNet output
on MPS matches CPU to ~1e-6.

Measured throughput: **~16 fps on M-series MPS** in `weight` mode. `nonoverlap`
is roughly 8x faster and less accurate. Frame rate does not depend on source
resolution — every frame is resized to 512x288 for the model.

## Layout

```
src/config.py    config loading, path resolution, cache directories
src/video.py     decode, frame iteration, clip extraction, overlay rendering
src/cli.py       command line entry point
src/shuttle.py   TrackNetV3 wrapper — the only module that imports external/
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

## Tests

```bash
uv run pytest
```

The tests build their own synthetic video, in which the marker's x position
encodes the frame index — that is what catches seek and off-by-one errors.
Tests that need the TrackNet checkpoint skip when it is absent.
