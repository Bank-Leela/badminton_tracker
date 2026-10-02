# Progress

Last updated 2026-10-01. Picks up from `badminton-analysis-plan.md`.

## Where things stand

| Phase | Built | Acceptance check |
|---|---|---|
| 1 — TrackNet wrapper | yes | 60 s run done; **waiting on me to watch the overlay** |
| 2 — Segmentation | **rebuilt** 2026-10-01 (line template) | checked by eye on both videos; **hand-marked P/R not done** |

Committed, merged into `main` and pushed (2026-10-01).

## Decided 2026-10-01

1. **Replace** the court-colour play-view heuristic with the line-template
   detector. Done — `src/segment.py`, config `segment:` section, tests, README.
2. **"Play" = main camera only.** Live rallies the director shows from a side
   camera are dropped by design.

## Resume here

1. **Sign off phase 1:** watch `data/cache/wc2026_ms_f/overlay.mp4`.
2. **Sign off phase 2:** look at both contact sheets (green = play). If the
   plan's formal number is wanted, hand-mark play spans in
   `data/labels/<match id>.play.csv` (`start_frame,end_frame`) and run
   `bda segment-eval`. Claude's eye check found every boundary right, but
   that is not an independent hand-marked truth.
3. Then phase 3 (court homography). Note the template already finds the
   main camera's court lines — a head start for line detection.

Running `bda` — open a terminal in the repo folder, then one at a time:

```fish
wsl
source .venv/bin/activate.fish
explorer.exe (wslpath -w data/cache/wc2026_ms_f_10m/segments.png)
explorer.exe (wslpath -w data/cache/cm2026_ws_f_10m/segments.png)
bda segment-eval --match-id wc2026_ms_f_10m --truth data/labels/wc2026_ms_f_10m.play.csv
```

- `wsl` first — `bda` only exists inside WSL. Typing `bda` in PowerShell gives
  "not recognized".
- The WSL shell is **fish**, so it's `activate.fish`, not `activate`.
- From PowerShell without entering WSL: `wsl .venv/bin/bda track ...`
- `track` on 10 minutes takes about 6 minutes; `segment` about 2.

## Phase 2 results (2026-10-01)

### With the new detector (current code)

| match id | range | `track` | `segment` |
|---|---|---|---|
| `wc2026_ms_f_10m` | f13800–31800 | done (46% frames detected) | 15 play spans (46% of frames), **15 rallies** (old code: 27, counting "rallies" inside close-ups and replays) |
| `cm2026_ws_f_10m` | f22200–40200 | not run | 19 play spans (37% of frames); no rallies without `track` |

0.1–0.2% of frames scored near `play_score`; `segment` takes ~1.5 min per
10 minutes (decode), seconds when re-run from cached masks. Spans are
identical to the hand-checked prototype ones on China Masters and within one
mid-crossfade frame on Worlds.

Not yet looked at: Worlds rallies 2 and 3 sit in one play span with a 2.1 s
gap — one point split in two, or two points with no cut between? Rally
splitting is unchanged from before and tuned by `rally_*` keys.

### The old heuristic failed

It calls **85–86% of both chunks play; the real share is 37–47%.**
Against the main-camera spans below (checked by eye), its frame-level
precision is **54% (Worlds) and 43% (China Masters)**. Recall is 100%. The
bar is 90% on both.

Two separate failures — threshold tuning can only touch the second:

- **Missed cuts.** Crossfades and logo wipes don't spike the colour histogram,
  and nor do cuts between two shots of green court on purple. So camera
  segments mix main view with replays and close-ups — e.g. Worlds #18 is half
  a low-angle replay, #23 starts with a 9 s close-up. The rally splitter then
  finds "rallies" inside the close-ups.
- **Wrong play-view verdicts.** `court_hue_ranges` counts blue as court, and
  Naraoka's team wears blue — close-ups and coach shots pass at
  `court_frac` 0.25–0.55. The Hawk-Eye graphic (green field, white line) is the
  most "court-like" thing on the sheet. Replays from the overhead and side
  cameras pass too.

The broadcasts bracket every replay with a logo wipe (World Championships
logo / HSBC World Tour logo).

### The replacement: match the main camera's line layout

The main camera is fixed, so its court lines land on the same pixels in every
play-view frame. Per frame: white-pixel mask at 320x180. Template = pixels
white in most play frames and few others (two passes; the second removes the
score graphic, which is white in nearly every frame). Score = geometric mean
of template recall and on-template precision. No thresholds to tune per
venue, no cut detection, no colour.

Score is cleanly bimodal on both videos — almost nothing between 0.40 and
0.85; transitions are the only in-between frames:

| | main view (≥0.85) | everything else | in between |
|---|---|---|---|
| Worlds | 8,326 frames | 9,620 (all < 0.40) | ~50 (crossfades) |
| China Masters | 6,561 frames | 11,391 (all < 0.35) | ~40 (wipes) |

Checked by eye on both videos: every span boundary (frames ±15 around each
edge) is right, and frames sampled every 2 s from everything rejected
contain no main-camera view. Decode costs ~85 s per 10 minutes — same as the
colour-signature pass it replaced.

Now in `src/segment.py`; the tests' fake broadcast includes a score-graphic
box in every frame and a second camera with green mat and white lines in a
different layout (the case the old heuristic got wrong).

### The catch: live play from another camera

China Masters f35190–36030 (19:33–20:01): a whole rally shown live from a
side camera — confirmed by the score graphic (An serving at 9–7 throughout,
Miyazaki serving at 9–8 straight after). The template detector misses it
(~810 frames). Every other alternate-angle stretch checked was bracketed by
the logo wipe, i.e. a replay. None in the Worlds chunk.

This is why decision 2 matters. If "play" = main camera only, the detector is
~100% precision and recall. If "play" = any live camera, recall on China
Masters drops to ~89% and catching side-camera rallies needs something else
(e.g. treat anything not between two logo wipes as live).

Claude's recommendation is main camera only: everything after phase 2
(homography, player positions in court metres) is far easier from the fixed
main camera, and losing an occasional rally costs data volume, while letting
replays through duplicates rallies — bad for labels and train/test splits.

### Risks of the template approach

- **Assumes a static main camera.** True in both broadcasts (no in-between
  scores), but the plan expects pan/zoom. A zooming camera would score low
  and be dropped, not misclassified.
- **Assumes the main view is common in the range** (the template takes pixels
  white in >25% of frames). Fine for match footage; a range that's mostly
  intro or interval would need the template built from the whole match.

## Test footage

Both in `data/raw/` (gitignored). 2026 singles finals — outside the
Shuttlecock Trajectory Dataset, which only covers 2018–2021. 1920x1080,
30 fps, H.264, video only (no audio). Full broadcasts from BWF TV's
live-stream archive, so they include intro graphics, crowd shots, close-ups
and replays.

| File | Match | Length | Play starts |
|---|---|---|---|
| `wc2026_ms_f.mp4` | World Championships 2026 MS Final — Naraoka vs Lanier (`go2uajSiBnE`) | 72 min, 129,600 frames | ~7:40, frame 13800 |
| `cm2026_ws_f.mp4` | China Masters 2026 WS Final — An Se Young vs Miyazaki (`1jtiSyVSZNw`) | 69.5 min, 125,100 frames | ~12:20, frame 22200 |

Re-download (in WSL):

```bash
uvx --from 'yt-dlp[default]' yt-dlp --js-runtimes node -f 137 -o data/raw/<name>.mp4 "https://www.youtube.com/watch?v=<id>"
```

Without `--js-runtimes node` YouTube returns 403. Format 137 is a single-file
1080p H.264 stream, so no ffmpeg is needed. Full matches are under the BWF TV
channel's **/streams** tab; its /videos tab is only 8-minute highlights.

## Phase 1 results — `wc2026_ms_f`, frames 14400–16200 (60 s)

Outputs in `data/cache/wc2026_ms_f/`: `shuttle.csv`, `overlay.mp4`.

- **Speed:** 1,800 frames in 53.3 s on CUDA, fp32, weight mode (median 9.4 s,
  decode+resize 10.6 s, inference 33.3 s). One unexplained ~7 s stall around
  frame 1200.
- **Detection rate:** 1,170 / 1,800 frames (65%). It follows play — 80–97%
  during rallies, 0–40% between points and in close-ups:

  ```
  8:00  41%   8:15  95%   8:30  81%   8:45  97%
  8:05  65%   8:20  81%   8:35  43%   8:50  59%
  8:10  29%   8:25   0%   8:40  94%   8:55  95%
  ```

- **Spot check:** cropped the source video around 12 detections spread over
  the minute. 9 are clearly on the shuttle, including against the purple
  surround and the ad boards — so coordinates are in the right pixel space.

**Failure modes (noted, not fixed — per the plan):**

- False detections in non-play views: a blurry close-up (f14475) and arena
  lights in a dark transition shot (f14615). Phase 2 should filter these views.
- f16197, 3 frames before the clip ends: detection on an empty ad board.
  Possibly a window-edge effect; one sample, unconfirmed.

**Still to do:** watch `overlay.mp4` and decide whether tracking is good
enough — that call is mine, not Claude's.

## Known issues

- **Shuttle cache is keyed by match id only.** A different `--start/--seconds`
  under the same `--match-id` silently reuses the old `shuttle.csv`. Use a new
  match id (as above) or `--force`. Documented in the README now; not changed.
- **Fixed:** the old segment re-run trap (`-o segment.x=...` silently reused
  `segments.csv`). It is now keyed on range + config + `shuttle.csv`.
- **Fixed:** the README's phase 2 steps reused the phase 1 match id.
- **README says `open`** — right for the macOS dev machine the config
  mentions, wrong in WSL (`explorer.exe (wslpath -w <file>)`). Left as is.
- **A replay from the main camera** would be classed as play — none seen in
  either chunk.
