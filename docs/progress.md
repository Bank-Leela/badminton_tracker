# Progress

Last updated 2026-10-03 evening. Picks up from `badminton-analysis-plan.md`.

## Where things stand

| Phase | Built | Acceptance check |
|---|---|---|
| 1 — TrackNet wrapper | yes; play-only tracking added | **all 32 matches tracked**; 30 fps and 25 fps 60 s overlays **waiting on me to watch** |
| 2 — Segmentation | rebuilt 2026-10-01; hardened on 32 matches | rallies match real point totals (below); **hand-marked P/R not done** |
| 3 — Court homography | yes (branch `phase3-court`) | **all 32 pass** the length/width check; overlays **waiting on me to look** |

Phase 2 is on `main` (pushed). Phase 3 + footage docs are committed on local
branch `phase3-court` (not merged, not pushed). **Everything since the night
of 2026-10-02 is uncommitted** — see "Overnight" below.

## Phases 1-3 on all 32 matches (finished 2026-10-03)

| match | fps | min | play % | thr | spans | court (m) | H | detected % | rallies | points | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| km_ae2019_f_axelsen | 25 | 111 | 20 | 0.53 | 185 | 13.40 x 6.10 | 185/185 | 82 | 89 | 104 | **86%** |
| km_chn2019_f_ginting | 25 | 110 | 25 | 0.68 | 122 | 13.40 x 6.10 | 122/122 | 82 | 118 | 118 | 100% |
| km_den2018_f_chou | 25 | 101 | 31 | 0.49 | 121 | 13.39 x 6.10 | 121/121 | 66 | 114 | 115 | 99% |
| km_den2019_f_chenlong | 25 | 65 | 26 | 0.65 | 69 | 13.40 x 6.10 | 69/69 | 64 | 63 | 68 | 93% |
| km_inam2019_f_antonsen | 25 | 97 | 22 | 0.66 | 106 | 13.40 x 6.10 | 106/106 | 76 | 104 | 109 | 95% |
| km_jpn2018_qf_lindan | 25 | 55 | 31 | 0.61 | 70 | 13.41 x 6.10 | 70/70 | 76 | 63 | 60 | 105% |
| km_jpn2019_f_christie | 25 | 74 | 24 | 0.59 | 73 | 13.40 x 6.10 | 73/73 | 73 | 72 | 71 | 101% |
| km_mas2018_f_leechongwei | 25 | 89 | 31 | 0.46 | 84 | 13.40 x 6.10 | 84/84 | 76 | 83 | 82 | 101% |
| km_mym2020_f_axelsen | 25 | 76 | 30 | 0.55 | 78 | 13.39 x 6.10 | 78/78 | 73 | 78 | 78 | 100% |
| km_sgp2019_f_ginting | 25 | 93 | 18 | 0.59 | 89 | 13.38 x 6.10 | 89/89 | 82 | 84 | 105 | **80%** |
| km_wc2018_f_shiyuqi | 25 | 69 | 27 | 0.66 | 65 | 13.40 x 6.10 | 65/65 | 79 | 65 | 66 | 98% |
| km_wc2019_f_antonsen | 25 | 80 | 19 | 0.65 | 84 | 13.39 x 6.10 | 84/84 | 61 | 56 | 54 | 104% |
| km_wtf2019_f_ginting | 25 | 113 | 22 | 0.65 | 111 | 13.40 x 6.10 | 111/111 | 84 | 111 | 111 | 100% |
| kv_ae2026_sf_linchunyi | 30 | 90 | 32 | 0.65 | 111 | 13.39 x 6.10 | 111/111 | 75 | 108 | 111 | 97% |
| kv_arc2025_f_chou | 30 | 92 | 38 | 0.62 | 110 | 13.40 x 6.10 | 110/110 | 59 | 108 | 106 | 102% |
| kv_cm2026_qf_antonsen | 30 | 110 | 27 | 0.50 | 114 | 13.41 x 6.10 | 114/114 | 85 | 114 | 116 | 98% |
| kv_den2025_qf_axelsen | 30 | 82 | 33 | 0.53 | 106 | 13.39 x 6.10 | 106/106 | 61 | 100 | 106 | 94% |
| kv_fra2024_f_shiyuqi | 30 | 77 | 28 | 0.65 | 89 | 13.40 x 6.10 | 89/89 | 79 | 84 | 82 | 102% |
| kv_fra2025_sf_popov | 30 | 61 | 29 | 0.55 | 82 | 13.39 x 6.10 | 82/82 | 76 | 77 | 74 | 104% |
| kv_ina2024_sf_antonsen | 30 | 102 | 31 | 0.54 | 110 | 13.40 x 6.10 | 110/110 | 76 | 108 | 108 | 100% |
| kv_inam2025_f_christie | 30 | 92 | 30 | 0.53 | 120 | 13.40 x 6.10 | 120/120 | 74 | 117 | 116 | 101% |
| kv_ind2023_f_axelsen | 30 | 86 | 30 | 0.69 | 106 | 13.40 x 6.10 | 106/106 | 76 | 106 | 106 | 100% |
| kv_ind2026_qf_lohkeanyew | 30 | 81 | 34 | 0.46 | 110 | 13.40 x 6.10 | 110/110 | 79 | 109 | 109 | 100% |
| kv_jpn2026_qf_tanaka | 30 | 80 | 29 | 0.56 | 118 | 13.39 x 6.10 | 118/118 | 78 | 119 | 118 | 101% |
| kv_mas2026_f_shiyuqi | 30 | 56 | 21 | 0.68 | 51 | 13.39 x 6.10 | 51/51 | 74 | 51 | 51 | 100% |
| kv_sgp2025_f_luguangzu | 30 | 68 | 16 | 0.55 | 58 | 13.39 x 6.10 | 58/58 | 84 | 57 | 58 | 98% |
| kv_tha2026_f_antonsen | 30 | 120 | 26 | 0.71 | 114 | 13.39 x 6.10 | 114/114 | 79 | 113 | 115 | 98% |
| kv_wc2022_f_axelsen | 30 | 75 | 25 | 0.66 | 63 | 13.39 x 6.10 | 63/63 | 82 | 63 | 63 | 100% |
| kv_wc2023_f_naraoka | 30 | 129 | 34 | 0.60 | 127 | 13.40 x 6.10 | 127/127 | 79 | 116 | 107 | **108%** |
| kv_wc2025_f_shiyuqi | 30 | 103 | 25 | 0.61 | 106 | 13.40 x 6.10 | 106/106 | 77 | 106 | 110 | 96% |
| kv_wc2026_qf_lanier | 30 | 90 | 31 | 0.53 | 115 | 13.40 x 6.10 | 115/115 | 76 | 114 | 114 | 100% |
| kv_wtf2025_sf_shiyuqi | 30 | 62 | 30 | 0.46 | 71 | 13.39 x 6.10 | 71/71 | 83 | 71 | 71 | 100% |

**2,941 rallies for 2,982 real points (98.6%)**; 29 of 32 matches within
93-105%. Points = the final score, read off the score graphic at each
match's last play span. `thr` = the automatic play threshold; `H` = play
spans with a homography; `detected %` = play frames with a shuttle.

The three outliers are understood: **All England 2019 (86%) and Singapore
2019 (80%)** show some points entirely from side cameras, which "main camera
only" drops by design; **World Champs 2023 (108%)** replays points from the
main camera after a logo wipe (decision below).

## Overnight 2026-10-02 → 03 (Claude, while I slept)

I said: finish the 32 videos, then do what you can. Claude did not push,
merge, download anything, or sign off any acceptance check.

### Phases 2-3 on all 32 matches: done

Every match: play detection + court fit. All 32 courts measure 13.38-13.41 m
x 6.10 m; every play span of every match has a homography, all from one fit
per match (no broadcast's main camera moved). Montage of all 32 overlays
checked by eye: lines on lines on green, red (World Tour Finals 2019/2025)
and grey (All England 2026) mats. Play share 16-38% of each video.

Four real failures turned up on the new venues, each fixed in
`src/segment.py` with a regression test:

1. **All England 2019 — score graphic outvoted the court lines.** The score
   bug is white in 54% of frames, the faint lines in 15-25%, so the pass-1
   template was mostly score bug and matched dark close-ups best. Fix: pass 1
   now picks the group of frames sharing the largest fixed layout
   (`_dominant_layout`), not the most frequent pixels.
2. **No fixed threshold fits every broadcast.** Play view scores 0.65-0.85
   in 2018-19 (score graphic on screen during play), 0.75-0.95 in 2026 with a
   non-play tail up to 0.55. Fix: `play_score: auto` — the valley of each
   match's score histogram (chose 0.46-0.71 across the 32).
3. **All England 2026 — 0.2-1.7 s glimpses of close-ups** passed; the court
   check caught all 18. Fix: play spans under `play_min_s` (2 s) dropped.
4. **India Open 2023 — the stream opens on the previous match** (men's
   doubles final, same court, same camera), which passed as play. Fix: play
   separated from the rest by more than `match_gap_min` (6 min) is dropped;
   the log names it. (Longest in-match gap across all 32: 4.7 min.)

Noted, not changed: 2018-19 broadcasts (and Singapore 2019 especially) show
more play from low side cameras, so "main camera only" drops more there.

### Shuttle tracking: play spans only

`bda track --play-only` (new) tracks only the play spans from
`view_segments.csv`, with one background median per match built from
play-view frames — a cleaner background than a whole-broadcast median, and
~30% of the frames to track. ~50 fps on the 5070.

- **25 fps check (Momota, Worlds 2019 F):** 60 s run, 86-88% detected
  during a long rally; 10 of 12 spot-checked detections in play are on the
  shuttle; the false ones are in a close-up after the span, which
  play-only tracking never sees. Overlay to watch:
  `data/cache/km_wc2019_60s/overlay.mp4`.
- **Rally counts vs real points** (points read off the score graphic at the
  last play span; the last span is match point). `rally_max_gap_s` raised
  2 → 4 s first: at 2 s, shuttles lost at a clear's apex split rallies
  (Arctic: 127 rallies for 106 points).

  | match | points | rallies | |
  |---|---|---|---|
  | All England 2026 SF (Lin 21-14, 18-21, 21-16) | 111 | 109 | 98% |
  | Arctic Open 2025 F (Chou 21-11, 13-21, 21-19) | 106 | 108 | 102% |
  | China Masters 2026 QF (Antonsen 18-21, 21-17, 21-18) | 116 | 114 | 98% |
  | Denmark Open 2025 QF (Axelsen 13-21, 21-12, 21-18) | 106 | 101 | 95% |
  | French Open 2024 F (Shi 22-20, 21-19) | 82 | 84 | 102% |
  | French Open 2025 SF (Popov 21-11, 22-20) | 74 | 77 | 104% |
  | Malaysia Open 2026 F (Kunlavut 23-21, 6-1, Shi retired) | ~51 | 51 | 100% |
  | Singapore Open 2025 F (Kunlavut 21-6, 21-10) | 58 | 57 | 98% |
  | India Open 2026 QF (Loh 14-21, 21-15, 21-17) | 109 | 109 | 100% |
  | Japan Open 2026 QF (Tanaka 20-22, 21-16, 21-18) | 118 | 119 | 101% |
  | Indonesia Masters 2025 F (Kunlavut 18-21, 21-17, 21-18) | 116 | 117 | 101% |
  | India Open 2023 F (Kunlavut 22-20, 10-21, 21-12) | 106 | 106 | 100% |
  | Indonesia Open 2024 SF (Antonsen 21-15, 19-21, 21-11) | 108 | 108 | 100% |

  | Thailand Open 2026 F (Antonsen 9-21, 24-22, 21-18) | 115 | 113 | 98% |
  | World Champs 2022 F (Axelsen 21-5, 21-16) | 63 | 63 | 100% |
  | **World Champs 2023 F (Kunlavut 19-21, 21-18, 21-7)** | 107 | **116** | **108%** |
  | China Open 2019 F (Momota 19-21, 21-17, 21-19) | 118 | 118 | 100% |
  | Denmark Open 2018 F (Momota 22-20, 16-21, 21-15) | 115 | 114 | 99% |
  | Denmark Open 2019 F (Momota 21-14, 21-12) | 68 | 63 | 93% |
  | **All England 2019 F (Momota 21-11, 15-21, 21-15)** | 104 | **89** | **86%** |

- **2018-19 broadcasts cut to a side camera mid-rally** (~1 s, then back).
  Each main-camera piece became its own rally: All England 2019 had **170
  rallies for 104 points**. Fix (method 6): rally pieces less than
  `rally_max_gap_s` apart share one rally id, so a rally can span several
  play segments with the off-camera second between them (no tracking
  there). Modern broadcasts barely change (two matches lose one rally
  each). All England 2019 → 89 (86%): the rest are points that broadcast
  showed entirely off the main camera — the price of "main camera only" on
  2018-19 footage. `segments.csv` consumers: a rally is all rows with that
  `rally_id`, possibly in more than one segment.

  15 of 16 matches within 95-104%: rallies are usable as the unit for the
  free rally-outcome labels. Exact rally start/end is phase 5's job
  (contacts). Note Malaysia 2026 ends in a retirement — a rally-outcome
  label must not treat the last point as a normal winner.

- **Known gap confirmed — replays from the main camera.** World Champs 2023
  has 127 play spans for 107 points: some spans are slow-motion replays shot
  by the main camera, and the frame before them is the tournament's logo
  wipe (the broadcast's replay transition). Only this broadcast so far.
  Shuttle speed alone does not separate them (between-point spans, shuttle
  in hand, are slow too). **Decision for me:** a candidate fix is "a play
  span that starts right after a logo wipe is a replay" — the wipe is one
  repeated, identical graphic per broadcast, so it can be learned per
  match like the line template. Or accept ~8% duplicate rallies on this one
  match, or drop it from training.

### Code changes (uncommitted)

- `src/segment.py`: dominant-layout seed, auto threshold, `play_min_s`,
  main-match cluster, rallies bridging short cuts away from the main
  camera, narrower near-threshold report; method version 6 in the cache
  key so all of this recomputes from cached masks.
- `src/shuttle.py`: `track_play_spans`, `median_input`/`verbose` on
  `track_shuttle`; cache keyed on a fingerprint of the play spans, not the
  file's mtime.
- `src/court.py`: same content fingerprint for its cache.
- `src/cli.py`: `bda track --play-only`.
- `configs/default.yaml`: `play_score: auto`, `play_min_s`, `match_gap_min`,
  `rally_max_gap_s: 4.0`; `template_min_freq` removed.
- Tests: 20 segment (was 13), 9 court, 21 shuttle — all passing.

### Tracking: done (2026-10-03, paused 09:10-16:00 for my GPU)

All 32 matches tracked over their play spans (~50 fps; ~7 h of GPU in
total), then one consistency pass (play detection + court fit) over all 32
with the final code: no errors, every play span has a homography. Results
in the table at the top.

To process a new match from scratch (WSL, venv active):

```fish
set id <file stem in data/raw>
bda segment --video data/raw/$id.mp4 --match-id $id --sheet
bda court   --video data/raw/$id.mp4 --match-id $id
bda track   --video data/raw/$id.mp4 --match-id $id --play-only
bda segment --video data/raw/$id.mp4 --match-id $id      # rallies, from the trajectory
```

Overviews: `data/cache/_overview/` (all 32 court overlays in one image).

## Decided 2026-10-02: training footage

- **Players:** men's singles featuring Kunlavut Vitidsarn or Kento Momota.
  Every match also contributes the opponent's shots, so ~25 other players.
- **Momota in his prime only:** 2018 to the Jan 2020 car accident. His four
  matches in the Shuttlecock Trajectory Dataset (TrackNet's training data)
  are excluded: 2019 Fuzhou and Korea Open finals vs Chou Tien Chen, 2019
  World Tour Finals SF vs Wang Tzu Wei, 2019 Indonesia Open R16 vs Huang Yu
  Xiang. The two Kunlavut-vs-Momota matches (2022, 2023) are post-injury, so
  left out.
- **32 matches, ~48 GB:** 19 Kunlavut (2022-2026) + 13 prime Momota
  (2018-2020). The list, with YouTube ids, is `docs/footage.csv` — the only
  record, since `data/raw/` is gitignored.
- Why this many: shot count was never the problem (~1,000 shots per match);
  match and player variety is, and splits must be by match. Plan: run the
  automatic pipeline on all of them, give every shot the free rally-outcome
  label, hand-label a spread subset.
- **Downloaded 2026-10-02, all 32 verified** (open, 1080p, decode to the
  end): 48.5 GB, 46.4 h of video. ~8 MB/s; one YouTube 403 mid-file
  (Worlds 2023 F), fixed by re-running the loop, which resumes.
- **Frame rate differs:** the 13 Momota 2018-2020 broadcasts are **25 fps**;
  everything 2022+ is 30 fps. The pipeline is fine with that (all timing
  config is in seconds, fps read per video), but TrackNet's accuracy at
  25 fps is unchecked — run the phase 1 overlay check on one `km_` match.
- Shuttle tracking all 42+ hours of video is ~a day of GPU at the measured
  speed; worth tracking only play spans (about half the frames) first.

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
   Quicker: `check_bounds.jpg` / `check_rejected.jpg` in each `_10m` cache
   folder — green tiles must be the main camera, red ones must not.
3. **Sign off phase 3:** open `court_overlay.png` in both `_10m` cache
   folders — cyan lines must sit on the painted lines.
4. **Try `bda court-click` once** — the click window is untested by hand
   (Claude can't click). It needs WSL's display; `--segment 0` on a test
   match id is harmless.
5. **Commit** phase 3. Then phase 4 (players: YOLO-pose, needs a model
   download).

Running `bda` — open a terminal in the repo folder, then one at a time:

```fish
wsl
source .venv/bin/activate.fish
explorer.exe (wslpath -w data/cache/wc2026_ms_f_10m/court_overlay.png)
explorer.exe (wslpath -w data/cache/cm2026_ws_f_10m/court_overlay.png)
bda court-click --video data/raw/wc2026_ms_f.mp4 --match-id wc2026_ms_f_10m --segment 0
```

(From PowerShell instead: `ii data\cache\wc2026_ms_f_10m\court_overlay.png`.)

## Phase 3 results (2026-10-01)

`bda court` on both 10-minute chunks, ~24 s and <1 GB each:

| match id | length | width | line cost | play spans with a homography |
|---|---|---|---|---|
| `wc2026_ms_f_10m` | 13.40 m | 6.10 m | 0.01 px | 15/15, all from the one match fit |
| `cm2026_ws_f_10m` | 13.40 m | 6.10 m | 0.06 px | 19/19, all from the one match fit |

- Zoomed 4x on all four corners and the centre-line / short-service
  crossings: the projected lines sit within ~1 px of the painted ones.
- **Independent check:** the net posts were not used in the fit, but stand
  on the doubles sidelines at the middle of the court. Their feet, read by
  hand off the background (+/-3 px, about +/-0.08 m), map to x = -3.06/+3.07,
  y = 0.03/0.05 (Worlds) and x = -3.05/+3.06, y = 0.00/0.02 (China Masters).
- No span needed a refit: the main camera did not move in either chunk.

How it works and what was learned building it (details in `src/court.py`):

- Saturation-based "white" misses horizontal lines: 4:2:0 chroma smears 2-3 px
  lines into the green. Brightness top-hat finds all of them.
- Hough segments sit on stroke *edges*; candidates are ranked on the
  skeleton by longest run with mat on both sides, so crowd and lettering —
  plenty of Hough segments, but bright neighbours — fall out.
- Coordinates: origin at court centre, x right as seen from the camera, y
  away from it. `homography.json` stores `image_to_court` per play span;
  `court.load_homographies()` reads it for phase 4.

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
