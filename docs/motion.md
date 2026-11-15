# Active vision under head motion

## Why this repository

foveal-playground explores a static scene from a **fixed** head with two eyes. fov-render can now move the head
between looks (fov-render #20: a per-look `head` in the request, D19; no body frame, D20). foveal-motion is where the
moving head is tried, starting monocular: one foveated eye, the cyclopean `C` at the head origin. The expectation
(Luiz, 2026-11-09) is that the foveated sensor helps under translation through the optical flow it sees.

Decisions (Luiz, 2026-11-12, in Chat):

| # | Question | Decision |
|---|---|---|
| M-D1 | Where the work lives | an independent repository, `visgraf/foveal-motion`, carrying foveal-playground's history, with fov-render as a submodule |
| M-D2 | The first setting | monocular, the cyclopean head, the Foveated sensor |
| M-D3 | The first test | the head moves straight from the back of the Classroom to the front, near the board |
| M-D4 | Where the eye looks during it | both, as two runs of the same path: fixating the board centre (`target`) and straight ahead (`ahead`) |
| M-D5 | Path and height | yarbus-agent's measured placements, back_of_room → near_board, at standing height 1.6 m |
| M-D6 | The dolly's hole limit (Luiz, 2026-11-12, issue #1) | 0.046 in `paths/classroom_dolly.json`: the scene's 0.025 failed at heads 6–46 (0.0250–0.0383), every hole on the right-hand window band; the measured maximum plus about a fifth, as D18 |
| M-D7 | The next step after M1 (Luiz, 2026-11-12, in Chat) | M2: range from the head's motion on the foveated raster, with M1's path run again fixating a target far off the path's line |
| M-D8 | The side target (Luiz, 2026-11-12, issue #3, option A) | accept what `target_ray` finds, `worldMap` at (−2.116, 3.540, 1.323): a hanging map 3.3 cm proud of the front wall, visible from every head, 22°→47° off the heading; the target sits 6.5 cm above the map's lower edge, so the fovea holds a small depth step. The spec is unchanged |
| M-D9 | M3, depth from RGB log-polar flow (Luiz, 2026-11-14, in Chat; #5 reviewed there) | #5's plan amended (A1–A8 of #6: shift + radial-stretch matcher, radial-gradient texture score, patch sizes in native pixels, ray-based sensor-agnostic code, M2's bracket as the reference, interval calibration on the calibration room, static figures first, the oracle ladder O1–O3 before any RGB matching) and split: M3a (#6, Chat, calibration room) builds and freezes the estimator; M3b (#7, Code, Classroom `dolly-1`) runs it unchanged |
| M-D10 | Freeze M3a's estimator (Luiz, 2026-11-14, in Chat; #6) | option A: freeze the estimator as it stands (matcher settings, calibration procedure), run the full calibration room as declared and report its result, coverage failure included; then the Classroom with the frozen estimator and scale. Both runs on the workstation (#7): the calibration room first, because it provides the interval scale the Classroom reuses unchanged. Calibration set: even ka plus (1, 6), (3, 8), (5, 10), whose truth Chat's smoke tests opened |
| M-D11 | Visuals before calibration (Luiz, 2026-11-14, in Chat; #7) | the full calibration-room and Classroom runs (M-D10) move to the end, overnight. First a quick look at the Classroom: the frozen estimator on three stride-5 pairs, with the process figures and the provisional scale (1.68, from the smoke calibration), labelled provisional. The three pairs' truth is opened for the error overlays, so they are excluded from the final Classroom gate |
| M-D12 | M3's acquisition (Luiz, 2026-11-14, in Chat; #7) | the final M3 runs re-render the calibration room and the Classroom `ahead` path at 1024 spp with an independent seed per look (`run --seed-base 1000`): 64 spp is far noisier than a camera at a normal exposure (σ 20–30 % of the signal), 1024 spp about 5–7 %. The estimator stays frozen at `7bb5e70`; the interval scale is recalibrated on the new calibration-room renders; M1–M2 keep their 64-spp results; the quick-look pairs (0, 5), (20, 25), (40, 45) stay out of the Classroom gate (`--exclude`); runs of about 45 min in total are OK'd |

## Experiment M1, the dolly

**Path** ([`paths/classroom_dolly.json`](../paths/classroom_dolly.json)): from (0.897, −3.793, 1.6) to (0.897, 0.759,
1.6) m, 4.552 m along +Y toward the board, in 46 steps of 0.0990 m (47 looks), the head level and facing +Y (the
playground's H0 axes). The board centre (0.897, 3.259, 1.334) is 7.06 m away at the start and 2.51 m at the end.
**Looks:** fov-render's `Foveated()` (E2 2°, e_max 45°), eye `C`, profile `full` (253 × 253 raster, 50,269 rays),
64 spp, one `Session` for both modes, oracle access (Position, Object Index).

**Gaze modes.** `target`: `Gaze.verged_point(board centre)`; the eye turns to keep the board centre on the fovea, so
the retinal flow is translation plus the eye's compensating rotation, zero at the fixated point. `ahead`:
`Gaze.yaw_pitch(0, 0)`; the eye looks along the path, so the focus of expansion is the fovea's centre and the flow is
pure translation (looming).

**Preflight** (`motion.dolly preflight`, [`motion/bl_preflight.py`](../motion/bl_preflight.py)). fov-render's scene
gate checks holes, back-faces and clearance from the session's head only (its PLAN, "Head motion"). The preflight runs
the same sanity panorama (`fov_render.bl.scene.sanity_checks`, 720 px, 4 spp) from **every** head of the path, with the
scene's own limits (clearance 0.25 m, holes 0.025 for the Classroom's windows, D18), and from a control head 10 cm in
front of the board, which must fail. `run` refuses to start without a passing preflight.

**Checks** (`motion.dolly checks`, [`motion/analysis.py`](../motion/analysis.py)), each with a control that must fail:

| id | what | passes when | control (must fail) |
|---|---|---|---|
| M1.0 | each record agrees with its Position pass | p99.9 angle ≤ the coarsest foveated pixel (fov-render S4's limit) | directions turned by 1.5 coarsest pixels |
| M1.1 | each record sits on its path head | `head_origin_w`, `head_R_wh` equal the path's, exactly; eye centre within 1 µm | the path shifted by one step |
| M1.2 | in `target` mode the fovea's centre is on the board centre | the sample on the eye axis is within 2 pixels at its range of the target, every look | the target 10 cm sideways: no look on it |
| M1.3 | look k+1's surface reprojects onto look k | ≥ 0.80 of look k+1's hits that fall in look k's field land on a pixel whose own Position is within 2 pixels at the range (incidence up to ~70°), every pair | look k's eye centre 10 cm sideways must lose half of the agreeing points |

The 0.80 of M1.3 was set from first principles before any Classroom run (what is lost is occlusion, depth edges and
grazing surfaces); the calibration room gives 0.92–0.95 (below). If the Classroom falls under it, the reason is found
before anything moves (rule 2).

**The flow.** For each point that M1.3 finds in both looks (not occluded), its retinal displacement between looks k and
k+1 in the eye frame, in degrees and in raster pixels, against its eccentricity in look k+1 (`flow.npz`, and binned
medians in `checks.json`). `visualize` writes the contact sheet, the flow figure (curves by eccentricity, the flow field
of the middle step on the raster) and a side-by-side video.

## Results

### Chat sandbox, calibration room (CPU, `small` profile)

`python -m motion.dolly sandbox --run runs/calib-dolly` at `aa15748` (clean), fov-render `0f6fa7d`, bpy 5.2.1 wheel,
numpy 2.2.6, CPU, 16 spp, 126 × 126 raster. Path [`paths/calib_dolly.json`](../paths/calib_dolly.json): 2.1 m in 21
steps of 0.10 m toward the Siemens-star rings, 0.3 m left of and 0.1 m below the EYE's line. Chat sandbox numbers:

- Preflight: 22 heads pass (holes 0, nearest surface ≥ 0.711 m, no back-faces); the control head (0.10 m from the far
  wall) fails on clearance.
- M1.0–M1.3 all PASS, every control fails. M1.1 errors are exactly 0. M1.2: `target` within 4.4 mm of the target
  (tolerance up to 12.6 mm); `ahead` never on it. M1.3: agreeing fraction 0.922–0.934 (`target`), 0.937–0.949
  (`ahead`); with the control at most 0.341 and 0.335.
- 44 looks in 29.2 s, the whole sandbox in 47 s.

Median flow per 10 cm step, degrees | raster pixels, by eccentricity (Chat sandbox, calibration room):

| eccentricity (deg) | 0–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–30 | 30–45 |
|---|---|---|---|---|---|---|---|
| `target` | 0.095 \| 0.75 | 0.134 \| 0.89 | 0.157 \| 0.65 | 0.234 \| 0.61 | 0.348 \| 0.52 | 0.594 \| 0.49 | 0.998 \| 0.55 |
| `ahead` | 0.012 \| 0.09 | 0.028 \| 0.16 | 0.074 \| 0.25 | 0.146 \| 0.30 | 0.285 \| 0.34 | 0.508 \| 0.39 | 0.962 \| 0.52 |

A first reading, to be checked on the Classroom: in degrees the flow grows with eccentricity in both modes, as
translational flow does; on the log-polar raster it is nearly flat, a few tenths of a pixel per step from 5° to 45°,
because the sensor's pixels grow with eccentricity about as the flow does. Looking ahead leaves the fovea almost still
(the focus of expansion); fixating the target moves the near-foveal background most in pixels, the parallax around the
fixated point.

### Classroom (workstation)

Measured by Code (issue #1): `python -m motion.dolly preflight | run | checks | visualize --path
paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX` at `8c1856a` (clean), fov-render `0f6fa7d` (clean),
Blender 5.2.1, OPTIX, RTX 4090, profile `full` (253 × 253 raster), 64 spp. Records: [`docs/runs/issue-1/`](runs/issue-1/)
(`preflight.json`, `run.json`, `checks.json`, the figures).

- Preflight: all 47 heads pass with M-D6's hole limit 0.046 (holes 0.0205–0.0383, every hole on the right-hand window
  band; nearest surface ≥ 0.793 m; back-faces on closed meshes ≤ 5.8e-6); the control head fails on clearance
  (0.100 m). Under the scene's 0.025 heads 6–46 had failed ([`preflight-limit-0.025.json`](runs/issue-1/preflight-limit-0.025.json)).
- M1.0–M1.3 all PASS, every control fails. M1.0: p99.9 ≤ 0.00082° (limit 1.17°). M1.1 errors are exactly 0. M1.2:
  `target` within 7.7 µm of the board centre (tolerance 4.4–12.4 mm); `ahead` never on it (its fovea is 0.266 m
  higher, the eye height above the board centre). M1.3: agreeing fraction 0.900–0.989 (`target`), 0.904–0.990
  (`ahead`), lowest at the first pair and rising toward the board; with the control at most 0.040 and 0.038.
- 94 looks in 19.5 s (session start 0.5 s; the first look 0.82 s, then 0.18–0.21 s a look, of which 0.06 s render);
  the preflight's 48 panoramas (720 px, 4 spp) in 46–56 s.

Median flow per 9.9 cm step, degrees | raster pixels, by eccentricity (Code, workstation, Classroom):

| eccentricity (deg) | 0–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–30 | 30–45 |
|---|---|---|---|---|---|---|---|
| `target` | 0.013 \| 0.19 | 0.031 \| 0.35 | 0.071 \| 0.51 | 0.150 \| 0.64 | 0.311 \| 0.74 | 0.600 \| 0.93 | 1.062 \| 1.18 |
| `ahead` | 0.013 \| 0.20 | 0.031 \| 0.35 | 0.072 \| 0.52 | 0.151 \| 0.64 | 0.312 \| 0.75 | 0.606 \| 0.94 | 1.066 \| 1.16 |

A first reading (Code, to be checked): the two modes give the same flow to within 0.01°. Here the board centre is
only 2.2°–6.1° below the path's forward direction, and the fovea sees the board, a plane facing the head, in both
modes. The calibration room's target, off the path line, separated them. Unlike the calibration room, the flow in
raster pixels is not flat: it rises from 0.2 px near the fovea to 1.2 px at 30–45°, where the periphery sees desks,
floor and ceiling close by.

Chat's review (2026-11-12, from `checks.json` and the geometry; nothing re-rendered):

1. **Same flow in both modes: confirmed, and expected for this path.** In `target` mode the eye pitches down to keep
   the board centre, from atan(0.266 / 7.052) = 2.2° to atan(0.266 / 2.5) = 6.1°, so it turns 0.03° per step at the
   back and 0.24° at the end (computed). For a plane facing the head, a translation plus the rotation that holds a
   point of the plane still is, to first order, an expansion about that point, with the same rate as the expansion
   about the focus of expansion in `ahead` mode. So the board, which fills the fovea and most of the field, flows the
   same in both modes, only centred on a different point (the focus of expansion is 2–6° above the fovea in
   `target` mode, about 29–56 raster pixels). The modes differ only by the parallax of points off the board's
   plane, which scales with that 2–6° offset and is lost in the medians. To separate them, the target must be far
   off the path's line (M2).
2. **The rise in raster pixels: partly a correction to Chat's own sandbox reading.** Chat wrote that the flow on the
   raster is "nearly flat" in the calibration room. That holds for `target` there (0.61 → 0.55 px from 5–10° to
   30–45°), not for `ahead` (0.30 → 0.52 px, ×1.7), which rises about as the Classroom does (0.64 → 1.16 px, ×1.8).
   The Classroom's pixel values are about twice the sandbox's mainly because the `full` raster has twice the
   `small` raster's resolution (s0 0.05° against 0.1°): in degrees the two scenes agree within about 10 % at 30–45°
   (1.07° against 0.96°). On a log-polar raster the pixel grows as (E2 + e), the translational flow as
   sin(e) / range(e); the raster flow is flat only where the range is about constant, and in both rooms the
   periphery sees nearer surfaces (floor, ceiling, desks) than the centre.

## Experiment M2, range from the head's motion

**Question.** The head's step between looks is known (the path) and so are the eye's poses (the records). Two looks
of a point therefore give its range by triangulation, with the path segment as the baseline, like stereo with the
eyes one behind the other. The precision is limited by the correspondence: a point is seen at a pixel, not at an exact
direction, and the foveated raster's pixels grow with eccentricity (E2 + e). The parallax, on the other hand, is zero
at the focus of expansion and grows with the angle psi from the heading. Where the eye looks decides which pixels get
which parallax: looking ahead puts the finest pixels where the parallax vanishes; fixating a target off the path's
line puts them where it is large. M2 measures how much that matters.

**Measurement** ([`motion/depth.py`](../motion/depth.py), `python -m motion.dolly depth --run R`): for every pair of
looks (k, k + stride), strides 1, 5 and 10 (about 0.1, 0.5 and 1 m), and every surface sample of the later look:

- correspondence by the oracle, as M1.3 (projection into the earlier look; occluded points have no match);
- **pixel-limited**: triangulate the ray through the later sample's pixel centre with the ray through the centre of the
  earlier look's pixel that the point projects to; when the two pixels see the same direction (the flow is under
  about half a pixel) there is no range;
- error: |range − truth| / truth, truth from the Position pass; a range within 10 % is counted **usable** (a reporting
  threshold proposed by Chat, not a pass/fail), as a fraction of all the later look's hits at that eccentricity;
- the precision model to compare with: range · sqrt((omega_a + omega_b)/12) / (|T| sin psi);
- the depth figure (`visualize`): error and usable fraction by eccentricity, error by psi against the model, and an
  error map on the raster for the middle pair at stride 5.

**Check** M2.1: with the exact rays through the point, triangulation returns its true range (relative error
≤ 1e-6 where the rays are ≥ 0.01° apart); control: the earlier eye centre 1 cm along the path, which must give a median
relative error ≥ 1e-3.

**Off-axis target** (M-D7). [`paths/classroom_dolly_side.json`](../paths/classroom_dolly_side.json): M1's path, limits and
control head, with the target given as a ray (`target_ray`) from the first head toward the front wall's plane 2.9 m left
of the board centre, at its height: about 22° left of the heading at the back and 49° at the front (computed for that
point). The preflight now finds the target (the first opaque surface along the ray; non-opaque objects passed through as
in fov-render's S5) and requires it **visible from every head**; its control, a point 0.5 m behind the target, must be
hidden from all of them. M1's own run (`runs/dolly-1`) is reused for M2 as it is: both its modes, no new rendering.

### Chat sandbox, calibration room (CPU, `small` profile)

`python -m motion.dolly sandbox --run runs/calib-m2` (M1's calibration path, both modes) and
`... sandbox --run runs/calib-m2-side --path calib_dolly_side.json --modes target` (the same path fixating the room's
left wall toward (−3, 3, 1): 31° left of the heading at the start, 48° at the end, computed). A first side target on
the floor was hidden from heads 12–17 by the card `ring_e40_m225`, which the new preflight caught. M2.1 PASS in both
runs (max relative error 2.1e-8 and 1.1e-8), control fails (median 0.100). Chat sandbox numbers, at `4d9ef0a` (clean), fov-render
`0f6fa7d`, bpy 5.2.1 wheel, CPU:

| run, stride | fovea (< 2°): median error \| usable | whole field: usable | no range |
|---|---|---|---|
| `ahead`, 0.1 m | — (no range) \| 0.00 | 0.03 | 0.78 |
| `ahead`, 0.5 m | 0.391 \| 0.06 | 0.36 | 0.02 |
| `ahead`, 1.0 m | 0.223 \| 0.21 | 0.60 | 0.00 |
| `target` near the path line (5–12° off), 0.1 m | 0.168 \| 0.31 | 0.10 | 0.15 |
| `target` near the path line, 0.5 m | 0.043 \| 0.68 | 0.48 | 0.00 |
| `target` near the path line, 1.0 m | 0.025 \| 0.80 | 0.67 | 0.00 |
| `target` off the path line (31–48° off), 0.1 m | 0.030 \| 0.96 | 0.26 | 0.02 |
| `target` off the path line, 0.5 m | 0.010 \| 0.97 | 0.86 | 0.00 |
| `target` off the path line, 1.0 m | 0.005 \| 0.97 | 0.90 | 0.00 |

The measured median error is 0.70–1.13 times the model's. A first reading, to be checked on the Classroom: looking
along the path, the fovea gets no range at all from 10 cm steps and little from 1 m; fixating a point far off the
path's line, the fovea gives 1–3 % range from a single 10 cm step for nearly every sample it sees, and the whole field
improves with it. The finest pixels are worth most where the parallax is, and the eye can put them there.

### Classroom (workstation)

Measured by Code (issue #3), workstation, RTX 4090, Blender 5.2.1, OPTIX, profile `full`, 64 spp, fov-render `0f6fa7d`.
Records: [`docs/runs/issue-3/`](runs/issue-3/).

- **`dolly-1`**: M1's run (#1, rendered at `8c1856a`), both modes, reused as it is; `depth` at `780b050` (clean),
  again at `5f68240` (clean) with the bracket. M2.1 PASS (max relative error 2.8e-8), control fails (median 0.101).
- **`dolly-side`**: `paths/classroom_dolly_side.json`, `target` only, at `0de439a` (clean). The target that
  `target_ray` finds is on `worldMap` (M-D8) at (−2.116, 3.540, 1.323): 22.3° left of the heading at k = 0 (7.93 m), 47.3°
  at k = 46 (4.11 m). Preflight: 47 heads pass, the target visible from every head, the hidden point hidden from all;
  the control head fails on clearance. M1.0–M1.3 PASS, every control fails (M1.2 within 5.7 µm, tolerance 7.2–13.9 mm;
  M1.3 0.905–0.977, control ≤ 0.068). M2.1 PASS (max 2.0e-8), control fails (median 0.101). `depth` again at
  `5f68240` (clean) with the bracket.
- Times: `dolly-side` 47 looks in 10.8 s (the first 0.98 s, then 0.19–0.21 s a look); `depth` 3.6–6.8 s, 4.9–9.0 s with the
  bracket; the side preflight 46.7 s. With the bracket, every error-based value of both `depth.json` files is identical
  to the one before it (1,210 values compared; M2.1 too).

| run, stride | fovea (< 2°): median error \| usable | fovea: bracket median \| usable by bracket | fovea: same pixel | whole field: usable | no range | measured / model |
|---|---|---|---|---|---|---|
| `ahead`, 0.10 m | — (no range) \| 0.00 | no range \| 0.00 | 0.921 | 0.16 | 0.22 | 1.07 |
| `ahead`, 0.49 m | 0.164 \| 0.31 | 0.463 \| 0.00 | 0.053 | 0.69 | 0.00 | 1.48 |
| `ahead`, 0.99 m | 0.094 \| 0.53 | 0.213 \| 0.00 | 0.013 | 0.87 | 0.00 | 1.75 |
| `target` the board centre (2–6° off the heading), 0.10 m | 0.201 \| 0.26 | 0.549 \| 0.00 | 0.925 | 0.18 | 0.06 | 1.11 |
| `target` the board centre, 0.49 m | 0.046 \| 0.81 | 0.112 \| 0.43 | 0.053 | 0.73 | 0.00 | 1.41 |
| `target` the board centre, 0.99 m | 0.026 \| 0.96 | 0.061 \| 0.84 | 0.013 | 0.89 | 0.00 | 1.68 |
| `target` on `worldMap` (22–47° off), 0.10 m | 0.010 \| 0.996 | 0.073 \| 0.75 | 1.000 | 0.57 | 0.02 | 0.75 |
| `target` on `worldMap`, 0.49 m | 0.008 \| 0.992 | 0.015 \| 0.99 | 0.212 | 0.87 | 0.00 | 0.97 |
| `target` on `worldMap`, 0.99 m | 0.004 \| 0.995 | 0.008 \| 0.995 | 0.050 | 0.87 | 0.00 | 1.17 |

Measured / model is the median of |error| / (range · sqrt(omega_a / 12) / (|T| sin psi)); 0.87 is rounding alone.
*Note:* with the model before Chat's review, sqrt((omega_a + omega_b) / 12), the same column read, row by row, 0.75,
1.00, 1.14, 0.78, 0.96, 1.10, 0.53, 0.67, 0.78. The whole field usable **by bracket**: `ahead` 0.000, 0.101, 0.612;
the board centre 0.001, 0.244, 0.730; `worldMap` 0.167, 0.829, 0.851 (at 0.1, 0.5, 1 m).

**With the bracket** (Code, Classroom). Fixating the map, 22–47° off the heading, the fovea supports 7.3 % from one 10 cm
step (about three quarters of its samples within 10 %), 1.5 % from 0.5 m and 0.8 % from 1 m. That is close to the calibration room's
8.9 %, 1.9 % and 1.0 %, so Chat's weaker reading holds here too: about 7–9 % from one step, not 1–3 %. Fixating the
board centre, 2–6° off the heading, the fovea supports 55 % from 10 cm and 6.1 % from 1 m. Looking ahead it supports
nothing usable at any stride, even at 1 m, where its errors happen to be under 10 % for half the samples. The new ratio
rises with the stride on `dolly-1` (1.07–1.11 at 10 cm, 1.41–1.48 at 0.5 m, 1.68–1.75 at 1 m), as the sandbox's 1.0–1.7
did: beyond 10 cm, the errors outgrow rounding.

**Does the sandbox reading hold?** (Code.) *The fovea gets no range looking ahead*: yes. At 10 cm it gets none, and at
1 m it is still the worst part of the field (0.094 | 0.53, against 0.83–0.94 usable beyond 5°). *Fixating off the
path's line, the fovea gives 1–3 % range from a single 10 cm step*: yes in number (1.0 % at 10 cm, 0.8 % at 0.5 m,
0.4 % at 1 m, 99 % usable), but at 10 cm this is not parallax the fovea measured.
[`same_pixel.py`](runs/issue-3/same_pixel.py) ([output](runs/issue-3/same_pixel.txt)) shows that at a 10 cm step **every**
foveal point lands on the same pixel in both looks (its flow is under half a pixel). Its range is then where the rays
through that one pixel from the two eye centres cross, which the eye's known rotation between the looks decides: the
surface of zero flow through the fixated point. Near the target, the map and its wall lie close to that surface, so the
range comes out right. At 1 m only 5 % of foveal points are same-pixel, and the range of the others (0.40 %) is measured
flow. Fixating the board centre, 2–6° off the heading, the eye turns too little per step for this: 92.5 % of foveal points
are same-pixel at 10 cm, with 21 % median error. Looking ahead the eye does not turn, so the rays are parallel and there is
no range. The model counts pixel rounding only, so it does not describe the same-pixel range; that is why the measured /
model ratio drops to 0.53 at 10 cm on the side run (with the model before Chat's review; 0.75 with the
corrected one).

**Chat's review of #3** (2026-11-12, from the records and the sandbox; nothing re-rendered).

1. *Same pixel.* Code is right, and `motion/depth.py`'s docstring was wrong: rays through one pixel are parallel, so
   give no range, only when the eye does not turn between the looks. A same-pixel match is itself a measurement, though:
   the flow is under half a pixel. In inverse range it is an interval of the same width, about p / (|T| sin psi),
   whether or not the eye turns. Looking ahead the interval is centred on 1/range = 0, so it bounds the range only from
   below: no range. Fixating, it is centred on the zero-flow surface through the fixated point, so it bounds the range
   on both sides. This is the active-vision argument that fixation makes depth well posed, relative to the fixated
   point. The fixating estimate is therefore legitimate. What was missing is how much the measurement supports,
   whatever the error happened to be. On the side run's fovea at 10 cm the error came out under what rounding allows
   (measured/model 0.53 with the old model, about 0.75 with the corrected one, against 0.87 for rounding alone):
   the map lies near the zero-flow surface, so its points sit near the pixels' centres. That is the geometry's luck,
   not the sensor's precision.
2. *What changed in `motion/depth.py`*. Each range now carries a **bracket**: the relative half-width of the ranges along
   b's ray that project into the same pixel of a, the precision the measurement supports. `depth.json` reports its
   median and `usable_bracket_fraction` (bracket ≤ 10 %) next to the error-based ones, and the **same-pixel fraction**.
   The error-based numbers are unchanged: in the two sandbox runs all 882 compared values (medians, p90s, usable
   counts and fractions, by eccentricity and by psi) are identical, as is M2.1. The model's sigma is now
   sqrt(omega_a / 12), Code's hypothesis (b's Position is its pixel's mean, on the centre's ray to first order).
   Uniform rounding then gives a measured/model ratio of 0.87; the sandbox gives 1.0–1.7, so beyond 10 cm steps the
   errors are somewhat larger than rounding alone. `depth.json` and `checks.json` now record the code that computed
   them (`computed_by`), next to the run's commits (`git`) (Code's provenance note).
3. *The reading, with the bracket* (Chat sandbox, calibration room; fovea < 2°, bracket median | usable by bracket):

   | run | 0.1 m | 0.5 m | 1.0 m |
   |---|---|---|---|
   | `ahead` | no range \| 0.00 | 0.594 \| 0.00 | 0.556 \| 0.00 |
   | `target` near the path line (5–12° off) | 0.421 \| 0.02 | 0.106 \| 0.35 | 0.069 \| 0.80 |
   | `target` off the path line (31–48° off) | 0.089 \| 0.64 | 0.019 \| 0.97 | 0.010 \| 0.97 |

   The conclusion holds with the honest measure, but it is weaker at 10 cm. Fixating off the path line, the fovea
   supports about 9 % from one 10 cm step, not 1–3 %, and about 2 % from 0.5 m. Looking ahead, the fovea supports
   nothing usable at any stride. The Classroom's brackets are for Code to measure (`depth` again on both runs, no
   rendering).

## Experiment M3, depth from RGB log-polar flow (M3a #6, M3b #7)

**Question.** M2 got its correspondences from the renderer. M3 asks whether the image texture can supply them,
under the motion the Foveated sensor is matched to: a forward step along the eye's axis (`ahead`), where a static
point keeps its azimuth and moves only along q = ln tan e. Plan: #5, amended and split by M-D9 into M3a (#6, Chat,
calibration room: build, validate, freeze) and M3b (#7, Code, Classroom `dolly-1`: run unchanged).

**Geometry** ([`motion/logpolar.py`](../motion/logpolar.py), ray-based: a sensor enters only through `rays()` /
`pixel_of()`). For a step B along the axis, d = q_b − q_a, z_b = B / expm1(d), r_b = z_b / cos e_b,
P = c_b + r_b R_b d_B(e_b, φ). A displacement interval [d_lo, d_hi] gives a range interval; reaching d ≤ 0 leaves the
far bound unbounded. A range is **usable** when its interval's relative half-width is ≤ 10 % (the analogue of M2's
bracket); **within 10 %** is error-based (M2's "usable").

**Checks fixed before measuring** (each with a control that must fail):

| id | what | passes when | control (must fail) |
|---|---|---|---|
| M3.0 | `ahead` looks are aligned (#5's gate) | heading = eye axis in both looks and no rotation, ≤ 1e-6 rad; B > 0; the estimator's loader refuses oracle fields | 0.1° yaw of b; b's centre 1 cm sideways |
| M3 selftest | Stage 1 geometry (14 known answers, numpy only) | round trips < 1e-6 px (Foveated `small`/`full`, 90° pinhole); Jacobian = #5's closed form within 1e-5; #5's known answer within 1e-9; analytic points within 1e-6 | wrong row sign; a sensor with E2 = 2.2°; native radius used as pinhole radius |
| M3.O1a | O1 keeps azimuth | \|φ_a − φ_b\| ≤ 1e-6 rad at e_b ≥ 1° | b yawed 0.1° |
| M3.O1b | O1 depth | relative range error ≤ 1e-6 at parallax ≥ 0.01° | baseline × 1.01 (median ≥ 1e-3) |
| M3.O3 | O3 interval propagation | nominal 95 % intervals cover the truth within ± 0.01 | intervals × 0.5 (coverage ≤ 0.80) |
| M3.PLY | the cloud | reads back identical, finite; every point on its anchor ray (≤ 1e-3°) | x and y swapped |

**The oracle ladder** ([`motion/flow_depth_oracle.py`](../motion/flow_depth_oracle.py), evaluation side; the
estimator never imports it). O1: b's Position seen from both eye centres, exact. O2: b at its pixel centre, P's
projection into a rounded to a's pixel centre, interval half a's pixel along the radial line. O3: O1 with a's location
moved radially by Gaussian noise of σ native pixels (seed 0), interval ± 1.96 σ. All are labelled oracle.

### Chat sandbox, calibration room (M3a Stages 0–2)

Source `runs/m3-calib-source`: `paths/calib_dolly.json`, `ahead` only, profile `full` (253 × 253, as the Classroom),
64 spp, CPU, rendered at `9cf234b` (clean), fov-render `0f6fa7d`, bpy 5.2.1 wheel, Python 3.13, numpy 2.2.6;
22 looks in 248 s. Everything below computed at `2f4e4e8` (clean). Records: [`docs/runs/issue-6/`](runs/issue-6/)
([table](runs/issue-6/ladder.txt), [script](runs/issue-6/ladder_table.py)). Chat sandbox numbers:

- Baseline: `selftest` passes; the two-mode sandbox (`small`, 16 spp) reproduces the recorded M1/M2 numbers exactly.
  On the M3 source, M1.0, M1.1, M1.3 PASS (M1.3 0.930–0.962), M1.2 not applicable (no `target`), M3.0 PASS with
  axis and rotation errors exactly 0, every control fails; M2.1 PASS (max 2.1e-8).
- M3 selftest: 14/14 PASS, every control fails. (One control was corrected before any measurement: a wrong E2 moves
  the Jacobian in opposite directions through K and E2 + e, so its error crosses zero near one eccentricity; the
  control is now that the wrong sensor fails the 1e-5 check, max 0.036.)
- O1: M3.O1a max 4.4e-16 rad (control 0.099); M3.O1b max 8.2e-14 (control median 0.010); M3.PLY 771,333 points at
  stride 5, on their rays within 1.2e-13° (control 89°). O3: coverage 0.9501 at every σ (control 0.673); identical
  across σ because the same draws are scaled, which is what an exact propagation must give.

Usable range (interval ≤ 10 %) as a fraction of **all** of look b's hits, whole field:

| stride (B) | O1 exact | O2 pixel | O3 0.05 px | O3 0.10 px | O3 0.25 px | M2 bracket | M2 within 10 % |
|---|---|---|---|---|---|---|---|
| 1 (0.10 m) | 0.953 | 0.000 | 0.265 | 0.025 | 0.000 | 0.000 | 0.117 |
| 5 (0.50 m) | 0.903 | 0.169 | 0.890 | 0.835 | 0.227 | 0.170 | 0.661 |
| 10 (1.00 m) | 0.849 | 0.529 | 0.845 | 0.832 | 0.682 | 0.530 | 0.796 |

O1's row is the ceiling: the jointly visible share. A first reading, to be checked by the RGB matcher (Stage 5):

1. **M3 ties to M2.** O2, through M3's own pipeline, gives M2's bracket fraction to within 0.001 at strides 5 and 10,
   and its error-based fraction (0.661, 0.795 against 0.661, 0.796). Its interval covers the truth for 98–99 % of
   matches. The geometry, the depth formula and the interval are consistent with M2's triangulation.
2. **Sub-pixel matching is where the gain is, and the requirement is about 0.1 px.** At stride 5, 0.1 px
   correspondences make 0.835 of the field usable (5× M2's bracket, 92 % of the ceiling); 0.25 px only 0.227. The
   5–40° band at stride 5 is at its ceiling for σ ≤ 0.1 px. This turns #5's synthetic gate (median ≤ 0.1 px,
   p90 ≤ 0.25 px) from a guess into the measured requirement.
3. **The fovea still gives nothing at short baselines.** Below 2°, stride 5 needs σ ≈ 0.05 px (0.716); at 0.1 px it is
   0.025, at pixel rounding 0. Looking ahead, the fovea's depth is a long-baseline matter, as M2 found.

### The matcher and its synthetic fixtures (M3a Stage 3, Chat sandbox)

**Matcher** ([`motion/matcher.py`](../motion/matcher.py), observation only: two luminance rasters, the sensor model, B).
*Coarse* on the log-polar grid (q step 0.01, 800 azimuths): a fronto-parallel patch moves by one q shift
everywhere, so every node is searched at once by shifting the whole grid; zero-mean NCC over ± 3 native px radially and
± 1 native px in azimuth (window sizes per grid row, A3); d = 0 included. *Fine* on the native raster through
`pixel_of`, at b's own pixels between 1° and 44°: Gauss–Newton on I_b(q + u, φ + v) = α I_a(q + u − d − κu, φ + v) + β,
with κ the radial stretch (A1; about B / z_a on a surface parallel to the heading), started at κ = 0 and, where that fit's
NCC is under 0.97, also at 0.25 and 0.5; σ_d from the Gauss–Newton covariance. *Rejections*, each recorded: invalid
support, radial texture under 0.02 (A2: rms radial gradient per native px over the patch mean), NCC under 0.8, a second
peak within 0.05 outside ± 1.5 px, beyond the 20 m envelope (unresolved), near boundary (0.25 m), Gauss–Newton moved
> 1.5 px or κ outside [−0.5, 1.5], forward–backward disagreement > 1 px, and fewer than 3 of the 8 neighbouring pixels
holding a candidate within 0.5 px (a test, not a filter: nothing is smoothed or replaced).

*Added after the first fixture run* (at the uncommitted state before `afe3c58`; the gates were already fixed): that run
passed every gate but one, the corridor's parallel surfaces at B ≈ 1 m (p90 0.85 px, 24 % accepted). Diagnosis: wrong
peaks of the long one-dimensional search (median error 4 px, NCC 0.90) and Gauss–Newton stuck at κ ≈ 0 where the true
stretch was 0.25–0.4. Hence the neighbour-support test and the κ restarts. No threshold of the gates moved.

**Fixtures** ([`motion/flow_fixtures.py`](../motion/flow_fixtures.py)): analytic planes ray-cast through the Foveated warp
(`full`, 4 × 4 rays a pixel: the box filter), band-limited multi-scale texture, Gaussian noise 0.005 (about 1 % of the
mean), the dolly's axes, B of strides 1, 5, 10. Truth: the point on each pixel's centre ray, its exact d. Gates of #6
(median ≤ 0.1 px, p90 ≤ 0.25 px of a's native pixels over accepted matches, at least 100 accepted), by surface.

`python -m motion.dolly flow-depth-fixtures --out runs/m3-fixtures` at `afe3c58` (clean), 637 s, Chat sandbox
([`fixtures.json`](runs/issue-6/fixtures.json)). Accepted fraction | median | p90 (native px):

| fixture / surface | B 0.10 m | B 0.49 m | B 0.99 m |
|---|---|---|---|
| front (3 m) | 0.938 \| 0.030 \| 0.088 | 0.891 \| 0.037 \| 0.115 | 0.844 \| 0.038 \| 0.117 |
| tilted 50° | 0.851 \| 0.040 \| 0.134 | 0.814 \| 0.040 \| 0.122 | 0.712 \| 0.041 \| 0.119 |
| corridor, parallel surfaces | 0.745 \| 0.048 \| 0.159 | 0.582 \| 0.050 \| 0.153 | 0.173 \| 0.062 \| 0.210 |
| corridor, end wall (8 m) | 0.963 \| 0.045 \| 0.119 | 0.969 \| 0.037 \| 0.100 | 0.940 \| 0.036 \| 0.102 |

All 13 gates PASS (with the occlusion fixture's interior: 0.859 | 0.039 | 0.121). Controls, each fails:
scrambled frame b (median 9.6 px); a's eye rolled 1° (median 0.159, p90 0.49 px); shift-only matcher on the
parallel surfaces at B 0.49 m (median 0.096, p90 0.257 px; 0.492 of anchors within 0.25 px against 0.568 with stretch).
Not gated, reported: the radial-edge wall and the log-periodic wall give **no** accepted match (rejected as low texture,
ambiguous or unresolved); the occlusion fixture's edges (within 3 px of the board's outline) keep 0.40 accepted with
6.5 % of them over 1 px (interior 0.1 %).

A first reading: the matcher reaches 0.03–0.06 px median on textured synthetic surfaces, below the 0.1 px the oracle
ladder asks for; long baselines on surfaces parallel to the heading are where it runs out (17 % accepted at 1 m), as
the A1 amendment expected. Whether the renders' noise and texture allow the same is the calibration room's question
(Stages 4–5).

### Real renders: first calibration-room pairs (M3a Stages 4–5 under way, Chat sandbox)

**Protocol** (`33bfcad`). `flow-depth --correspondence rgb` runs the matcher on observation fields only and hashes its
inputs; `flow-depth-calibrate` opens truth for the declared calibration set only (pairs with even ka), sets the interval
scale to the 95th percentile of |d − d_true| / σ_d over accepted **interior** anchors / 1.96, and writes `freeze.json`
(settings, scale, hashes of every estimation output); `flow-depth-evaluate` verifies the freeze, writes the PLYs without
truth, then opens the held-out pairs (odd ka) for M3.COV (≥ 0.90 coverage of nominal 95 % intervals; control: intervals
× 0.5), M3.CAL (#5's calibration gate at stride 5; control: baseline × 1.2) and M3.PLY. Anchors within 2 px of an
object boundary in b are **edge** and evaluated apart: Position averages a pixel's samples, so its truth is not a
surface point there (#5's failure-mode table). The first evaluation code omitted that; it is now in.

**What the first pairs showed** (pair (0, 5), calibration set):

1. **Foreground fattening at the cards' edges.** 14 % of accepted matches were > 1 px off, in bands just outside the
   Siemens-star cards: a patch straddling a card edge is dragged to the card's motion. Two tests followed, both checked
   on this pair before being kept: *half windows* (the inner and outer radial halves, fitted alone with κ held, must be
   informative, σ ≤ 0.25 px, and agree within 0.5 px) and *depth edge* (no candidate within ± 3 px may deviate from the
   anchor's d + κ u by more than 0.5 px). With both, the interval scale is 1.66 (2.46 without), usable interior range
   is about the same (0.066 against 0.062 of all hits), and usable edge points over 50 % wrong fall from 11 % to 7 %.
2. **91 % of the remaining > 1 px errors lie within 2 px of an object boundary**, where the truth itself is unreliable.
   Interior accepted matches: median 0.069 px, p90 0.20 px, 0.9 % over 1 px.

Fixtures rerun with the two tests (`33bfcad`, clean, 687 s, [`fixtures_33bfcad.json`](runs/issue-6/fixtures_33bfcad.json)):
all 13 gates PASS, every control fails; acceptance 3–8 points lower; occlusion edges now 1.1 % over 1 px of accepted
(was 6.5 %).

**Smoke evaluation** (pairs (0, 5), (1, 6) calibrating, (3, 8) held out; [`smoke_evaluation.json`](runs/issue-6/smoke_evaluation.json)).
Both smoke runs opened truth of a pair meant for the held-out set, (1, 6) and then (3, 8); both are therefore moved to
the calibration set for the full run. Scale 1.68. M3.COV PASS (0.943, control 0.730); M3.PLY PASS; **M3.CAL FAIL on
coverage**: of the textured, jointly visible, interior 5–40° hits only 0.113 are usable (gate ≥ 0.50), with median range
error 1.8 % and p90 5.2 % (gate ≤ 5 %, ≤ 10 %). Field usable 0.108, against 0.114 for pixel-rounded oracle matches (O2)
and 0.846 for 0.1 px oracle matches (O3) on the same pair: **the matcher is precise where it accepts, but it accepts
about a quarter to a third of the anchors**, mainly rejected as low NCC (plain checker walls under 64 spp noise) and
ambiguous (the Siemens stars' spokes look periodic along a radial search line).

**Frozen (M-D10).** The estimator is frozen at the commit that adds this paragraph: `motion/matcher.py` (its
`SETTINGS`), `motion/logpolar.py`, `motion/flow_depth.py` unchanged since `33bfcad`, whose fixture gates hold. Added for
the hand-off: `flow-depth-calibrate --from-result` (imports a frozen scale; opens no truth of the new result),
`flow-depth-evaluate --gate classroom` (#5's Classroom thresholds: median ≤ 10 %, p90 ≤ 25 %, > 50 % errors ≤ 5 % of
accepted, ≥ 25 % coverage), `flow-depth-preview` (cloud views). The smoke evaluation of (5, 10) in testing them is the
third opened held-out pair; it too moves to the calibration set. The full runs are #7's.

**Process figures** (`flow-depth-figures`, [`motion/flow_figures.py`](../motion/flow_figures.py)), per pair:
`overview` (native looks with the focus of expansion and eccentricity rings, the decision per anchor, the displacement,
range and interval width on b's raster, the evaluation error map, the cloud over the oracle reference), `matching` (the
unwrapped (q, φ) images, |J_b − J_a| before and after moving J_a by the estimated d, matching curves of a few anchors with
estimate, interval and the evaluation truth) and `slit_scan` (one azimuth over the whole sequence, with the traces the
pair's two-frame depths predict for every frame, and the multi-frame relation cot e = (z_b − s) / L). Truth overlays are
drawn dashed and marked EVALUATION; `--no-truth` omits them. An unfrozen result needs `--scale`, shown as provisional.

**Flat-patch guard (#7, `7bb5e70`; Luiz's OK on #7, freeze moved there).** The frozen matcher raised `LinAlgError` on
Classroom pair (20, 25): an anchor whose look-a patch lay on the ceiling lamps' glass (constant luminance 2.00018) gave
a gain of 5e14 and a rank-1 system that stopped the whole batch (Code's diagnosis). A look-a patch flatter than 1e-9 is
now rejected (`gn_failed`), and a singular batch is solved row by row. Against `1a1ffc9` the outputs are bit for bit
identical on calibration-room pair (0, 5) and the front and corridor fixtures (Chat sandbox,
[`identity_check.txt`](runs/issue-7/fix/identity_check.txt)). On the Classroom, which has flat emitters, they are not
quite: at pair (0, 5) 3 of 48,752 anchors move from accepted to `half_disagree` through the guard in `_half`, every
other array identical (Code, workstation, `fix_effect.txt` on `experiment/m3b-flow-depth-classroom`).

**Render noise limits the Classroom (#7, Code, workstation, OPTIX; `docs/runs/issue-7/noise/` on
`experiment/m3b-flow-depth-classroom`).** At 64 spp the per-pixel noise of a Classroom look is σ ≈ 0.04 RMS, about 20–30 %
of the pixel's luminance at every contrast; median local contrast over noise is 0.97, and only a quarter of the pixels
reach the c / σ ≥ 2 that NCC ≥ 0.8 needs (estimated). At 1024 spp σ falls by 3.97 (1/√spp: 4). The frozen matcher on pair
(20, 25) with 1024-spp looks: accepted 0.26 (0.08 at 64 spp), usable 0.21 of anchors (0.04), interior median | p90 range
error 0.019 | 0.057 (0.038 | 0.119); `low_ncc` 13 % (48 %), `ambiguous` 25 % (11 %): once the noise is gone, the board's
weak or repetitive texture is the limit. `dolly-1` used one seed for every look, so its noise repeats across frames
(residual correlation 0.158 between looks 20 and 25; 0.006 with different seeds). `run --seed-base N` now gives each look
its own seed (N + k), recorded in `run.json`; without it, runs are as before.

### M3 final results (M-D12: 1024 spp, a seed per look; estimator frozen at `7bb5e70`; #7)

Code, workstation: renders on OPTIX, everything else on the CPU; every record computed at `3938239` (clean), fov-render
`0f6fa7d`. Records: [`docs/runs/issue-7/final/`](runs/issue-7/final/) (`b73c860`). The interval scale, calibrated on
the calibration room's declared set (29 pairs, 280,293 accepted interior anchors), is **1.625** (1.67 / 1.58 / 1.64 at
strides 1 / 5 / 10 alone); the Classroom imports it unchanged and opens no truth before its freeze.

**Gates** (held-out pairs; the Classroom without the three quick-look pairs):

| gate, stride 5 | calibration room (21 pairs) | Classroom (122 pairs) |
|---|---|---|
| M3.COV, 95 % intervals cover the truth (≥ 0.90) | **PASS** 0.949 (control 0.787) | **PASS** 0.926 (control 0.712) |
| M3.PLY | **PASS** (205,614 points) | **PASS** (424,490 points) |
| median range error | 0.9 % (≤ 5 %) | 1.8 % (≤ 10 %) |
| p90 range error | 2.8 % (≤ 10 %) | 5.1 % (≤ 25 %) |
| > 50 % errors, of accepted | 0.3 % | 0.04 % (≤ 5 %) |
| **coverage** of textured, jointly visible, interior 5–40° hits | **0.332** (≥ 0.50) | **0.116** (≥ 0.25) |
| verdict | M3.CAL **FAIL**, on coverage only | M3.ROOM **FAIL**, on coverage only |

Accepted held-out interior displacement error: median 0.039 / 0.076 px, p90 0.137 / 0.218 px (calibration room / Classroom).

**Field usable** (interval ≤ 10 %, of all hits) against the oracle ladder on the same pairs:

| stride | room: RGB | O2 | O3 0.1 px | Classroom: RGB | O2 | O3 0.1 px |
|---|---|---|---|---|---|---|
| 1 | 0.048 | 0.000 | 0.022 | 0.017 | 0.000 | 0.007 |
| 5 | 0.243 | 0.204 | 0.829 | 0.203 | 0.097 | 0.874 |
| 10 | 0.190 | 0.537 | 0.830 | 0.205 | 0.608 | 0.910 |

**Reading.**

1. **Depth from the image's own forward motion works on the foveated sensor where the matcher accepts.** At 0.5 m the
   held-out range error is about 1–2 % median and 3–5 % p90, the intervals are honest (0.93–0.95 coverage, with the
   scale transferred from one scene to the other), and the slit scans show one pair's depths predicting the image streaks
   over all 47 frames.
2. **It fails on coverage, in both scenes**: a fifth to a quarter of the field gets a usable range, against 0.83–0.91
   for oracle matches of 0.1 px. The losses are ambiguity (periodic or sparse texture: the Siemens stars' spokes, the
   checkers, the board's chalk), weak texture (walls, ceiling, the board's interior) and the conservative edge tests.
   At stride 5 the matcher beats pixel rounding (0.24 / 0.20 against O2's 0.20 / 0.10); at stride 10 it stays far below
   it (0.19 / 0.21 against 0.54 / 0.61): two-frame matching over a long baseline is the weak path (stretch, occlusion,
   a long and ambiguous search), as the A1 fixtures predicted.
3. **Rendering noise mattered more than any matcher setting.** 64 spp with one seed (σ 20–30 % of the signal, repeated
   across frames) left 0.04 of Classroom pair (20, 25)'s anchors usable (0.08 accepted); 1024 spp gave 0.21 usable
   (0.26 accepted), and with a seed per look the stride-5 pairs accept 0.22–0.29.
4. **Failure modes seen where #5 listed them**: the +X wall's window panes (transparency and reflections) hold most of
   the Classroom's large errors; card edges in the calibration room (foreground fattening, now limited by the edge tests,
   and Position truth that is not a surface point within 2 px of a boundary).

The natural next step is temporal integration (M4): chaining stride-1 matches and fitting cot e = a + b s over many
frames, which averages noise, keeps stretch small and lets consistency over the sequence resolve the ambiguous matches
that two frames cannot.
