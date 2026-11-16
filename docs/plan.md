# deep-motion: depth from motion along curved paths

The plan, as decided by Luiz in Chat on 2026-11-16. Its editable original is a Claude doc,
"M6 reformulated: depth from motion along curved paths"; this file is the repository's record (rule 3). Decisions
taken later are added under "Decisions", numbered P-D1, P-D2, …

## Context

M6 asks for depth from motion along a smooth curved path. M6a's tests (visgraf/foveal-motion #21, #22; Chat sandbox,
calibration room; records in `docs/runs/issue-22/`) showed that the geometry gains, but extending M4's matcher does not
deliver the gain.

- **T0 (geometry).** The law about each frame's own epipole matches the exact projection to about 1e-12, and reduces to
  M4 when the epipole is on the axis. A both5 window on an arc spreads its epipoles over ±W·s/(2R), whatever the
  sensor's orientation.
- **T1 (pure rotation).** Derotation with the recorded poses is exact (1e-11°). It cannot remove the resolution cost:
  the reference's central degree, seen by a frame turned 30°, has six times the noise floor's residual.
- **T2 (crab dolly, epipole off the fovea).** M6's matcher equals M4 on M4's own run, decision for decision. The oracle
  shows the geometry's gain: the fovea becomes fully usable at 0.5 px once the epipole is 10° off, against 0.004 on
  the dolly.
- **T2 (RGB).** The matcher became more accurate as the epipole moved off (median error 0.61 to 0.30 %) but accepted
  less (field usable 0.34 to 0.27). The loss sits where the motion crosses M4's radial windows.
- **Option A (rectification about the epipole).** Accepted 0.302 against 0.320 at 6× the cost. It did not cure the
  loss.

The brainstorm that followed concluded that extending M4's representation is the wrong road. M6 needs a representation
built on the nature of the motion itself.

## Principles

**What transfers from biology, and what doesn't.** Marr's three levels keep the domains apart. The computational
theory (multi-view geometry) transfers completely: it is physics. Representations and algorithms transfer only as
hypotheses to test. Implementation does not transfer. The log-polar retina is an anatomical fact. Matching in
log-polar coordinates was a computational choice that suited forward motion; T2 showed it doesn't carry over once the
gaze is free.

**The ledger.** Every borrowing from biology is recorded with its claimed computational benefit and the evidence for
it.

| Borrowed feature | Claimed benefit | Status |
| --- | --- | --- |
| Foveation and log-polar coordinates under forward motion | Uniform radial flow, so cheap, well-shaped matching | Shown (M3, M4) |
| Fovea at the focus of expansion | None: the worst place for depth | Refuted (T0, T2 oracle) |
| Gaze decoupled from heading | A large depth gain in the fovea | Geometry yes (T2 oracle), matcher no (T2 RGB) |
| Fixation during motion | Fovea matched against fovea, epipole far away | Predicted only (T0 estimate) |
| Different time scales across the field | Sampling rate matched to each region's parallax rate | Proposed (this brainstorm) |

**Avoid correspondence rather than solve it.** Biology samples densely in time, fixates and tracks, and works sparsely
and by task. Discrete looks 10 cm apart turn correspondence into a search that a continuously sampling eye mostly
never faces.

**Time scales.** For a point 3 m away, 20° from the epipole, a 10 cm step moves it about 13 px in the fovea and 1.5 px
at 30° eccentricity (estimates). A 1 cm step moves it about 1.3 and 0.15 px. Regions with different parallax rates
want different sampling rates in time, which is what the eye does.

**Caution: representation is necessary but not sufficient.** Option A was the straight-line instance of the
formulation below, and it did not rescue T2's RGB. Real matching stays the hardest problem. As in the project's
binocular work, the formulation is built and judged with an oracle (perfect matcher) first, and real matching comes
last.

## The formulation

**Choose coordinates in which the motion is a shift.** Log-polar worked for M4 because forward translation, seen from
a camera on the heading, becomes a uniform shift in q = ln tan e for each axial inverse depth. That gave both the 1-D
search (one hypothesis, one shift) and the temporal reading (slit-scan traces). The generalisation is to adapt the
coordinates to the symmetry of the motion.

**Primitives.** A path of constant curvature κ = 1/R, travelled at constant speed with the head locked to the
tangent, is the orbit of a one-parameter group of rigid motions. One family covers three primitives:

| Primitive | κ | Group | Canonical coordinates | Depth invariant | Blind set |
| --- | --- | --- | --- | --- | --- |
| Line | 0 | translation | epipolar-plane pencil about the path line: plane angle ω, u = cot ψ | D, distance from the path line | the path line (focus of expansion and its opposite) |
| Arc | between 0 and ∞ | rotation about an axis offset by R | concentric mosaics: arc position θ, slit angle β to the tangent | ρ, distance from the turn axis | none for range (stage 3): the turn axis is stationary in the head, but every other range along its ray moves |
| Pivot | ∞ | rotation about the eye | azimuth relative to the heading | none: no parallax | everything |

**What the coordinates give.**

- *Line:* after moving t, u′ = u − t/D. A point's trace is a straight line in (t, u) with slope 1/D, for any gaze; the
  gaze only selects which rays the sensor covers.
- *Arc:* the rays at angle β to the tangent are all tangent to the circle of radius R cos β. A point at radius ρ
  appears in slit β at θ_P + f(β, ρ). Between two slits the disparity is a uniform shift in θ that depends on ρ alone.
- *Both:* the image sequence is stationary in time. Points with the same invariant leave traces that are time-shifts
  of each other. A depth hypothesis is one trace kernel for the whole field, and matching is a 1-D search over the
  invariant.
- *Trace-scan:* the slit-scan generalises to sampling the space-time volume along the trace a hypothesis predicts. The
  right hypothesis is the one along which appearance stays constant.

**General paths.** A general path is approximated by a sequence of (κ, length) pieces, joined by clothoids later if
curvature continuity matters. With known poses, a hypothesis's trace is exact on any path. The symmetry buys
uniformity and legibility, not correctness. The integration window lives inside one piece, which reconciles the action
with the representation.

**Prior art to check.** Epipolar-plane image analysis (Bolles, Baker, Marimont), concentric mosaics (Shum, He), stereo
panoramas from one moving camera (Peleg, Ben-Ezra), crossed-slits projections (Zomet, Feldman, Peleg, Weinshall). As
far as we know, none treats depth from motion along composed paths in this way; this needs a literature search before
it is claimed.

## The sensor

**Any sensor design is acceptable, as long as it does the job.** The foveated eye was an inspiration, not a
requirement. In the synthetic realm every design is just a choice of rays to trace, including non-uniform sampling in
space and in time.

**Candidate: a heading-locked slit fan.** A fan of vertical slits at angles β to the path tangent, laid out in the
canonical coordinates of the current primitive. It is locked to the heading, not to the gaze.

- **Time sampling per slit.** Lateral slits (β near ±90°) see the fastest parallax, about 0.2° per cm at 3 m, so they
  want centimetre steps. Slits near the heading see almost none and can be sampled every 10 cm or more.
- **Resolution by usefulness.** Lateral slits carry the most depth per metre travelled.
- **The gaze sensor stays separate.** A foveated eye can remain for attention and recognition, reading depth from a
  shared map. Remapping the foveated sensor's periphery to canonical coordinates is a middle option.

Rendering a slit fan needs rays outside fov-render's current sensors: either a new fov-render sensor (an issue there)
or a small tracer of our own.

## Decisions

All taken by Luiz in Chat on 2026-11-16.

1. **Problem.** Depth from motion along a curved path stays M6's focus. Binocular vision, transitions between regimes
   and the action policy are relevant but deferred.
2. **Representation.** Coordinates adapted to the motion's symmetry, giving 1-D search and direct temporal analysis
   (trace-scans).
3. **Paths.** Compositions of primitives: lines, arcs and pivots, parametrised by curvature. General paths are
   approximated by them.
4. **Sensor.** Any design that does the job, including non-uniform sampling in space and time.
5. **Quantities.** Estimate and integrate in the primitive's invariant (D on a line, ρ on an arc). Report, fuse and
   gate in world 3D and range along the ray, for continuity with M4.
6. **Order.** Formulation first, judged with an oracle (perfect matcher). Real matching comes last, with Luiz's ideas
   for it still to be discussed.
7. **No T2c.** M4a-style fixtures under the crab geometry are dropped.
8. **Where.** The work continues in this repository, visgraf/deep-motion, forked from foveal-motion at `08db383`
   (foveal-motion's M-D19).

Later decisions, numbered P-D1, P-D2, …:

- **P-D1** (Luiz, 2026-11-16, in Chat; #4). The slit fan is rendered on fov-render's existing Panorama sensor.
  Columns are slits, and per-slit time sampling is emulated by decimating dense looks. A `SlitFan` sensor in
  fov-render only if cost bites.
- **P-D2** (Luiz, 2026-11-16, in Chat; #4). Stage 5 is split. 5a (Chat) builds the oracle evaluation on short windows
  in the sandbox: a line, an arc and a pivot. 5b (Code) runs the full windows and a composed path at the design
  resolution on the GPU.
- **P-D3** (Luiz, 2026-11-16, in Chat; #5). 5b is split in two. **5b.1** (Code, #6, now): 5a's line, arc and pivot at
  the design resolution (0.1°/px), 1 m windows at 1 cm steps, with 5a's commands. **5b.2** (#7): Chat first builds
  and tests the composed path (line, arc, line; the reference look sliding along it) and the fusion into world 3D,
  gated in range, at 0.5°/px in the calibration room; then Code runs it at the design resolution in the calibration
  room and the Classroom.

## The plan

Each stage ships checks with controls that must fail, as in M3 to M6a. Chat does the numpy and calibration-room work;
Code does the GPU and Classroom work.

1. **Close M6a.** Done in foveal-motion: M-D19, PR #24, merged at `08db383`.
2. **Geometry of the primitives** (numpy, no renders). Canonical coordinates for line, arc and pivot; trace
   prediction; invariants; blind sets. Checks: closed forms equal the exact projection; same-invariant traces are
   time-shifts; on a line the oracle reproduces M4's. Control: a wrong radius or a wrong path line. (Chat)
3. **Information atlas per primitive** (numpy). Precision of the invariant against slit angle, distance and time step.
   Output: the time-sampling rate each slit needs. (Chat)
4. **The sensor.** Design the heading-locked slit fan and its rendering path. Check, as in T1: every rendered ray is
   the designed ray. (Chat on the calibration room; Code for heavy renders)
5. **Oracle evaluation on renders.** Perfect traces from Position: coverage and precision of the invariant for a line,
   an arc, a pivot, and a composed line, arc, line path; fused into world 3D; gated in range. (Chat on the calibration
   room)
6. **Classroom.** A composed tour on the GPU, judged by the same oracle evaluation. (Code)
7. **Real matching.** Last, starting from Luiz's ideas. Trace-scans on dense time samples are the first candidate.

## Deferred topics and open questions

**Deferred, but recorded.**

- **Regimes and their transitions.** Binocular (lateral baseline, simultaneous), lateral sway (lateral, sequential)
  and forward motion (axial, growing) are one geometry with different baselines. Transitions can be read as weighting
  by information rather than switching. A binocular eye fills forward motion's blind spot at the fovea.
- **The action policy.** Choosing to verge, peer, walk or hold, driven by an uncertainty map.
- **Track and filter.** Pairs matched with exact geometry, folded into a per-point inverse-depth filter: causal, with
  fovea and periphery on different time scales.
- **Beyond planar lines and arcs.** Clothoid joints; helices (screw motions) for paths in 3D.
- **Far field and near field.** Priors, size and texture beyond parallax; accommodation and time-to-contact up close.

**Open questions.**

- [x] **Where the work lives.** This repository (decision 8).
- [ ] **Rendering the slit fan.** A new sensor in fov-render, or a small tracer of our own.
- [ ] **Planar paths first?** Lines, arcs and pivots in a horizontal plane before anything in 3D.
- [ ] **Literature search** on concentric mosaics, crossed-slits and EPI analysis for depth along composed paths,
  before any claim of novelty.

## Results

### Stage 2: the geometry of the primitives (#2; Chat sandbox)

`python -m deep selftest --out docs/runs/issue-2` at `0b04866` (clean), numpy 2.2.6, Python 3.13, about 1 s. Record:
[`stage2_selftest.json`](runs/issue-2/stage2_selftest.json). Module: [`deep/primitives.py`](../deep/primitives.py).
8 checks PASS; each of the 8 controls fails as it must:

| check | result | control (must fail) | control result |
|---|---|---|---|
| head axes are `motion.paths.head_axes` | 1.1e-16 | right flipped | ≥ 1.42 |
| line: u = (x_P − s)/D, ω constant, against exact projection | 6.8e-14 | path line turned 1° | median ≥ 0.046 |
| arc: slit β sees P at θ = φ_P − δ(β, ρ), v = dz cos β /(ρ sin δ), against exact projection (2,800 sightings, both branches) | 3.1e-15 | radius 2 % wrong | ≥ 2.1e-4 rad |
| same orbit, same trace shifted in time (line along f, arc about the axis), exact projection | 7.8e-16 | the moved point 5 % farther | median ≥ 4.1e-3 |
| blind sets: the path line and the turn axis keep their head direction | 2.2e-16 | 1 cm off | ≥ 3.2e-4 |
| pivot: rotation alone, at any depth | 3.3e-16 | pivot 5 cm behind the eye | ≥ 3.8e-3 |
| line with the gaze along it: −ln(u(s+t)/u(s)) is M4's d_k(ζ) | 3.6e-15 | head 1° off the path line | median ≥ 2.6e-4 |
| composition (line, arc, pivot, line) joins continuously | 0 | an arc 1 % short | 0.015 m gap |

Figure ([`stage2_primitives.png`](runs/issue-2/figures/stage2_primitives.png), `python -m deep figures`):
- **Line.** Traces in (u, s) are straight, with slope −1/D.
- **Arc.** θ − φ_P against β: one curve per ρ, the same for every point at that radius.
- **A composed path** from above, with the turn axis marked.

**An observation for stage 3.** On an arc, the traces of all radii converge as the slit turns sideways (β → ±90°).
For R = 1.6 m, the crossing angle's sensitivity to ρ at ρ = 3.1 m falls with cos β:

| β | 0° | 30° | 60° | 80° | 89° |
|---|---|---|---|---|---|
| ∂δ/∂ρ (deg per m) | 11.1 | 9.2 | 4.9 | 1.7 | 0.17 |

The sideways (radial) directions point at the turn axis, the arc's blind set. On an arc, the blind directions are
therefore the lateral ones, as the heading is on a line. The plan's "lateral slits carry the most depth per metre"
holds for lines; for arcs, the slits near the tangent carry the depth. Stage 3's atlas measures this properly, before
the sensor is designed.

**Correction (stage 3).** The observation above is wrong. It measured the sensitivity of a slit's *absolute* crossing
angle, which depends on the point's unknown angle φ_P. What carries range is the difference between slits, and at
β → ±90° the curves converge with different slopes (dθ/dβ = −(1 + R/ρ)). Stage 3's Fisher atlas shows that on an arc
the lateral directions are the most precise, as on a line, and that no direction is blind.

### Stage 3: the information atlas (#3; Chat sandbox)

`python -m deep selftest --stage 3 --out docs/runs/issue-3` at `ea67d0e` (clean). Records:
[`stage3_selftest.json`](runs/issue-3/stage3_selftest.json), [`stage3_atlas_summary.json`](runs/issue-3/stage3_atlas_summary.json).
Module: [`deep/atlas.py`](../deep/atlas.py). 6 checks PASS; each of the 6 controls fails as it must.

**The model** (a design model, not a measurement):
- an anchor ray at the window's centre, with the point on it at eye height;
- its bearing measured every 1 cm over a 1 m window, with 0.1° noise (a proposed angular resolution), the anchor's own
  bearing exact;
- known poses, so the arc's rotation is predicted;
- σ_r is Fisher's bound on range along the ray.

| check | result | control (must fail) | control result |
|---|---|---|---|
| analytic d b/d r equals the numerical derivative | 4.2e-8 of its median size | steps 2 % longer | 1.9e-2 |
| line, 10 steps of 0.1 m: equals T0's Fisher bound (`motion/derotate.py`) | 2.9e-15 relative | T0's bound with steps 20 % longer | median 21 % |
| an arc with R = 1e7 m is the line | 1.0e-7 relative | R = 20 m | 5.2 % |
| line: no information on the path line | I = 1.9e-33 | 1 cm off | I = 1.4e-6 |
| arcs: no blind cell on the grid; the turn axis's range carries information | 0 blind cells; I = 1.28 | the line's grid | 60 blind cells |
| time step: the parallax spread is \|sin ψ\| (1/r_lo − 1/r_hi) | 3.5e-8 relative | arcs with the rotation left in | ≥ 10 % off |

Three checks were corrected on their first run. No threshold moved, and the reasons are in the code and the commit:
- Check 1's error was first measured relative to the derivative, which is undefined where the derivative crosses zero.
  It is now measured against the derivative's median size.
- Check 1's control, a constant yaw, cannot change d b/d r. It now lengthens the steps by 2 %.
- Check 4 declared the turn axis blind (I ≤ 1e-20) and **failed at I = 1.28**. That is how the arc's lack of a blind
  direction was found. The check now states what is true, and the line's blind cells are its control.

**σ_r / r at 3 m, by bearing from the heading** (1 m window, 1 cm steps, 0.1°):

| bearing | 1° | 10° | 30° | 90° | 150° | 179° |
|---|---|---|---|---|---|---|
| line | 0.097 | 0.0097 | 0.0035 | 0.0019 | 0.0035 | 0.097 |
| arc R = 4 m | 0.038 | 0.011 | 0.0036 | 0.0018 | 0.0036 | 0.038 |
| arc R = 1.6 m | 0.015 | 0.011 | 0.0037 | 0.0018 | 0.0037 | 0.015 |

Exactly on the heading, the line is blind; the arcs give 0.034 (R = 4 m) and 0.014 (R = 1.6 m).

**The largest time step** that keeps the unpredicted parallax within one resolution element (0.1°), in cm:

| angle from travel ψ | 5° | 10° | 30° | 60° | 90° |
|---|---|---|---|---|---|
| prior band, 0.5–20 m | 1.0 | 0.52 | 0.18 | 0.10 | 0.09 |
| converged ±10 % at 3 m | 30 | 15 | 5.2 | 3.0 | 2.6 |

Figures: [`stage3_atlas.png`](runs/issue-3/figures/stage3_atlas.png) (precision around the head, polar),
[`stage3_profiles.png`](runs/issue-3/figures/stage3_profiles.png) (the 3 m profile and the time steps).

**Reading:**
- **Arcs have no blind direction.** Curving the path fills the line's blind cone at the heading: tighter turns fill
  it more. Elsewhere, lines and arcs agree within about 10 % (at 10° from the heading the arcs are slightly worse). Lateral directions are the most precise
  on both, at about 0.2 % at 3 m.
- **Time sampling should be driven by uncertainty, not fixed.** Before a point's range is known, lateral rays need
  sub-millimetre steps to stay within a resolution element. Once range is within ±10 %, 2.6 cm suffices laterally and
  30 cm near the heading. Sampling can start fine and coarsen as each point converges: coarse-to-fine in time,
  per direction and per point.
- **For the sensor (stage 4).** Lateral slits deserve the finest angular resolution and the densest time sampling. On
  arcs, slits near the heading also carry range, which a line never gives. M4's windows of 10 cm steps sit far
  outside the tracking regime for lateral rays: one 10 cm step moves a lateral point at 3 m by about 1.9°, about
  19 resolution elements.

### Stage 4: the sensor, a slit fan on fov-render's Panorama (#4; Chat sandbox)

**Rendering path** (proposed on #4; a decision for Luiz, reversible). fov-render's existing **Panorama** sensor
(head-locked, equirectangular) carries the slit fan, so fov-render needs no change.
- With the head locked to the tangent, each column is a vertical slit at bearing −lon from the heading, and each row an
  elevation. That is exactly the arc's concentric-mosaic (β, v = tan lat); the line's pencil coordinates come from the
  same rays.
- Per-slit time sampling is emulated by rendering dense looks and keeping each slit's samples at its own rate.
- A `SlitFan` sensor in fov-render (rendering only the needed columns, per look) is worth it only if cost bites.
- Cost (Chat sandbox, calibration room, width 720 = 0.5°/px): 10.1–10.3 s per look at 16 spp; 39 s at 64 spp in a
  first, untracked trial (one look, not in the repo). The design resolution of 0.1° (width 3600) is Code's, on the GPU.

**The schedule is solved exactly.** Stage 3's ds_max is the first-order step. Over a finite step a near point gets
closer and its parallax grows. S4.3's first run caught this: at 10° from travel, a point at 0.5 m overshot one pixel
by 5.5 %. `SlitFan.step` now solves the straight step exactly by bisection on the bearings of the band's two ends.

**Run** (`runs/stage4-arc`, record [`run.json`](runs/issue-4/run.json)): `python -m deep stage4-render --run
runs/stage4-arc` at `e7d821a` (clean). Three looks on a left-turning arc (R = 1.6 m) from the calibration room's
EYE, heading +Y, at s = 0, 1 and 10 cm; width 720, 16 spp, seeds 9000 + k, CPU, 31 s.

**Checks** (`python -m deep selftest --stage 4 --run runs/stage4-arc` at `e7d821a`, clean; record
[`stage4_selftest.json`](runs/issue-4/stage4_selftest.json)). 3 checks PASS; each control fails as it must:

| check | result | control (must fail) | control result |
|---|---|---|---|
| S4.1 the carrier's pixel rays have the designed slit coordinates (bearing = −lon, v = tan lat) | 5.9e-16 | bearing sign flipped | median 1.61 rad |
| S4.2 every rendered ray is the designed ray R_head d_local; its bearing is its column's | 6.4e-15°, bearing 2.5e-14° | designed head yawed 0.5° | median 0.43° |
| S4.3 each slit at its scheduled step: unpredicted parallax within one pixel (line and arc, 7 slits, band 0.5–20 m) | ≤ 1.002 px | twice the step | ≥ 1.93 px |

**The schedule at 0.5°/px**, largest step per slit in cm, while 0.5–20 m uncertain / once ±10 % at 3 m (figure
[`stage4_slit_fan.png`](runs/issue-4/figures/stage4_slit_fan.png), at `efbc768`):

| angle from travel | 12° | 30° | 60° | 90° | 120° | 150° |
|---|---|---|---|---|---|---|
| uncertain, 0.5–20 m | 2.1 | 0.88 | 0.51 | 0.45 | 0.52 | 0.91 |
| converged, ±10 % at 3 m | 45 | 23 | 14 | 13 | 16 | 31 |

At the design resolution of 0.1° the steps are five times shorter (stage 3's table).

**Reading.** The sensor exists without new code in fov-render: a panorama is a slit fan whose columns can each be kept
at their own rate. What it does not give for free is cost: the dense looks that the uncertain phase needs (sub-cm
laterally) are many renders. That is the case for Code's GPU, and later perhaps for a `SlitFan` sensor that renders
only the slits due at each look.

### Stage 5a: the oracle on renders, short windows (#5; Chat sandbox)

**What it is.** [`deep/oracle.py`](../deep/oracle.py): a perfect matcher, as M4's O-M. Every hit of the reference look
(a window's middle) has its true point P from Position. P is projected into every other look and kept where that look
sees the same surface (M1.3's agreement, 2 px at the range). There its bearing and elevation are exact or carry
Gaussian noise of σ ∈ {0.25, 0.5, 1} px (0.5°/px). Range along the anchor ray is fitted by Gauss–Newton through the
known poses, with analytic derivatives. Usable: a 95 % interval within ±10 % (as M4).

**Runs** (calibration room, from the EYE heading +Y, Panorama 720 × 360, 16 spp, CPU; records in
[`runs/issue-5/runs/`](runs/issue-5/runs)):

| run | looks | window | rendered at | seconds |
|---|---|---|---|---|
| line20 (`s5a-line`) | 21 | 20 cm, 1 cm steps | `5ec7a6f` | 222 |
| arc20 (`s5a-arc`), R = 1.6 m, left | 21 | 20 cm, 1 cm steps | `5ec7a6f` | 225 |
| pivot (`s5a-pivot`) | 21 | 20°, 1° steps | `5ec7a6f` | 221 |
| line50 (`s5a-line50`), addendum | 21 | 50 cm, 2.5 cm steps | `43e6177` | 234 |
| arc50 (`s5a-arc50`), R = 1.6 m, addendum | 21 | 50 cm, 2.5 cm steps | `43e6177` | 236 |

Commands: `python -m deep stage5a-render --kind {line,arc,pivot} --run runs/s5a-K --seed-base 9200|9300|9400`, and for
the addendum `--ds 0.025 --seed-base 9500|9600`.

Three departures from #5, said here:
- The pivot has 21 looks over 20°, not 11 over 10°; `run_poses` shares the look count, and a longer pivot is the
  harder control.
- #5 named the sectors but not the angle they are measured in. The first report used bearing over all elevations,
  where floor and ceiling anchors straight ahead sit far from the heading in 3-D and hide the line's blind cone. The
  tables now give three ways: the 3-D angle ψ from the heading (primary), bearing at eye height (|lat| < 10°, stage 3's
  setting), and bearing over all elevations. The checks did not change.
- The 50 cm windows are an addendum, with tables only and no checks. The 20 cm windows cannot show the arc's heading
  gain; the atlas predicted that, and the addendum tests it.

**Checks** (`python -m deep stage5a --line runs/s5a-line --arc runs/s5a-arc --pivot runs/s5a-pivot --out
docs/runs/issue-5` at `60dd9cb`, clean; record [`stage5a.json`](runs/issue-5/stage5a.json)). 3 checks PASS; each control
fails as it must:

| check | result | control (must fail) | control result |
|---|---|---|---|
| O5.1 no noise: the fit recovers the true range on every anchor seen by another look | 1.4e-13 | the looks' centres off by 1 cm | median 2.8 % |
| O5.2 eye height, every look sees it: the oracle's Fisher equals stage 3's atlas (line and arc; 2232 and 2316 anchors) | median 9.4e-5 | the atlas over a window twice as long | median 100 % |
| O5.3 the pivot carries no range | I ≤ 9.8e-24 of the line's; none usable | the line is usable | 90.9 % |

**Usable fraction, %, by the 3-D angle from the heading** (`python -m deep stage5a-table line20=… arc20=… line50=…
arc50=… pivot=… --out docs/runs/issue-5` at `594c7cc`, clean; record [`stage5a_table.json`](runs/issue-5/stage5a_table.json)):

| run | σ | heading ψ < 15° | lateral 60–120° | behind ψ ≥ 165° | all |
|---|---|---|---|---|---|
| line20 | 0.25 px | 0 | 100 | 0 | 90.9 |
| | 0.5 px | 0 | 90.9 | 0 | 69.0 |
| | 1 px | 0 | 62.5 | 0 | 39.4 |
| arc20 | 0.25 / 0.5 / 1 px | 0 / 0 / 0 | 99.9 / 90.9 / 62.6 | 0 / 0 / 0 | 90.9 / 68.8 / 39.4 |
| line50 | 0.25 px | 27.8 | 100 | 59.1 | 98.3 |
| | 0.5 px | 16.6 | 100 | 0 | 93.5 |
| | 1 px | 0 | 99.8 | 0 | 78.3 |
| arc50 | 0.25 px | 16.2 | 100 | 63.2 | 98.2 |
| | 0.5 px | 5.8 | 100 | 0 | 93.4 |
| | 1 px | 0 | 98.8 | 0 | 78.0 |
| pivot | every rung | 0 | 0 | 0 | 0 |

At eye height (the walls, 1.7–5 m), the lateral sector is usable at 99.8 / 50.3 / 1.2 % (line20, 0.25 / 0.5 / 1 px)
and 100 / 99.9 / 98.9 % (line50); arcs within a point of the lines, except arc50 at 1 px (92.5 %). The usable anchors'
95 % intervals cover the truth 94.9–95.0 % of the time on every run and rung.

**Near the heading** (median 95 % half-interval / range at 0.25 px, by ψ; the profile in the same record and figure):

| ψ | 0–2° | 2–5° | 5–10° | 175–178° | 178–180° |
|---|---|---|---|---|---|
| median range, m | 4.8 | 4.8 | 4.8 | 2.7 | 2.7 |
| line20 / arc20 | 3.33 / 1.88 | 1.01 / 0.95 | 0.49 / 0.52 | 0.60 / 0.56 | 1.57 / 1.14 |
| line50 / arc50 | 1.14 / 0.44 | 0.39 / 0.30 | 0.18 / 0.19 | 0.25 / 0.19 | 0.65 / 0.25 |

Figures (at `60dd9cb`): [`stage5a_maps.png`](runs/issue-5/figures/stage5a_maps.png) (the half-interval over each
reference panorama, the ±10 % contour) and [`stage5a_profile.png`](runs/issue-5/figures/stage5a_profile.png) (its
median against ψ).

**Reading:**
- **The oracle works on renders and agrees with the atlas.** It recovers range exactly without noise, its Fisher bound
  is stage 3's to 1e-4, the pivot carries nothing, and its intervals are calibrated (95.0 % coverage).
- **The calibration room does not test visibility.** Over windows up to 50 cm, every anchor is seen by another look
  (≥ 99.999 %). Occlusion is the Classroom's question (5b, stage 6).
- **Window length and pixel noise set precision.** At 0.5°/px, 20 cm windows give the walls only at 0.25 px; 50 cm
  windows give the lateral walls at 1 px. Precision near the heading and behind grows fastest with the window.
- **The arc's heading gain is real on renders but small at these lengths.** At 20 cm, arc and line are the same
  everywhere but the innermost cone. At 50 cm the arc's median interval in the 2° cone is 2.6 times narrower than the line's
  (±44 % against ±114 %), and behind it is also better. The wall ahead is at 4.8 m, and neither reaches
  ±10 % there; the atlas puts the arc at ±10 % at 1.5 m with a 50 cm window and 0.25 px. Between 5° and 15° the arc is
  slightly worse than the line, as stage 3 found, so the ±15° sector counts fewer usable anchors on arc50 than on
  line50. The gain needs longer windows, nearer surfaces or finer pixels (the design's 0.1° is five times finer):
  that is 5b's question.

### Stage 5b.2, Chat part: the composed path and the fusion in world 3D (#7; Chat sandbox)

**What it is.** [`deep/fusion.py`](../deep/fusion.py). The design and the checks were declared on #7 before the code
was written.
- **Why not windows.** Sliding windows share looks. Merging their estimates would count the same measurement several
  times. So the oracle's fusion is a joint fit.
- **Cells.** Surface cells are 2 cm voxels of the true points.
- **The fit.** Each cell's 3-D point is fitted by Gauss–Newton to its bearing and elevation in every look that sees it
  (M1.3's agreement), each look counted once. A step that raises the cost is halved.
- **Covariance.** σ² × the pixel angle² × H⁻¹.
- **Usable.** The 95 % interval along the ray is within ±10 % of range from every look that sees the cell.

**Runs** (calibration room, from the EYE heading +Y, Panorama 720 × 360, 16 spp, CPU, 37 looks every 2.5 cm). Each
path is rendered as two runs (looks 0–18 and 19–36). Records are in [`runs/issue-7/runs/`](runs/issue-7/runs).
- **composed:** 30 cm line, then a 30 cm left arc (R = 1.6 m, 10.7°), then a 30 cm line. Rendered at `5d34b5f`,
  194 + 187 s.
- **straight:** a 90 cm line, for comparison. Rendered at `f5dfa76`, 191 + 181 s.

Commands: `python -m deep stage5b2-render [--path straight] --run runs/s5b2-{a,b,sa,sb} --first 0|19 --last 18|36
[--seed-base 11000]`.

**Checks** (`python -m deep stage5b2 runs/s5b2-a runs/s5b2-b --out docs/runs/issue-7` at `b8300e9`, clean; record
[`stage5b2.json`](runs/issue-7/stage5b2.json)). 3 checks PASS; each control fails as it must. The straight path passes
the same checks at `f5dfa76` ([`stage5b2_straight.json`](runs/issue-7/stage5b2_straight.json)). There, the F1 control
moves every look, since a straight path has no join.

| check | result | control (must fail) | control result |
|---|---|---|---|
| F1 no noise: the fused points are the true points (432 633 cells, 99.95 % seen by two looks) | 7.4e-14 of range | looks after the first join moved 1 cm | median 1.0e-2 |
| F2 the fused σ is honest: std of along-ray error / Fisher σ (30 draws, 0.5 px, 2000 usable cells) | 0.9993 | each look counted twice (the window-overlap mistake) | 1.413 |
| F3 the usable cells' 95 % intervals cover the truth (0.25, 0.5, 1 px) | 95.0, 94.9, 94.9 % | σ halved | ≤ 67.4 % |

**F1 failed on its first run, at `5d34b5f`** (record kept:
[`stage5b2_first_run_F1_fail.json`](runs/issue-7/stage5b2_first_run_F1_fail.json)).
- **The failure.** Two cells were off by up to 2.3e5 of range: one on the ceiling straight above the path, one on the
  floor straight below. Both are seen at up to 89.8° elevation, where lon is singular, and a full Gauss–Newton step
  threw them away.
- **The fix.** The fit now halves such steps. No threshold moved.
- **Also changed.** A dry run on 5a's line50 run (not a record) had found one such cell still moving after 15 steps,
  so the fit now iterates to convergence.

**Results** (from the same records):

| path | σ | usable cells | coverage | median / 95th-percentile 3-D error |
|---|---|---|---|---|
| composed | 0.25 px | 96.7 % | 95.0 % | 1.3 / 10.1 cm |
| | 0.5 px | 91.5 % | 94.9 % | 2.5 / 15.9 cm |
| | 1 px | 77.0 % | 94.9 % | 4.1 / 20.4 cm |
| straight | 0.25 / 0.5 / 1 px | 96.7 / 91.6 / 76.7 % | 95.0 / 95.0 / 95.0 % | 1.3 / 2.5 / 4.1 cm median |

**By bearing from the path's middle, relative to the start heading** (Fisher gate at 0.5 px;
[`stage5b2_cones.json`](runs/issue-7/figures/stage5b2_cones.json), at `db65ef7`):

| path | ahead < 15° | ahead 15–30° | lateral 60–120° | behind > 165° |
|---|---|---|---|---|
| composed | 65.6 % | 78.5 % | 100 % | 91.0 % |
| straight | 61.7 % | 82.2 % | 100 % | 90.7 % |

Figures (at `db65ef7`): [`stage5b2_topdown.png`](runs/issue-7/figures/stage5b2_topdown.png) (the cells from above,
usable or not, with each path) and [`stage5b2_progress.png`](runs/issue-7/figures/stage5b2_progress.png) (the usable
fraction as the path is travelled).

**Reading:**
- **Joint fusion is right, and window fusion would not be.** The fused σ is honest to 0.1 %. Counting each look twice
  (what merging overlapping windows does) makes σ too small by √2, and F2's control shows it at 1.413.
- **A 90 cm path in the calibration room fuses most of the room at 0.5°/px.** 91.5 % of the visible surface is usable
  at 0.5 px, and 77 % at 1 px.
- **A short arc changes almost nothing overall.** A 10.7° turn moves the totals by under half a point, and the
  progress curves lie on each other. It shifts precision between neighbouring directions: more ahead within 15° of the
  start heading (65.6 against 61.7 %), less at 15–30° (78.5 against 82.2 %). Stage 3 predicted that trade.
- **What is not usable off the heading is mostly what few looks see.** Off the heading, cells that fail are seen by a
  median of 13 of the 37 looks, against 37 for those that pass. If every look saw them, only 0.5 % would fail (a
  diagnostic, not in the records). They are surfaces at grazing incidence, where M1.3's 2 px agreement rejects most
  looks, or cells occluded behind the boards; this run does not separate the two.
- **Cost.** The checks take 5.7 min in the Chat sandbox, just over the ~5 min batch rule. On Code's machine, at
  3600 wide, the cells stay the same (2 cm voxels), but each look has 25× the samples to load and bin. Code may need to
  subsample the hits before binning, and should say how.
