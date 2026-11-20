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

**On the Riemann sphere** (Luiz's question, 2026-11-19, in Chat; checks M1–M4 with controls in
[`deep/moebius.py`](../deep/moebius.py), all passing at `c7d755d`, Chat sandbox,
[record](runs/issue-11/moebius_selftest.json)). Project the viewing sphere stereographically with the focus of
expansion at z = 0 and its opposite at ∞, and write ℓ = ln |z| = ln tan(ψ/2).

- *Rotations of the eye are Möbius maps:* SO(3) sits inside PSL(2, ℂ) as its unitary part, and a rotation preserves the
  cross-ratio of any four directions at any depths (M3). The pivot is an elliptic Möbius flow about the vertical, which
  is why it carries no depth.
- *Translation is not a Möbius map, but follows the hyperbolic ones.* The hyperbolic maps z ↦ kz fixing the focus of
  expansion and its opposite move every point along a meridian, the line's epipolar great circle, by the same shift
  in ℓ. A unit-speed translation moves each point along the same meridian at dℓ/dt = 1/r (M1). It is the hyperbolic
  generator weighted pointwise by inverse range, and depth is exactly what breaks the Möbius structure.
- *The line's coordinate is ℓ's sinh:* u = cot ψ = −sinh ℓ. A finite translation shifts sinh ℓ by t/D: the straight
  trace u′ = u − t/D (M2). M4's q = ln tan ψ ≈ ℓ + ln 2 near the focus of expansion, which is why log-tangent behaved
  as a shift in the fovea and less so towards 45°.
- *The arc is their sum:* at each instant, the turn's elliptic field (depth-free) plus the translation's 1/r-weighted
  hyperbolic field about the tangent (M4). Depth on an arc is a trace's departure from the turn-in-place diagonal.
- *Echoes to check in the literature:* PSL(2, ℂ) is the Lorentz group acting on the celestial sphere, where a boost is a
  hyperbolic Möbius map (aberration; Penrose 1959). Log-polar coordinates, w = log z, turn the hyperbolic and
  elliptic maps about the same fixed points into translations, which is the usual reading of Schwartz's retino-cortical
  log-polar map.

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

### Stage 5b.1: 5a's windows at the design resolution (#6; Code, RTX 4090)

**What Code ran.** 5a's commands with `--width 3600 --n 101 --ds 0.01 --device OPTIX`: 1 m windows at 1 cm steps, and
the pivot at 21 looks. Renders were at `b36d8c6`; the tables at `b36d8c6` (stride 25) and `6b0a87b` (stride 7).
Records, logs and timing are in [`runs/issue-6/`](runs/issue-6). The measured numbers are Code's, on the workstation;
the full report is on #6.

- **Cost.** About 12.2–12.8 s per look. Of that, 1.55 s is Cycles; the other ~10.6 s is fov-render writing the EXR
  (207 MB) and the record (183 MB). About 40 GB per 101-look run. The oracle evaluation peaks at 58–61 GB, because it
  holds every record at once.
- **Departures.**
  - `deep/figures.py` (`6b0a87b`): at a stride, each map pixel takes its nearest anchor along its row. The profile's
    pixel angle now comes from the run.
  - A stride-7 addendum. Stride 25 divides 3600, which puts the whole 2° cone on one column.

**Results** (σ = 0.25 / 0.5 / 1 px, at 0.1°/px; stride 7):

| run | ψ 0–2° usable | 2–5° | 5–15° | median 95 % half-interval, 0–2° / 178–180° (0.25 px) |
|---|---|---|---|---|
| line | 93.9 / 60.0 / 1.7 % | 100 / 100 / 76.5 % | 100 / 100 / 99.9 % | ±4.6 % / ±3.3 % |
| arc, R = 1.6 m | 100 / 100 / 100 % | 100 / 100 / 100 % | 100 / 99.4 / 97.4 % | ±1.1 % / ±0.7 % |
| pivot (stride 25) | 0 everywhere | | | |

Everywhere else, line and arc are usable at every rung. Coverage is 94.8–95.1 %. Every anchor is seen by another look.

**Chat's review.**
- The tables match the records.
- The arc's median in the 2° cone (±1.08 %, wall at 4.7 m) matches stage 3's atlas for a 1 m window at 0.025°:
  ±1.10–1.14 % at bearings 0.25–1.5° and 4.7 m (Chat sandbox, `deep/atlas.py`).
- Code's figure change is correct for a scene where every pixel is a hit, as its comment says.

**Reading.**
- With 1 m windows and 0.1° pixels, **the arc makes the heading usable**: ±1.1 % at 4.7 m in the 2° cone, against ±4.6 %
  on the line. In 5a it was ±44 % against ±114 %.
- Between 5° and 15° the arc remains slightly worse at 1 px (97.4 % against 99.9 %), as stage 3 predicted.
- **Two costs to cut for longer paths.** The EXR write is unneeded by the oracle. The oracle's evaluation holds every
  record at once: `deep/fusion.py` (5b.2) loads them one at a time; `deep/oracle.py` should too.
- **Done at `981f361`** (Chat):
  - `deep/oracle.py` loads records one at a time. Stage 5a's checks reproduce their record exactly.
  - `--hit-stride` bins a random share of each look's hits, so it no longer falls on fixed columns.
  - `--drop-exr` deletes each EXR once its record exists. That saves disk only: fov-render always writes the EXR,
    because it builds the record from it. Saving the write time would be an fov-render issue.
  - The map's colour floor follows the runs.
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

**Addendum: a 90° turn** (Luiz agreed on #7 to a real turn, 2026-11-16, in Chat).
- **The path.** 30 cm line, then a left arc of 90° at **R = 1 m**, then a 30 cm line. The turn was proposed at
  R = 1.6 m, but at 1.6 m the arc runs into a board of the calibration room: 0.8 cm from its surface points, against
  18 cm at 1 m.
- **The run.** `python -m deep stage5b2-render --path turn90 --step 0.1 --run runs/s5b2-turn --seed-base 12000` at
  `4892e22` (clean). 22 looks every 10 cm, ending at 2.1 m of the path's 2.17 m; 240 s. Record:
  [`runs/issue-7/runs/s5b2-turn`](runs/issue-7/runs/s5b2-turn).
- **What else changed.** `run.json` now records the path, its start and its step. Records are loaded one at a time.
  `--hit-stride` and `--max-cells` bound memory and time for Code. The refactored fusion reproduces the composed
  path's record exactly (cells, F1–F3, rungs, progress).

**Checks** (`python -m deep stage5b2 runs/s5b2-turn --out docs/runs/issue-7` at `4892e22`, clean; record
[`stage5b2_turn90.json`](runs/issue-7/stage5b2_turn90.json)). All pass: F1 3.2e-14 (control 4.2e-3); F2 1.0018
(control 1.417); F3 95.0 / 95.0 / 94.9 % (σ halved ≤ 67.3 %).

| path | length | looks | usable at 0.25 / 0.5 / 1 px | median 3-D error at 0.5 px |
|---|---|---|---|---|
| straight | 0.9 m | 37 | 96.7 / 91.6 / 76.7 % | 2.5 cm |
| turn90 | 2.1 m | 22 | 99.5 / 98.4 / 92.8 % | 1.6 cm |

By bearing from the path's middle, relative to the start heading, at 0.5 px
([`turn90/stage5b2_cones.json`](runs/issue-7/figures/turn90/stage5b2_cones.json)):
- turn90: ahead < 15° 98.6 %, 15–30° 98.0 %, behind 100 %;
- straight: 61.7 %, 82.2 % and 90.7 %.

On the turn, 92.8 % of the cells first become usable on the arc, 5.4 % on the first line and 0.3 % on the last. 1.5 %
never do. Figure: [`turn90/stage5b2_topdown.png`](runs/issue-7/figures/turn90/stage5b2_topdown.png).

**Reading:**
- **A real turn closes the line's blind cone and most of the grazing losses.** As the heading sweeps, the wall ahead
  and the boards seen edge-on from the straight path come into view from other directions.
- **Turning, not only length** (an equal-length straight path, added after the first reading). `straight_long` is a
  2.17 m line from 2 m further back (`--start 0,-2,90`, so it clears the box ahead), 22 looks every 10 cm, rendered at
  `ff7f8ee`; checks at `ff7f8ee`, all pass ([`stage5b2_straight_long.json`](runs/issue-7/stage5b2_straight_long.json)).
  The figures are in [`turn90_vs_long/`](runs/issue-7/figures/turn90_vs_long).

  | path | usable at 0.25 / 0.5 / 1 px | median 3-D error at 0.5 px | ahead < 15° | ahead 15–30° |
  |---|---|---|---|---|
  | turn90 | 99.5 / 98.4 / 92.8 % | 1.6 cm | 98.6 % | 98.0 % |
  | straight_long | 98.0 / 94.2 / 85.3 % | 1.3 cm | 74.3 % | 89.9 % |

  At equal length and sampling, the turn fuses 4 to 7 points more of the room, and it closes the cone ahead (98.6 %
  against 74.3 %). The straight path's median error is smaller: it sees more of the room from the side, at longer
  baselines. The starts differ by 2 m, so the scene each path sees is not quite the same.
- **Subsampling hits changes the cell set.** With `--hit-stride 5 --max-cells 50000` (a smoke test of the options,
  Chat sandbox, not a record), the turn gives 99.1 % at 0.5 px. Paths compared against each other must use the same
  stride and cell settings.

### Stage 5b.2, Code part 1: the calibration room at the design resolution (#7; Code, RTX 4090)

**What Code ran.** The 90° turn and the equal-length straight path at 3600 wide, 1 cm steps (218 looks each), with
`--drop-exr`. Renders were at `0b125a5`. Checks used `--cell 0.02 --hit-stride 23 --max-cells 200000`; records are at
`53b81c8` and `1fd9b3d`, in [`runs/issue-7/code/`](runs/issue-7/code). The numbers are Code's, measured on the
workstation; the full report is on #7.

**Cost.** About 12–13 s a look (1.6 s of it Cycles): 45 and 43 min per path. Each fusion check took about 17 min.

**Results.**
- `straight_long` passes every check.
- `turn90` passes F2, F3 and every control.
- Usable at 0.25 / 0.5 / 1 px: turn 99.95 / 99.87 / 99.75 %, straight 99.87 / 99.71 / 99.36 %.
- The 3-D error at 0.5 px has a median of 1.1 mm on both paths. Its 95th percentile is 7.4 mm on the turn and 19.9 mm
  on the straight.
- Ahead within 15°: 99.63 % on the turn against 98.19 % on the straight. From about 0.66 m the turn pulls ahead,
  reaching 95 % of the cells at about 0.68 m against about 0.87 m.

**F1 failed on the turn: one cell of 199,991, at 2.6 % of range.** Code traced it to `fuse`.
- **The cause.** The cell is a floor point 0.34° from a look's nadir, under the arc. Starting 5 % beyond the truth
  along one ray, the fit stalled on another look's nadir, where lon is undefined.
- **Reproduced in the Chat sandbox without renders.** 5 of 2000 synthetic floor cells under the turn's looks stall
  the same way.
- **The fix** (Code's option a, `5a00828` → `3edf1ac`). Each cell starts from the least-squares intersection of its
  observed rays, or from the earlier start, whichever fits the observations better. A numerically singular step falls
  back to the pseudo-inverse. Two intermediate versions crashed on noisy, almost parallel rays in the sandbox (see the
  commits).
- **New check F1b.** No noise; the 2000 floor cells under the turn's looks are all recovered (7.6e-16). Its control,
  the earlier start, stalls on 5.
- **Sandbox reruns at `3edf1ac`** ([`runs/issue-7/rays/`](runs/issue-7/rays)). The composed, turn90 and straight_long
  paths pass F1, F1b, F2, F3 and every control. Usable fractions and coverage are unchanged to four decimals.
- **The estimator and every threshold are unchanged.**

**The Classroom.** Code's smoke test renders `scenes/classroom/classroom_eye.blend` with `sensor.render` as it stands:
97.6 % of rays hit, and the window band carries `inf`, which the oracle and the fusion already skip. Chat accepted
Code's recommended start poses (2026-11-17, #7):
- the turn from `0.897,0.459,90` (up the aisle at x = 0.897; the arc ends toward the front-door side);
- the straight from `0.897,-1.541,90`, 2 m behind the turn's start, as in the room.

### Stage 5b.2, Code part 2: the room after the F1 fix, and the Classroom (#7; Code, RTX 4090)

Every step ran at `3d82252` with `deep/` clean. Records are at `28e3ac2`, in [`runs/issue-7/code/rays/`](runs/issue-7/code/rays)
and [`runs/issue-7/code/classroom/`](runs/issue-7/code/classroom). The numbers are Code's, measured on the
workstation; the report is on #7. Chat checked them against the records.

**The room again.** Both paths pass F1, F1b, F2, F3 and every control. F1 on the turn went from 0.026 to 6.5e-14 of
range, and every other number is part 1's.

**The Classroom** (`classroom_eye.blend`; turn from `0.897,0.459,90`, straight from `0.897,-1.541,90`; 218 looks each
at 1 cm, 3600 wide, about 14.5 s a look; `--cell 0.02 --hit-stride 23 --max-cells 200000`). Every check passes on
both paths.

| | turn90 | straight_long |
|---|---|---|
| seen by two looks | 99.978 % | 99.959 % |
| usable at 0.25 / 0.5 / 1 px | 99.71 / 99.23 / 98.33 % | 99.64 / 99.28 / 98.59 % |
| median / p95 3-D error at 0.5 px | 1.5 / 25.4 mm | 1.4 / 21.6 mm |
| coverage at 0.25 / 0.5 / 1 px | 94.96 / 95.03 / 94.99 % | 95.06 / 94.96 / 94.96 % |
| 95 % of cells usable at | 1.35 m | 1.36 m |

- **The window band's edge** (Code's diagnosis, no check changed). Cells within 2 px of a missed ray (7–8 %) are exact
  under F1, at least as usable as the rest, and their coverage stays within 95 ± 1 %. Of the cells seen by fewer than
  two looks, 3 of 44 (turn) and 7 of 82 (straight) are at the band.
- **Cells outside the room.** The map holds some: through the front door, through an opening in the left wall, and
  beyond the right-hand windows.

**Reading (Chat).**
- **With perfect correspondences at the design resolution, both paths saturate.** About 99 % of the surface is usable
  at 0.5 px in both scenes, so the ±10 % gate no longer separates paths.
- **The geometry shows only at the margins, and the margins depend on the scene.**
  - The room: the turn reaches 95 % sooner (0.68 against 0.87 m) and has the smaller error tail (p95 7.4 against
    19.9 mm).
  - The Classroom, mostly near desks and chairs: the curves coincide, and the straight path's tail is smaller (21.6
    against 25.4 mm).
- **The comparison is of paths with their own surroundings.** The starts are 2 m apart, so each path sees a different
  cell set.
- **What decides the result from here is matching, not the path.** Under the oracle, a smooth path with a real turn
  costs nothing and closes the line's blind cone (5b.1: the heading becomes usable). The open question is which
  correspondences a real matcher can deliver along such a path: stage 7.

### Demo: the Classroom tour (#9; Chat, then Code)

**The tour.** The straight path up the aisle to the turn's start (2.00 m), then the 90° turn: 418 looks at 1 cm,
4.17 m. It reuses #7's renders (3600 wide).

**What it shows.** `deep/demo.py`, after D1, draws two videos: range and error. Each frame has the panorama, a top
view, a 3-D view and the progress curve. The depth is the oracle's at 0.5 px (DERIVED: a display).

**Checks.**
- Chat sandbox at `06153e1`: V1 (the video's first-order points equal `fusion.fuse`, median 0.0014 σ) and V2 (the tour
  is the planned curve, 5e-16) pass, and their controls fail.
- Code, Classroom at `2213227`: V1 0.00013 σ and V2 6.7e-16 pass, and their controls fail. 99.3 % of the cells are
  usable at the end. The run took 37 min at 1.4 GB.
- Records: [`runs/issue-9/`](runs/issue-9).

**Code's one change** (`2213227`, display only): the 3-D views fit the projected cells per azimuth with one scale for
the orbit. Cells seen outside the room (door, opening, windows) had pulled the fixed framing off the room.

**Where the videos are.** On the workstation, 2 × about 140 MB, 27.9 s each, with sha256 sums in the records. Luiz
decided (2026-11-18, in Chat) that they stay on the workstation only: not attached to the issue, not a release asset.

### Stage 7a: real matching on a Lambertian world (R0) and on renders (R1) (#14, #15; Chat sandbox)

**The question.** Can traces be matched from images alone, now that their geometry is known? The test is on the anchors with texture along the trace: 20,000 per run, from the 15 % with the strongest gradient along the trace. The truth is the reference look's positions.

**The data.**
- **R0:** one clean 256-spp panorama of the room, re-projected to every pose. It is point-sampled: no noise, but pixels change in whole jumps.
- **R1:** Code's renders (#15): the calibration room at 720 wide and 128 spp; 101 looks at 1 cm on a 1 m line and on a 1 m arc (R = 1.6 m). The release is `data-r1-calib-720`; the sha256 sums were checked in the Chat sandbox.

**The three matchers** (`deep/match.py`):
- **B, the trace sweep:** a global search over inverse range along each anchor's ray. It is scored by the truncated spread of the image along the predicted trace over the 101 looks, and it marks an anchor reliable when its best cost is low and twice as good as the second-best.
- **A2, Lagrangian:** Gauss–Newton on inverse range from a 2 m start, with the window of looks doubling from ±1 to ±50. Temporal continuity takes the place of the search.
- **A1, Eulerian:** the instantaneous flow at fixed pixels from three looks (2 cm), solved for inverse range with the flow's direction known.

The checks K1–K7 were declared on #14 before each was built.

**Results** (Chat sandbox; records `docs/runs/issue-14/stage7a_b.json` at `26bb99e`, `stage7a_k3.json` at `d1c0ba8`, `stage7a_a_r0.json` and `stage7a_a_r1.json` at `41b2177`). Each cell gives the anchors within ±10 %, with the median |error| in brackets.

| | R0 line | R0 arc | R1 line | R1 arc |
|---|---|---|---|---|
| B, all anchors | 79.3 % (1.3 %) | 79.0 % (1.5 %) | 87.1 % (0.9 %) | 86.9 % (0.9 %) |
| B, off depth edges | 92.8 % | 91.4 % | 96.9 % | 95.5 % |
| B, at depth edges (about 21 % of anchors) | 32.1 % | 34.4 % | 51.0 % | 52.0 % |
| B, marked reliable: kept, then within ±10 % | 51.9 %, 99.3 % | 49.5 %, 99.3 % | 77.6 %, 94.8 % | 75.6 %, 95.2 % |
| A2, all anchors | 72.7 % (1.5 %) | 70.9 % (1.8 %) | 85.6 % (1.0 %) | 84.8 % (0.9 %) |
| A2, marked reliable: kept, then within ±10 % | 59.0 %, 98.3 % | 54.6 %, 97.9 % | 79.5 %, 93.6 % | 76.1 %, 94.2 % |
| A1, ψ 15–165° | 14.7 % (44 %) | 16.7 % (39 %) | 43.4 % (11.9 %) | 36.6 % (15.5 %) |
| anchors per second, B / A2 / A1 | 1,507 / 3,364 / 134,133 | 1,509 / 3,267 / 138,044 | 1,556 / 3,762 / 134,119 | 1,537 / 3,774 / 137,155 |

**The checks:**
- **K1, K2 (B on R0) fail**, at 79 % against 90 %.
- **K3 (B's mark on R0):** the line passes; the arc fails on coverage.
- **K4 (A2 on R0) fails.**
- **K4b, the hypothesis that temporal continuity recovers the anchors B refuses, fails.** On B's unreliable anchors off depth edges, A2 is 65.9 % (line) and 63.1 % (arc) within ±10 %, against B's own 79.3 % and 77.6 % there. The no-schedule control does fail (40.2 % and 37.7 %): the growing window matters to A2, but it does not beat the global sweep.
- **K5 (A1 on R0) fails.**
- **K6 (the marks on R1):** B on the arc passes. B on the line fails, at 94.8 % against 95 %. A2 fails on both.
- **K7 (A1 on R1) fails.**
- Every control fails, as it must: the mirrored path, random marking, and A2 without its schedule.

**What it says:**
- **Renders are easier than R0.** Area-sampled pixels make the traces smooth, so B and A2 both gain about 8 points. At depth edges they gain about 18 points.
- **On R1, B and A2 nearly agree.** Both are right on 84 % of the anchors, only one of them on 4–5 %, and neither on 12 %. The failures are where both fail:
  - depth edges (about half right);
  - the heading cone at 0–15° (37–49 % right). At 720 wide the oracle is also weak there (#15: heading ±15° usable 70 % (line) and 89 % (arc) at σ = 0.5 px).

  So what is left is in the data, not in the search.
- **The search is not the bottleneck.** A2 reaches B's answers without sweeping, at 2.4 times B's speed.
- **The repeated-texture reading of K3 is not supported.** B's unreliable anchors off depth edges are 78–79 % right, so its ratio test is cautious, not confused.
- **The instantaneous flow is dense and fast** (90 times B), **but at 2 cm it carries too little baseline:** a median of 12–16 % on R1. Precision comes from the length of the trace, so the flow's use would be as a dense start for a filter, not as an estimate. This answers the note's §9 for now.

**Next.**
- Depth edges, about a fifth of the anchors and the larger part of the failures: the two-depth handling (one surface before the edge, another after).
- Then the anchors without texture, seeded from the reliable ones.

**Depth edges, method E (K8, declared on #14; Chat sandbox, `9b707d9`, record `docs/runs/issue-14/stage7a_k8.json`).**

- **The diagnosis** (`diag_edge.py`). At the R1 depth-edge anchors B gets wrong:
  - the pixel centre is on the far surface for 94 % of them;
  - B's estimate sits on the near surface for 65–67 %.
  
  B follows the occluding contour, which moves with the foreground. This is stereo's foreground fattening.
- **E:** two flanks at ±2 px along the trace, each ranged by B, with the near or far range picked by pixel coverage.
- **K8 fails on both runs.**
  - At true edges: 50.3 % (line) and 55.7 % (arc), against B's 51.0 % and 52.0 % and the 70 % bar.
  - Off edges, E loses 4–5 points.
  - The controls fail as they must. The coverage rule does carry information: at edges it beats the flipped rule by 7–14 points and the coin by 2.5–6.
- **Why it fails** (`diag_flanks.py`, `diag_flanks_line.txt`).
  - At edge anchors, the flanks are ranged no better than the anchors themselves: 50–51 % against 94 % off edges.
  - This holds whatever the flank's own gradient, so the flanks are not short of texture.
  - The fattening reaches beyond 2 px: over the 1 m window, the foreground's contour sweeps across the background next to it. A background point there is hidden in part of the window, and the contour's coherent motion wins the cost.
  
  An edge cannot be split in space alone. The far surface has to be scored only on the looks where the near one does not hide it.

**Depth edges, method F (K9, declared on #14; Chat sandbox, `2ebef0a`, record `docs/runs/issue-14/stage7a_k9.json`).**

- **F:** the occlusion ordering in time. A far hypothesis, at least 10 % beyond B's range, is scored on each half-window. It replaces B's estimate when its cost is under half of B's.
- **K9 fails on both runs.**
  - At true edges F reaches 49.3 % (line) and 49.9 % (arc), against B's 51.0 % and 52.0 % and the 70 % bar.
  - Off edges it loses 5 points.
  - Both controls fail as they must, but the ordering is barely better than its reversal (49.3 against 47.1 %).
  - The far ranges taken (12 % of anchors, 26–28 % of the edge anchors) are right only 22–26 % of the time, with medians of 15–17 %.
- **What E and F share, and why both fail.** Each scores the far surface against the anchor's own value. At a contour that value is a mixture of both surfaces, which no single surface reproduces. On a half-window, a wrong range at the far side then fits it better than the near trace does.
- **What remains.** Two directions are left, and the choice between them is Luiz's:
  - **A two-layer model per contour anchor:** value = α·near + (1 − α)·far, each moving on its own trace.
  - **An evaluation that accepts what B does.** Most of B's wrong edge anchors sit at the near surface's range, on its silhouette, within a pixel. As 3-D points they lie on a true surface. Judging them by distance to the nearest true surface, rather than by range along the pixel centre's ray, would score them right. That changes the metric, so it is Luiz's decision.

**(b), the surface metric for contour anchors (K10, declared on #14; Chat sandbox, `c28a479`, record `docs/runs/issue-14/stage7a_k10.json`; residual diagnostic `diag_residual.py`, `diag_residual_{line,arc}.txt`).**

- **The metric:** an estimate is surface-right if it is within ±10 % of the true range at any pixel of the anchor's 3×3 neighbourhood. The old metric stays beside it in every record.
- **K10 passes on both runs** (B on R1):

  | | line | arc |
  |---|---|---|
  | at true depth edges, surface-right (old metric) | 75.6 % (51.0 %) | 75.1 % (52.0 %) |
  | overall, surface-right (old metric) | 92.4 % (87.1 %) | 91.6 % (86.9 %) |

  Off edges nothing changes.
- **The controls fail, as they must:**
  - the blend between the neighbourhood's surfaces: 43–45 % at edges;
  - off by 25 %: 2–3 %;
  - the mirrored path: 5–7 %.
- **The reliable marks under the new metric** (reported, not gated):
  - B: 98.3 % and 98.6 %;
  - A2: 97.3 % and 97.8 %.
  
  K6 stays failed: it was declared on the old metric.
- **Half of B's old edge failures were the metric.** The rest, about a quarter of the edge anchors and 5 % of all anchors:
  - mostly lies between the surfaces of its 5×5 neighbourhood (65–71 %), at large depth jumps (median ratio 2.8–3.0);
  - 21–26 % lies in front of every surface;
  - 36 % would count right within 2 px, and widening to 11×11 adds only 3 points;
  - only 18–21 % of it is marked reliable.
  
  So the residual is chiefly blends at big discontinuities: (a)'s target.

**(a), method G, two layers per contour anchor (K11, declared on #14; Chat sandbox, `aa67307`, record `docs/runs/issue-14/stage7a_k11.json`; diagnostic `diag_g.py`, `diag_g_line.txt`).**

- **G:** the flanks at ±3 px are the candidate layers.
  - Each flank is ranged as the near layer on the whole window.
  - Each is ranged as the far layer on each half-window, restricted by the ordering and scored against the flank's own value.
  - A far layer counts as found when B's mark passes it. Pixel coverage then decides near or far.
- **K11 fails, and G does harm.** On the old metric:
  - overall 65.3 % (line) and 67.1 % (arc), against B's 87.1 % and 86.9 %;
  - at edges 48.9 % and 52.7 %;
  - off edges 69.8 % and 70.6 %, against B's 96.9 % and 95.5 %.
  
  Both controls fail as they must.
- **Why.** G finds a "far layer" for 64–65 % of the anchors off depth edges. The diagnostic (line, + flank, the half-window after the reference) traces this to texture:
  - false far layers depend on the flank's texture: 66 % where the flank's gradient along the trace is under a tenth of the anchor's, 29 % where it is over half;
  - they are not a bound effect (0.1 % at the bound).
  
  A half-window fit on a flank with little texture passes B's mark at a wrong range. The flank is not an anchor, and B's mark was calibrated on anchors.
- **Where the edge work stands.** Three ways to model edges (E, F, G) have failed: per-anchor layer decisions are fragile. What does work is B with its mark, judged in 3-D:
  - 92 % surface-right overall;
  - 98 % among the 76–78 % it marks reliable.
  
  The residual is blends, 80 % of them already marked unreliable.

**Anchors without texture, method H (K12, declared on #14; Chat sandbox, `37bd5d8`, record `docs/runs/issue-14/stage7a_k12.json`).**

- **H:** every pixel on a 2-px grid (50,400 targets) takes the best of its 8 nearest reliable seeds' ranges (15,100–15,500 seeds), verified by B's trace cost and refined.
- **The checks:**
  - **K12a fails on coverage:** 75.3 % of the targets, against the 95 % bar. The accuracy part holds: 86.2 % (line) and 87.0 % (arc) surface-right, against 85 %.
  - **K12b passes.** In the least-textured third, H is surface-right 78.3 % and 78.2 %, against B alone at 63.1 % and 62.8 %: +15 points.
  - **The shuffled-seed control fails, as it must:** 75.5 % and 76.4 %.

| surface-right on the targets (line / arc) | coverage | all | least texture | at true edges | ψ 0–15° | ψ 60–120° |
|---|---|---|---|---|---|---|
| H | 75.3 % | 86.2 / 87.0 % | 78.3 / 78.2 % | 75.4 / 74.5 % | 54.0 / 60.6 % | 94.5 / 94.5 % |
| the nearest seed, unverified | 75.3 % | 87.6 / 88.0 % | 80.9 / 79.3 % | 78.2 / 74.2 % | 73.9 / 72.1 % | 96.5 / 95.8 % |
| B alone at every target | 100 % | 76.7 / 76.7 % | 63.1 / 62.8 % | 69.6 / 68.5 % | 44.3 / 53.2 % | 75.7 / 75.9 % |
| control: shuffled seeds | 75.3 % | 75.5 / 76.4 % | 70.3 / 71.2 % | 60.5 / 57.4 % | 38.5 / 28.9 % | 82.4 / 81.3 % |

**What it says:**
- **Seeding is what helps:** +10 points over B alone overall, and +15 where texture is least.
- **Coverage is set by the reach.** The 3×3 block of 10-px cells leaves a quarter of the targets with no seed nearby: the large walls without texture.
- **Verification by the trace cost hurts slightly.** The unverified nearest seed is 1–1.5 points better overall, and 12–20 points better in the heading cone, where traces barely move and the cost cannot tell ranges apart. Yet the cost does carry information where traces move: shuffled seeds are rescued to 75–76 %, and laterally to 81–82 %.
- **So the cost should decide only where it discriminates.** That is a declared change for a next method, not a retune of H.

**Method H2, grown in rounds (K13, declared on #14; Chat sandbox, `971a45a`, record `docs/runs/issue-14/stage7a_k13.json`; diagnostic `diag_rounds.py`, `diag_rounds_line.txt`).**

- **H2:** H's search in rounds, each round's filled targets seeding the next, with the trace cost used only where it discriminates.
- **K13a and K13b fail on both runs.** The control fails, as it must.
  - Coverage reaches 100 %, but H2 is 72.5 % (line) and 73.0 % (arc) surface-right, below B alone at 76.7 %.
  - Round 1 is H's: 75 % of the targets, 88 % right. The later rounds are mostly wrong: 40–44 % in round 2, and 9–32 % after it.
  - The gate seldom engages: H2 is within 0.3 points of "never verify." Always verifying gives 75.5–75.7 %.
- **Where the later rounds land.** 90 % of the targets they fill lie above 30° of elevation, at a true height of 3.2 m: the ceiling, which has no texture.
- **What it means.** Nothing in the images ranges the ceiling: there is no texture on it, so there is no trace to match. Carrying a constant range across it is a guess, and a wrong one, because a plane's range varies across the image.
  - Filling it needs a prior: planes fitted to the reliable seeds, or a room model. That is reconstruction, not matching.
  - The 95 % coverage bar of K12 and K13 was a guess that ignored this. It stays failed as declared.

**The planar prior, method P (K14, declared on #14 after Luiz's decision of 2026-11-20 to stop at H; Chat sandbox, `76e090f`, record `docs/runs/issue-14/stage7a_k14.json`; diagnostics `diag_ceiling.py`, `diag_planes.py` and their outputs).**

- **P:** sequential RANSAC planes (≤ 8, ≥ 500 inliers within 3 cm) on H's points. It fills H's empty targets at the nearest plane met within its support.
- **K14a and K14b fail on both runs.** The control (shuffled ranges) fails, as it must.
  - P fills 28 % of H's empty targets, at 53.3 % (line) and 51.2 % (arc) surface-right.
  - H with P covers 82.3 %, at 83.4 % and 83.9 % surface-right.
- **What P finds:**
  - the floor (twice, 1.7–5 cm apart), all four walls (x = ±3.0, y = ±2.5 m), and a slanted spurious plane or two;
  - **no ceiling.**
  - What it fills on the floor is 100 % right. The ceiling rays it fills meet the walls near their top, and are 37–39 % right.
- **Why** (diagnostic, line):
  - H's 4,374 ceiling points lie a median of 38 cm off the true ceiling; only 604 are within 3 cm. The floor's lie 2 cm off.
  - B's reliable anchors whose pixels are on the ceiling (676) lie a median of 52 cm off it; 134 are within 3 cm. They are mostly fixture and junction edges: right by the surface metric, but not on the ceiling plane.
  - **The calibration room's ceiling carries almost no evidence of its own,** not even enough to fit a plane to.
- **What would close it:** a stronger prior on the room's layout, not on the evidence. For example, a Manhattan room: the ceiling parallel to the floor, closing the walls at their top. That is Luiz's decision.

**Decision (Luiz, 2026-11-20, in Chat): (ii).**
- **The ceiling stays "no evidence."** The planar prior is kept where it is supported, and reported, not gated.
- **The coverage work stops here.** The matcher (B with its mark, then H) goes to the Classroom: R2, #17.
- **On #14's closing:** Luiz closed #14 before K14 was declared, so K14's declaration and its result are comments on the closed issue.

**Stage 7a in short.**
- **The matcher's output:** B's reliable anchors, 98 % surface-right on R1; then H, seeded from them, which covers 75 % of the pixels at 86–87 % surface-right.
- **The ceiling of the calibration room** has no evidence to range it from, and a fitted plane cannot find it.
- **Edges:** three ways of modelling them (E, F, G) failed. Half of the edge "failures" were the old metric, and K10's surface metric records it.

### Stage 7c: R2, the matcher in the Classroom (#17; Code, RTX 4090)

**The run.** 720 wide, 128 spp, 101 looks at 1 cm:
- **line:** `straight_long` from (0.897, −1.541), looks 0–100;
- **arc:** `turn90` from (0.897, 0.459), looks 30–130, R = 1 m.

Renders and matcher at `a0fc8a6`. Records are on `experiment/p7c-r2`, in `docs/runs/issue-17/`:
- `stage7a_r2.json` at `922b117`;
- `diag_r2.txt` at `1407393`.

**The checks, declared on #17 before the data:**
- **K15 fails on coverage only.** B's reliable anchors are 95.4 % (line) and 95.9 % (arc) surface-right, but 45.9 % and 45.5 % are kept, against 50 %.
- **K16 fails.** H is 70.4 % and 78.4 % surface-right on what it fills.
- **K17 fails.** Seeding gains 6.3 and 7.5 points where texture is least, against 10.
- **Every control fails, as it must.**

**Reported:**
- **The Classroom's edges:** 44 % and 31 % of the anchors are at true depth edges, against about 20 % in R1.
- **B off edges:** 84.2 % and 83.9 % surface-right, against 96.9 % and 95.6 %.
- **A2 off edges:** 77.0 % and 75.4 %, worse than B.
- **The planar prior P:** fills 91 % and 98 % of what H leaves empty, at 89.8 % (line) and 34.6 % (arc) surface-right. The arc is not diagnosed.

**Why B drops off edges** (`diag_r2.py`; truth-based; Chat wrote it and ran it on R1, Code ran it on R2). B's wrong anchors off edges are 15.8 % and 16.1 % of the off-edge anchors, against 3.1 % and 4.4 % on R1. They divide among three causes, each of which grows two to five times as a share of the off-edge anchors:
- **occluded:** 22 % and 15 % of the wrong ones;
- **photometric** (the true trace is not consistent where it is seen): 38 % and 31 %;
- **ambiguous** (consistent, but another range costs less): 36 % and 51 %.

**What it says:**
- **B's mark already excludes most of the wrong anchors.** Only about 2 % of all anchors are both marked reliable and wrong.
- **So what is lacking is anchors that are confidently right,** not a better mark.
- **About two-thirds of the wrong ones** (photometric and ambiguous) are failures of B's cost: a single pre-blurred pixel value compared with the reference value. It is too weak a descriptor to tell repeated structures apart, and it is not tolerant of shading or gloss.
- **Temporal continuity does not help:** A2 is worse than B.

**B2, a patch cost (declared on #17; Luiz, 2026-11-20).**
- **What it is:** B's sweep, unchanged, with a different cost per look: 1 − ZNCC of a 5×5 patch translated to the predicted point, truncated at 1.
- **Its mark:** B's rule, restated: mean cost ≤ 0.25 and ratio ≥ 2.

**K18 passes on R1** (Chat sandbox, `5237ca3`, record `docs/runs/issue-17/stage7a_b2_r1.json`):

| surface-right (line / arc) | all | off edges | at edges | reliable: kept, surface-right |
|---|---|---|---|---|
| B | 92.4 / 91.6 % | 96.9 / 95.6 % | 75.6 / 75.1 % | 77.6 / 75.6 %, 98.3 / 98.6 % |
| **B2** | **93.2 / 94.6 %** | **97.5 / 98.1 %** | 77.6 / 80.3 % | 72.8 / 69.6 %, 98.3 / 98.7 % |

- **The controls fail, as they must:**
  - the mirrored path: median 76 % and 79 %;
  - random marking: 93.4 % and 94.5 %. That is close to the bar on the arc, because B2 is mostly right on R1 anyway.
- **B against B2 on the same anchors:**
  - B2 only: 3.0 % and 4.6 %;
  - B only: 2.1 % and 1.6 %;
  - neither: 4.7 % and 3.8 %.
- **Seconds:** B2 takes 217–230 s per path in the Chat sandbox, against B's 14 s (16 times as long).
