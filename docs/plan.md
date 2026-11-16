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
| Arc | between 0 and ∞ | rotation about an axis offset by R | concentric mosaics: arc position θ, slit angle β to the tangent | ρ, distance from the turn axis | the turn's vertical axis |
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
