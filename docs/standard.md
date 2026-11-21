# The process standard, S2

**Status: S2 in force** since 2026-11-21 (Versions, at the end). S1 was proposed by Chat on #27 and accepted by Luiz
the same day, in Chat (P-D4 in `docs/plan.md`); S2 changes one rule of its gate (P-D5, #28). It comes from the diagnosis
of #26 and its tech note, "Why the Real Matcher Fell Short: A Diagnosis on a Straight Path" (`straight-diagnosis/` in
[lvelho/foveal-depth-tr](https://github.com/lvelho/foveal-depth-tr)). The name S1 below means this standard; its
version is the one in force.

Every number below says where it comes from (CLAUDE.md, rule 1). *Provisional* marks a value that binds but that no
run has tested yet; the run that tests it can change it, through an issue and Luiz.

## Why

Stage 11 found that the impression "M4 works, deep-motion does not" rested on three confounds (#26, `docs/plan.md`):
- a weaker matcher;
- renders with an eighth of the samples;
- a yardstick, the perfect matcher, that ignores the images.

One rule also changed meaning between sensors: M4's depth-edge tolerance, given in pixels, is about five times stricter
in depth on the panorama to the side (estimated). No stage made a mistake of its own: each was run and judged under its
own conditions. S1 fixes those conditions per task, so that results can be compared, and it puts errors before coverage
(Luiz, 2026-11-21, in Chat).

## How it is used

1. **A run declares a profile.** The issue that plans a run names its profile (below). Before any data, it lists every
   departure from the profile and the reason for it, as with checks.
2. **The record carries it.** Every run record holds `"standard"` (the version in force, now `"S2"`), `"profile": "<name>"`, the values as run and
   the departures.
3. **Like with like.** Rows compared in one table share every value of the profile except the one under study. A row
   that differs in more than one value is labelled with all of them. (#26: B at 128 spp was read against M4 at 1024.)
4. **Errors first.** Of two variants, the one that passes the error gate wins; if both pass, the one that covers more
   wins. Coverage never buys back errors.
5. **Changes** go through an issue and Luiz, and raise the version (S2, …). Earlier records are neither relabelled nor
   re-judged.

## The six parts

### 1. Rendering

- **Fixed and recorded:**
  - the scene;
  - the spp;
  - the pixel filter and colour (fov-render's `measurement`);
  - one seed per look (`seed_base + k`, as `deep/sensor.py` does). With a single seed, the noise would follow the
    pixel grid from look to look.
- **The noise model.** Cycles runs with no adaptive sampling and no denoiser (fov-render `bl/config.py`), so σ falls as
  1/√spp.
- **The noise is measured, not assumed.**
  - In the real-matching profiles, each reference look is rendered a second time, independently: another seed at the
    same spp, or another spp with the ratio known.
  - The gradient SNR along the trace is computed as in stage 11b (`deep/straight.py`, `attributes`).
  - The record gives its median over the surface's pixels, and the share of those pixels under 1 and under 3.
  - #26 (measured, Code): at 1024 spp, the median gradient SNR was 1.65. Below SNR 1, N covered 3.3 % of the surface
    and the perfect matcher 84.8 %.
- **The spp.**
  - Every real-matching run in S1 uses 1024 spp, the level at which M4 was validated.
  - 128-spp renders serve geometry and the perfect matcher only.
  - Whether to set a target in SNR rather than in spp is left open (below).
- **Cost** (measured, Code, [`r3_1024_timing.json`](runs/dm-issue-26/r3_1024_timing.json)):
  - 2.4 s of rendering per 720-wide look at 1024 spp on the RTX 4090;
  - 218 looks in 10 min 36 s;
  - a second render of 12 references therefore takes about 35 s (estimate).

### 2. Path

- **Fixed and recorded:**
  - the path spec (`paths/*.json`) and its primitives (lines, arcs, pivots) with their lengths;
  - the head's facing;
  - the look spacing.
- **The look spacing is 1 cm,** as in every stage since 5b.1. Coarser time sampling is the sensor's choice, made by
  decimating these looks (P-D1).
- **Reported:** which primitives each reference's window covers.

### 3. Sensor

- **Fixed and recorded:**
  - the sensor and its resolution. S1's default is fov-render's Panorama at 720 wide, 0.5°/px; M4's is the Foveated
    sensor;
  - the set of rays processed;
  - each ray's time sampling.
- **The ray set is a declared mask on observable quantities:**
  - the angle ψ from the heading;
  - the latitude;
  - once a model exists, the estimated range.

  The default is N's: ψ from 1° to 179°, latitude under 70° either way. Choosing the ray set is how a sensing policy avoids the
  difficult cases (Luiz, 2026-11-21, in Chat).
- **Never on truth.** A ray set may not use the true range, the true depth edges or anything else from the Position
  pass. Those serve evaluation only.
- **Time sampling.** By default each ray has 11 frames 10 cm apart over ±0.5 m (M4's and N's). Schedules per ray are
  allowed, and recorded.
- **Where the errors are,** #26 at 10 % of ζ, all cells (measured, Code,
  [`stage11c.json`](runs/dm-issue-26/stage11c.json)):

  | ψ from the heading | 0–15° | 15–60° | 60–120° | 120–165° | 165–180° |
  |---|---|---|---|---|---|
  | cells right | 99.9 % | 97.4 % | 93.4 % | 93.0 % | 99.3 % |

### 4. Matcher

- **Fixed and recorded:**
  - the matcher's name and commit;
  - its full settings;
  - its windows;
  - its depth envelope.
- **Each tolerance is stated in the quantity it protects.**
  - A test that protects depth gives its tolerance in depth, as relative inverse depth ζ, and not in pixels, unless the
    profile fixes the sensor.
  - Tolerances in pixels stay where the quantity really is in pixels, such as localisation on a fixture.
  - #26: M4's depth-edge tolerance (0.5 px of the longest step's displacement) is about 9 % of ζ in M4's field and
    about 1.6 % to the side on the panorama (estimates from the two sensors' mappings). At 10 % of ζ, N's depth-edge
    rejections fell from 433,295 to 112,068 (measured, Code).
- **The depth envelope** searched along each ray is declared. By default it is the matcher's own. Once a model exists, a
  policy may narrow it around what has been built; the policy is declared.
- **Frozen baselines.**
  - M4 (`motion/`, CLAUDE.md rule 7);
  - N as at #26 (`deep/ntrace.py` at `280b561`), at 0.5 px and at 10 % of ζ;
  - the perfect matcher (σ = 0.5 px on a 2-px grid, kept within ±10 %; stage 9).

  A new matcher is compared with them under the same profile.

### 5. Fusion

- **Fixed:** stage 8's fusion:
  - 2 cm world cells, keyed by the estimated point;
  - each cell's point is the median of its estimates;
  - no truth is used.
- **The model is its confirmed cells** (the 0.5 m is provisional until #28 measures it).
  - A cell is confirmed when it is supported by at least two references at least 0.5 m apart along the path.
  - The cells that are not confirmed are candidates. They are kept and reported separately, and never counted as the
    model.
- **Why the references must be apart:**
  - References 10 looks apart share 91 of their 101 looks, and agreement between them is weaker evidence than it looks
    (stage 9, Code's reading).
  - In #26, confirmation by any two references raised N at 0.5 px from 97.23 to 98.70 % right, but N at 10 % only from
    95.16 to 95.76 % (measured, Code, `stage11c.json`). This suggests that N's errors at 10 % repeat from one nearby
    reference to the next.
  - Two references 0.5 m apart, the window's half-length, share 51 of their 101 looks and see each point from 0.5 m
    apart. The first run under S1, #28, tests the value (below).

### 6. Evaluation

- **The yardstick is stage 9's metric, unchanged.**
  - A cell is right when its range is within ±10 % of the truth in a 3×3 neighbourhood of its dominant reference.
  - Coverage is the share of the surface's cells that lie within 2 cm of a right cell.
- **The error gate comes first** (Luiz, 2026-11-21: 99.5 %).
  - It applies to the model: cells right ≥ 99.5 %, overall and in each sector of ψ that the ray set includes.
  - A sector that fails is either dropped from the ray set, by a new run declared before its data, or the run fails.
  - Below 1,000 cells, in the model or in a sector, that part of the gate is not evaluated. It is never reported as
    passed.
  - **The whole gate** (S2, Luiz, 2026-11-21, #28): it fails if any part fails, and it is not evaluated if the model
    as a whole has under 1,000 cells. Otherwise it passes, and the sectors under 1,000 cells are listed as unverified.
    Their cells still count in the overall. (Under S1 an unverified sector blocked a pass: on #28's straight path the
    165–180° sector never reached 1,000 cells, even for the perfect matcher, so no full-field model could pass.)
  - For scale (measured, Code, #26):
    - the perfect matcher's cells are 99.98 % right;
    - M4's cells confirmed by two references, in its own field, 99.83 %;
    - N's at 0.5 px, 98.70 %.
- **Coverage comes second.** It is reported for the model against three references:
  - the perfect matcher, the ceiling set by the geometry;
  - the perfect matcher kept where the gradient SNR is at least 3, a rough ceiling set by the noise (stage 11b);
  - the frozen M4, where it ran on the same line (its field, 1–44°).
- **Reported by class,** for the model and the candidates alike. Each class gives its cells, cells right and coverage:
  - the sector of ψ: 0–15, 15–60, 60–120, 120–165 and 165–180° (stage 9's);
  - the true range: under 1 m, 1–2, 2–3, and 3 m or more;
  - at a depth edge or interior. A pixel is at a depth edge when the true range over ±2 px varies by a ratio above
    1.10 (stage 9's);
  - the gradient SNR: under 1, 1–3, 3–10, and 10 or more.

  The gates apply only to classes the method can observe (the sector). The other classes are reported to diagnose. #26
  (measured, Code): at 10 %, cells at depth edges were 83.1 % right and interior cells 98.1 %.
- **The standard controls,** each of which must fail (CLAUDE.md, rule 2):
  - the mirrored path's poses (geometry; stage 9's A1, #26's C3);
  - the members moved 20 % (fusion; stage 8's E2);
  - steps 20 % too long, and the scene scaled 1.5× (the matcher on fixtures; #26's C1).

  An ablation or build run ships at least the first.

## Procedure

CLAUDE.md already holds most of the procedure. S1 adds what #26 taught:

1. **A dry run first,** in the `dry` profile, through the same code path as the real run, before any GPU work.
2. **Provenance is read before the run:** the commit and a clean tree, recorded at the start. In #26, records that read
   it after the run said `clean: false`.
3. **Declarations come before data:** the profile, the departures, the checks and the controls. Amendments are allowed
   only before data, and are recorded (as on #22 and #26).
4. **A variant keeps its name.** A variant found not to be what was defined is renamed, and its definition stays fixed
   (#26: `ROW_ALIAS`).

## The profiles

| | **dry** | **fixture** | **oracle** | **ablation** | **build** |
|---|---|---|---|---|---|
| task | a stage's dry run | a matcher's unit test | geometry, perfect matcher | one change, on one primitive | incremental reconstruction |
| who | Chat sandbox | Chat sandbox | Chat; Code at design resolution | Code | Code |
| scene | M4's fixture planes, ray-cast exactly | M4's fixtures, rebuilt for the sensor (`deep/nfixtures.py`) | calibration room; Classroom | Classroom | Classroom |
| rendering | ray-cast, Gaussian σ 0.005 and 0.005·√8 | as M4's fixtures (noise 0.005) | the Position pass; any spp | 1024 spp, plus a second render of the references | 1024 spp, plus a second render of the references |
| path | the stage's | 10 cm steps, ±5 frames | as declared | one primitive (R3's `straight_long` first) | a composed path (#9's tour) |
| sensor | the stage's | the matcher's | Panorama 720 (3600 at design resolution) | Panorama 720, the default ray set | Panorama 720, a declared ray set |
| matcher | the stage's | the one under test | perfect | the variant, beside the frozen baselines | the accepted one |
| fusion | stage 9's | none | stage 9's | stage 9's; the model and candidates also reported | the model, grown as each window completes (stage 10) |
| gates | none ("not gated") | M4's: median ≤ 0.1 px, p90 ≤ 0.25 px, ≥ 100 accepted; periodic ≤ 1 % over 1 px; radial ≤ 1 % accepted | coverage and cells right | the error gate, on the model | the error gate, on the model and each included sector |
| reports | consistency only | per fixture and label | per sector | every class, with the three coverage references | every class, with the three coverage references |
| controls | none | steps 20 % long; scene ×1.5 | mirrored poses | mirrored poses; members moved 20 % | mirrored poses; members moved 20 % |
| cost | interactive or batch | batch | batch | batch; longer if the issue says (#26: about 1 h) | batch; longer if the issue says |

A demo is made from a build run and its record. It adds no gate.

## Decided at adoption, and open

Decided by Luiz on 2026-11-21, in Chat (#27):
- **The gate is 99.5 %:** one wrong cell in 200. Applied to #26's cells as they stand, not to an S1 model, N at 10 %
  would pass in 0–15° only (99.9 %, 9,396 cells; measured, Code).
- **A matcher may use the noise level σ.** σ is a calibration of the sensor, like its intrinsics. A matcher may then
  skip anchors whose gradient SNR is below a floor, declared with its other settings.
- **The first run under S1** tests the confirmation distance (#28).

Open:
1. **The confirmation distance, 0.5 m** (#28). On #26's straight path, from its cached anchors (12 references, 10 looks
   apart): the model's cells right and coverage for N at 0.5 px and at 10 %, with the supporting references spanning
   at least 0, 0.2 and 0.5 m. Code, CPU (estimate: minutes).
2. **A noise target in SNR rather than in spp.** The two routes are 4096 spp (median gradient SNR about 3.3,
   estimated from 1/√spp; renders about four times as long) and more frames per window: the windows hold 101 looks,
   and N uses 11.

## Adoption (2026-11-21)

- **The record:** P-D4 in `docs/plan.md`.
- **CLAUDE.md** (approved by Luiz): rule 7 names this file among the places for new work, and the project facts say
  that each run declares an S1 profile and that its record carries it.
- **The code:** `deep/standard.py` holds the profiles as data, the error gate, and `check_record`, which every harness
  from #28 on calls on its record. Its controls: a record with an undeclared departure fails, and so does a gate on
  98 % of 2,000 cells (`python -m deep standard-selftest`).

## Versions

- **S1** (Luiz, 2026-11-21, in Chat; #27, P-D4): the standard as proposed, with the gate at 99.5 %, the noise level σ
  open to matchers, and the first run (#28). #28's stage 12 ran under it.
- **S2** (Luiz, 2026-11-21, in Chat; #28, P-D5): a sector under 1,000 cells is listed as unverified and does not block
  the gate; the gate still fails if any evaluated part fails. Records made under S1 are not re-judged.
