# deep-motion

Depth from motion along **curved paths**: an independent experimental repository, forked from
[visgraf/foveal-motion](https://github.com/visgraf/foveal-motion) at `08db383` (its history is kept here), rendering through
[visgraf/fov-render](https://github.com/visgraf/fov-render) (a submodule, `external/fov-render`).

foveal-motion measured depth from a foveated eye on a moving head. Its experiments ran from M1 (the dolly) to M4
(multi-baseline depth from forward motion, frozen), then D1 (a Classroom tour with turns in place). M6a showed that
extending M4's matcher to curved paths does not collect the geometry's gain. deep-motion starts from that finding
(foveal-motion's M-D19).

**The approach** ([`docs/plan.md`](docs/plan.md)):
- **Paths** are composed of primitives: lines, arcs and pivots, parametrised by curvature.
- **Coordinates** are adapted to each primitive's symmetry, so that matching is a 1-D search over a depth invariant
  and the space-time volume can be read directly along predicted traces.
- **Sensors** may be any design that does the job, including sampling that differs in space and in time.
- **Order**: the formulation is built and judged with an oracle (a perfect matcher) first; real matching comes last.

How the work is done (Luiz, Chat, Code; GitHub Issues as the channel): [`CLAUDE.md`](CLAUDE.md).
foveal-motion's README, as it stood at the fork: [`docs/foveal-motion-README.md`](docs/foveal-motion-README.md).

    git clone --recurse-submodules https://github.com/visgraf/deep-motion.git && cd deep-motion
    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m motion.dolly m6-selftest     # foveal-motion's M6a geometry, numpy only, seconds
