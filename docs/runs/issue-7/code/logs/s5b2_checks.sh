#!/bin/bash
# Stage 5b.2 (#7), Code part 1: the fusion checks and figures on the two calibration-room runs, one after another.
set -u
cd /home/lvelho/rd/deep-motion
L=runs/s5b2-logs
step() { local log=$1; shift; echo "$(date -Is) \$ $* (at $(git rev-parse --short HEAD))" > $L/$log.log; /usr/bin/time -v "$@" >> $L/$log.log 2>&1; echo "$(date -Is) exit $?" >> $L/$log.log; }
step check-turn .venv/bin/python -m deep stage5b2 runs/s5b2c-turn --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code
step check-slong .venv/bin/python -m deep stage5b2 runs/s5b2c-slong --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code
step figures .venv/bin/python -m deep figures --stage 5b2 --runs turn90=runs/s5b2c-turn straight_long=runs/s5b2c-slong --out docs/runs/issue-7/code/figures
