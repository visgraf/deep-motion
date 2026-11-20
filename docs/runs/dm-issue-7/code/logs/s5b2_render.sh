#!/bin/bash
# Stage 5b.2 (#7), Code part 1: the two calibration-room renders at 3600 wide, one after another.
set -u
cd /home/lvelho/rd/deep-motion
L=runs/s5b2-logs
step() { local log=$1; shift; echo "$(date -Is) \$ $* (at $(git rev-parse --short HEAD))" > $L/$log.log; /usr/bin/time -v "$@" >> $L/$log.log 2>&1; echo "$(date -Is) exit $?" >> $L/$log.log; }
step render-turn .venv/bin/python -m deep stage5b2-render --path turn90 --step 0.01 --width 3600 --device OPTIX --drop-exr --run runs/s5b2c-turn --seed-base 14000
step render-slong .venv/bin/python -m deep stage5b2-render --path straight_long --start 0,-2,90 --step 0.01 --width 3600 --device OPTIX --drop-exr --run runs/s5b2c-slong --seed-base 15000
