#!/bin/bash
# Stage 5b.2 (#7), Code: the room's checks again after the F1 fix, then the Classroom (Chat's comment at 3d82252).
set -u
cd /home/lvelho/rd/deep-motion
L=runs/s5b2-logs
step() {
  local log=$1; shift
  local tree; git diff --quiet HEAD -- deep && tree="deep/ clean" || tree="deep/ MODIFIED"
  echo "$(date -Is) \$ $* (at $(git rev-parse --short HEAD), $tree)" > $L/$log.log
  /usr/bin/time -v "$@" >> $L/$log.log 2>&1
  local rc=$?
  echo "$(date -Is) exit $rc" >> $L/$log.log
}
P=.venv/bin/python
step rays-check-turn  $P -m deep stage5b2 runs/s5b2c-turn  --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code/rays
step rays-check-slong $P -m deep stage5b2 runs/s5b2c-slong --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code/rays
step rays-figures     $P -m deep figures --stage 5b2 --stride 23 --runs turn90=runs/s5b2c-turn straight_long=runs/s5b2c-slong --out docs/runs/issue-7/code/rays/figures
step k-render-turn    $P -m deep stage5b2-render --path turn90 --start 0.897,0.459,90 --height 1.6 --blend scenes/classroom/classroom_eye.blend --step 0.01 --width 3600 --device OPTIX --drop-exr --run runs/s5b2k-turn --seed-base 17000
step k-render-slong   $P -m deep stage5b2-render --path straight_long --start 0.897,-1.541,90 --height 1.6 --blend scenes/classroom/classroom_eye.blend --step 0.01 --width 3600 --device OPTIX --drop-exr --run runs/s5b2k-slong --seed-base 18000
step k-check-turn     $P -m deep stage5b2 runs/s5b2k-turn  --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code/classroom
step k-check-slong    $P -m deep stage5b2 runs/s5b2k-slong --hit-stride 23 --max-cells 200000 --out docs/runs/issue-7/code/classroom
step k-figures        $P -m deep figures --stage 5b2 --stride 23 --runs turn90=runs/s5b2k-turn straight_long=runs/s5b2k-slong --out docs/runs/issue-7/code/classroom/figures
