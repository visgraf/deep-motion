#!/usr/bin/env bash
# M6a T2 (#22): after alpha = 10's render, render alpha = 0 and 20, then the oracle and RGB on refs 8 12 16.
set -u
cd /home/claude/foveal-motion
PY=.venv/bin/python
while [ ! -f runs/m6-crab-a10/run.json ]; do sleep 10; done
echo "[t2] a10 rendered $(date +%T)"
for A in 0 20; do
  P=paths/calib_crab_a$A.json; [ $A = 0 ] && P=paths/calib_dolly.json
  $PY -m motion.dolly preflight --path $P --run runs/m6-crab-a$A --device CPU --width 360 --spp 2 | tail -1
  $PY -m motion.dolly run --path $P --run runs/m6-crab-a$A --device CPU --modes ahead --spp 256 --seed-base $((8000 + 10*A)) | tail -1
  echo "[t2] a$A rendered $(date +%T)"
done
for A in 10 0 20; do
  $PY -m motion.dolly m6-t2 --run runs/m6-crab-a$A --refs 8 12 16 --out /tmp/t2out/a$A --oracle --rgb
  echo "[t2] a$A estimated $(date +%T)"
done
$PY -m motion.dolly m6-t2 --run runs/m6-crab-a10 --refs 8 12 16 --out /tmp/t2out/a10 --control-aligned
echo "[t2] done $(date +%T)"
