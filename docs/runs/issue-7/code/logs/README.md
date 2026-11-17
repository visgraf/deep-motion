# Logs of #7's Code part 1 (calibration room, 3600 wide)

Each log starts with the command, its time and the commit it ran at, then the command's own output and
`/usr/bin/time -v`'s report. The drivers are `s5b2_render.sh` (the two renders) and `s5b2_checks.sh` (the two
`stage5b2` checks and the figures), kept as they ran.

**The last line, `<time> exit N`, is wrong in every log here.** The drivers' `step()` wrote `$?` after a `$(date -Is)`
substitution in the same `echo`, so it records the exit status of `date` (always 0). The command's real status is
`/usr/bin/time`'s `Exit status:` line. It is 0 everywhere except `check-turn.log`: 1, because F1 failed. The `#6` logs
(`docs/runs/issue-6/logs/`) agree with their `Exit status: 0` lines. `classroom/smoke/render.log` has the same wrong line (its `Exit status:` is 0); the `f1/` logs read `$?` correctly.
