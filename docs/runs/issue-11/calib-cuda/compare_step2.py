"""Step 2 of #11: the calibration room on CUDA (runs/m4-calib-cuda) against Chat's CPU result
(docs/runs/issue-10/stage5) and the frozen record (docs/runs/issue-10/frozen.json).

Stop rule (Chat, #11): a CUDA scale more than 5 % from the frozen one, or any gate verdict or check that
differs -> stop. Prints a table per window and a STOP / GO line; exit 1 on STOP.
Usage: python compare_step2.py REPO CUDA_OUT
"""
import json
import sys
from pathlib import Path

repo, cuda = Path(sys.argv[1]), Path(sys.argv[2])
frozen = json.loads((repo / "docs/runs/issue-10/frozen.json").read_text())
stop = []
for w in ("both5", "before5"):
    cpu_d, gpu_d = repo / "docs/runs/issue-10/stage5" / w, cuda / w
    cc, gc = (json.loads((d / "calibration.json").read_text()) for d in (cpu_d, gpu_d))
    ce, ge = (json.loads((d / "evaluation.json").read_text()) for d in (cpu_d, gpu_d))
    fz = frozen["windows"][w]["interval_scale"]
    dscale = gc["interval_scale"] / fz - 1
    print(f"== {w}: scale CUDA {gc['interval_scale']:.6f}  frozen {fz:.6f}  ({dscale:+.3%});  "
          f"calibration anchors CUDA {gc['n']}  CPU {cc['n']};  refs CUDA {gc['refs']}  CPU {cc['refs']}")
    if abs(dscale) > 0.05:
        stop.append(f"{w}: scale {dscale:+.2%}")
    if ge["held_out_refs"] != ce["held_out_refs"]:
        stop.append(f"{w}: held-out refs differ {ge['held_out_refs']} vs {ce['held_out_refs']}")
    for tag in ("M4", "M3"):
        for k in ("field_usable_of_all_hits", "accepted_fraction", "interval_coverage_interior",
                  "usable_interior_median_rel", "usable_interior_p90_rel", "fovea_usable"):
            a, b = ge[tag][k], ce[tag][k]
            print(f"   {tag} {k:28s} CUDA {a:.6f}  CPU {b:.6f}  diff {a - b:+.2e}")
        for k in ("subset", "accepted_usable", "coverage_of_subset", "median_rel", "p90_rel", "over_50pct", "gate_pass"):
            a, b = ge[tag]["gate"][k], ce[tag]["gate"][k]
            extra = "" if isinstance(a, (bool, int)) and not isinstance(a, float) else f"  diff {a - b:+.2e}"
            print(f"   {tag} gate.{k:23s} CUDA {a}  CPU {b}{extra}")
        if ge[tag]["gate"]["gate_pass"] != ce[tag]["gate"]["gate_pass"]:
            stop.append(f"{w}: {tag} gate verdict differs")
    gch = {c["id"]: c for c in ge["checks"]}
    for c in ce["checks"]:
        g = gch[c["id"]]
        print(f"   {c['id']}: ok CUDA {g['ok']} CPU {c['ok']}; control fails CUDA {g['control_fails']} CPU {c['control_fails']}")
        if (g["ok"], g["control_fails"]) != (c["ok"], c["control_fails"]):
            stop.append(f"{w}: {c['id']} differs")
print("STOP: " + "; ".join(stop) if stop else "GO: every scale within 5 % of the frozen one; every gate verdict and check as on the CPU")
sys.exit(1 if stop else 0)
