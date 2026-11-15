"""M3a Stage 2 (#6): the oracle ladder against M2, from the runs' JSON files (Chat sandbox).

    python docs/runs/issue-6/ladder_table.py runs > docs/runs/issue-6/ladder.txt
"""
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
m2 = json.loads((runs / "m3-calib-source" / "depth.json").read_text())["summary"]["ahead"]
src = [("O1 exact", "m3-calib-O1"), ("O2 pixel", "m3-calib-O2"), ("O3 0.05 px", "m3-calib-O3/sigma-0.050"),
       ("O3 0.10 px", "m3-calib-O3/sigma-0.100"), ("O3 0.25 px", "m3-calib-O3/sigma-0.250")]
ev = {n: json.loads((runs / p / "evaluation.json").read_text()) for n, p in src}
f = lambda v: "   -  " if v is None else f"{v:6.3f}"
for n, e in ev.items():
    print(f"{n:11s} computed_by {e['computed_by']['foveal-motion']['commit'][:7]} clean {e['computed_by']['foveal-motion']['clean']}"
          f"  checks " + ", ".join(f"{c['id']} {'PASS' if c['ok'] else 'FAIL'}/ctrl {'fails' if c['control_fails'] else 'DOES NOT FAIL'}" for c in e["checks"]))
print()
print("Usable range (interval relative half-width <= 10 %) as a fraction of ALL of look b's hits; M2: usable by bracket")
print("and (in brackets) within 10 % by error. Calibration room, ahead, full profile, 64 spp, Chat sandbox.")
for region in ("field", "band_5_40", "fovea_2deg"):
    print(f"\n{region}")
    print("stride  B (m)   " + "  ".join(f"{n:>11s}" for n in ev) + "    M2 bracket (M2 within 10 %)")
    for s in (1, 5, 10):
        k = f"stride_{s}"
        if region == "field":
            mm = (m2[k]["usable_bracket_fraction"], m2[k]["usable_fraction"])
        elif region == "fovea_2deg":
            mm = (m2[k]["fovea_2deg"]["usable_bracket_fraction"], m2[k]["fovea_2deg"]["usable_fraction"])
        else:
            mm = (None, None)
        cells = "  ".join(f"{f(ev[n]['summary'][k][region]['usable_of_all']):>11s}" for n in ev)
        print(f"{s:6d}  {ev['O1 exact']['summary'][k]['B_m']:5.3f}   {cells}    {f(mm[0])} ({f(mm[1])})")
print("\nError within 10 % of all hits (error-based, as M2's 'usable'):")
for s in (1, 5, 10):
    k = f"stride_{s}"
    print(f"{s:6d}  " + "  ".join(f"{f(ev[n]['summary'][k]['field']['within_10pct_of_all']):>11s}" for n in ev)
          + f"    M2 {f(m2[k]['usable_fraction'])}")
print("\nBy eccentricity, stride 5: usable of all hits | median relative error")
bins = [b["bin"] for b in ev["O1 exact"]["summary"]["stride_5"]["by_eccentricity"]]
print("bin (deg)  " + "  ".join(f"{n:>17s}" for n in ev))
for i, b in enumerate(bins):
    row = []
    for n in ev:
        x = ev[n]["summary"]["stride_5"]["by_eccentricity"][i]
        row.append(f"{f(x['usable_of_all'])} | {f(x['median_rel'])}")
    print(f"{b[0]:4g}-{b[1]:<4g}  " + "  ".join(f"{r:>17s}" for r in row))
print("\nO2 interval coverage of the truth (field):", ", ".join(
    f"stride {s} {ev['O2 pixel']['summary'][f'stride_{s}']['field']['coverage']:.3f}" for s in (1, 5, 10)))
