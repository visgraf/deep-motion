import sys, json, subprocess, time
sys.path.insert(0, 'external/fov-render')
from pathlib import Path
from motion import analysis, m4_pipeline as mp
def git(p):
    c = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    clean = subprocess.run(['git', '-C', p, 'status', '--porcelain', '--untracked-files=no'], capture_output=True, text=True).stdout.strip() == ''
    return {'commit': c, 'clean': clean}
G = {'foveal-motion': git('.'), 'fov-render': git('external/fov-render')}
run = analysis.load_run('runs/m3-calib-1024'); ks = sorted(e['k'] for e in run['looks']['ahead'])
out = Path('runs/m4-calib'); log = lambda m: print(m, flush=True)
M3_SCALE = 1.625204104083527                     # M3's frozen calibration-room scale (#7, docs/runs/issue-7/final/calib/rgb/freeze.json)
for w in sys.argv[1:]:
    refs = mp.references(ks, w)
    have = set(json.loads((out / w / 'config.json').read_text())['refs']) if (out / w / 'config.json').exists() else set()
    cal = [r for r in refs if mp.is_calibration(r) and r not in have]
    held = [r for r in refs if not mp.is_calibration(r) and r not in have]
    if cal: mp.estimate(run, w, cal, out, git=G, log=log)
    c = json.loads((out / w / 'calibration.json').read_text()) if (out / w / 'freeze.json').exists() else mp.calibrate(out, w)
    log(f'[m4] {w}: interval scale {c["interval_scale"]:.3f} from {c["n"]} anchors, |z| quantiles {c["abs_z_quantiles"]}')
    if held: mp.estimate(run, w, held, out, git=G, log=log)
    res = mp.evaluate(out, w, M3_SCALE, log=log)
    log(f'[m4] {w} EVALUATION ' + json.dumps({k: res[k] for k in ('M4', 'M3')}, default=float))
