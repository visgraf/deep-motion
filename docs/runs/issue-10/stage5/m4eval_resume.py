import sys, json
sys.path.insert(0, 'external/fov-render')
from pathlib import Path
from motion import m4_pipeline as mp
res = mp.evaluate(Path('runs/m4-calib'), sys.argv[1], 1.625204104083527, log=lambda m: print(m, flush=True))
print(f'[m4] {sys.argv[1]} EVALUATION ' + json.dumps({k: res[k] for k in ('M4', 'M3')}, default=float), flush=True)
