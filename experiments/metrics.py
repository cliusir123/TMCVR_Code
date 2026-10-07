import json
from pathlib import Path
import numpy as np
N_VRF_CHAINS = 1000

def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

class SynthData:

    def __init__(self, path):
        rows = load_jsonl(path)
        self.meta = {}
        runs = {}
        for r in rows:
            if r.get('kind') == 'transport':
                self.meta[r['label']] = r
            elif r.get('kind') == 'run':
                runs[r['label'], r['sampler'], r['gamma']] = r
        self.runs = {}
        for (label, s, g), r in runs.items():
            if label in self.meta:
                self.runs.setdefault((label, s), []).append(r)
        for k in self.runs:
            self.runs[k].sort(key=lambda r: r['gamma'])

    def labels(self):
        return list(self.meta)

    def __bool__(self):
        return bool(self.meta)

def run_stats(run, j, arm='main'):
    c = run['chain'][arm]
    out = {}
    for key, name in (('fbar', 'plain'), ('mu_fixed', 'fixed')):
        est = np.array([row[j] for row in c[key][:N_VRF_CHAINS]], dtype=float)
        out[name] = float(est.var(ddof=1))
    return out

def grid_metrics(meta, runs, j, arm='main'):
    g = np.array([r['gamma'] for r in runs])
    var_pi, n = (meta['var_pi'][j], meta['n_steps'])
    st = [run_stats(r, j, arm) for r in runs]
    with np.errstate(divide='ignore', invalid='ignore'):
        vp = np.array([x['plain'] for x in st])
        vc = np.array([x['fixed'] for x in st])
        return {'gamma': g, 'ess_n': var_pi / (n * vp), 'vrf': vp / vc, 'eff': var_pi / (n * vc)}
