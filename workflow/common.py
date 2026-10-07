import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
RESULTS = ROOT / 'results'
REFERENCES = RESULTS / 'references'
OUTPUT = ROOT / 'output'
import os
FLOW_SET = os.environ.get('GTI_FLOWS', 'flows')
RESULT_TAG = os.environ.get('GTI_TAG', '')

def flow_dir(t):
    return RESULTS / FLOW_SET / t

def raw_dir(t):
    d = RESULTS / 'raw' / t
    d.mkdir(parents=True, exist_ok=True)
    return d

def out_dir(t):
    d = OUTPUT / t
    d.mkdir(parents=True, exist_ok=True)
    return d

def setup(dtype=torch.float64, device=None, seed=0):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(dtype)
    torch.set_default_device(torch.device(device))
    torch.manual_seed(seed)

def make(t):
    from distributions import make_target
    from workflow.config import TARGETS
    target = make_target(TARGETS[t].get('base', t), **TARGETS[t]['kwargs'])
    target.name = t
    return target

def truth(target):
    gt = target.ground_truth()
    return {'mean': [float(v) for v in gt['mean']], 'var': [float(v) for v in gt['var']]}

def draws(target, n, generator=None):
    if hasattr(target, 'sample'):
        try:
            return target.sample(n, generator=generator)
        except NotImplementedError:
            pass
    p = REFERENCES / f'reference_{target.name}_draws.npz'
    x = np.load(p)['x'].reshape(-1, target.dim)
    if getattr(target, 'reference_method', '') == 'nuts_sym':
        x = np.concatenate([x, -x])
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator).item()) if generator is not None else 0
    idx = np.random.default_rng(seed).choice(len(x), n, replace=len(x) < n)
    return torch.as_tensor(x[idx], dtype=torch.get_default_dtype(), device=torch.get_default_device())

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_default))

def _default(o):
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
