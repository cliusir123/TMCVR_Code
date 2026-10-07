import json
import numpy as np
import torch
from common import make, raw_dir, setup
from workflow import config as C
N, REPEATS = (20000, 100)

def ess(lw):
    w = torch.exp(lw - lw.max())
    return float(w.sum() ** 2 / (w * w).sum() / lw.numel())

def main():
    setup(C.DTYPE)
    t = 'banana'
    target = make(t)
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(7)
    members = json.loads((raw_dir(t) / 'samplers_family_main.json').read_text())['members']
    out = {}
    for m in members:
        c = m['param']
        T = target.shear_transport(c)
        lws = []
        with torch.no_grad():
            for _ in range(N * REPEATS // 100000):
                z = torch.randn(100000, target.dim, generator=gen)
                x, ld = T(z)
                lws.append((target.log_prob(x) + ld + 0.5 * (z * z).sum(-1)).double().cpu())
        lw = torch.cat(lws)
        e = [ess(lw[i * N:(i + 1) * N]) for i in range(REPEATS)]
        out[c] = dict(median=float(np.median(e)), q10=float(np.quantile(e, 0.1)), q90=float(np.quantile(e, 0.9)))
        print(f"c={c:<6} ESS_IS/N median {out[c]['median']:.3f} [{out[c]['q10']:.3f}, {out[c]['q90']:.3f}]", flush=True)
    (raw_dir(t) / 'family_ess_is_repeats.json').write_text(json.dumps({'n': N, 'repeats': REPEATS, 'members': out}, indent=1))
if __name__ == '__main__':
    main()
