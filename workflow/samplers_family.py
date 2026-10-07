import argparse
import json
import time
import numpy as np
import torch
from common import make, raw_dir, setup, truth
from workflow import config as C
from mcmc import LatentTarget, make_generator, make_proposal, run_chains
from transports.diagnostics import delta_dispersion, ess_is, fisher
GRIDS = {'banana': ('shear', [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]), 'funnel': ('scale', [0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]), 'gmix_d10_h6': ('scale', [0.9995, 0.999, 0.997, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7])}
GAMMAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]
CHAINS, BURN, STEPS = (100, 500, 4000)
SAMPLERS = ['rwm', 'mala', 'pmala']

class MeanRecorder:

    def __init__(self, target, j):
        self.target, self.j, self.S, self.n = (target, j, 0.0, 0)

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        self.S = self.S + self.target.F(s.x)[:, self.j]
        self.n += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(GRIDS))
    ap.add_argument('--main-fn', action='store_true', help='the main test function (config main_fn) instead of the headline, without pMALA; -> samplers_family_main.json')
    a = ap.parse_args()
    t = a.target
    setup(C.DTYPE)
    target = make(t)
    tr = truth(target)
    j = C.TARGETS[t]['main_fn'] if a.main_fn else C.TARGETS[t]['headline']
    samplers = ['rwm', 'mala'] if a.main_fn else SAMPLERS
    fname = 'samplers_family_main.json' if a.main_fn else 'samplers_family.json'
    var = tr['var'][j]
    family, values = GRIDS[t]
    out = {'family': family, 'gammas': GAMMAS, 'chains': CHAINS, 'burn': BURN, 'steps': STEPS, 'headline': j, 'members': []}
    gen0 = torch.Generator(device=torch.get_default_device()).manual_seed(11)
    for val in values:
        T = getattr(target, f'{family}_transport')(val)
        lat = LatentTarget(target, T)
        zx = T.sample_latent(20000, generator=gen0)
        row = {'param': val, 'fisher': float(fisher(lat, zx)), 'ess_is': float(ess_is(target, T, 20000)), 'dispersion': float(delta_dispersion(lat, zx[:5000]))}
        for s in samplers:
            t0 = time.perf_counter()
            prop = make_proposal(s)
            gen = make_generator(2026)
            holder = {}

            def factory(g):
                holder['r'] = MeanRecorder(target, j)
                return holder['r']
            res = run_chains(lat, prop, torch.randn(CHAINS, target.dim, generator=gen), torch.tensor(GAMMAS), n=STEPS, burn=BURN, generator=gen, recorder_factory=factory)
            means = (holder['r'].S / holder['r'].n).reshape(len(GAMMAS), CHAINS)
            ess_n = (var / (STEPS * means.var(1, unbiased=True))).cpu().numpy()
            per_cost = ess_n / prop.cost(target.dim)
            row[s] = {'ess_n': ess_n.tolist(), 'ess_per_cost': per_cost.tolist(), 'best_ess_per_cost': float(np.nanmax(per_cost)), 'best_gamma': float(GAMMAS[int(np.nanargmax(per_cost))]), 'acc': res['acc'].reshape(len(GAMMAS), CHAINS).mean(1).tolist()}
            print(f"[samplers {t}] {family}={val:g} F={row['fisher']:.3g} {s}: best ESS/cost {row[s]['best_ess_per_cost']:.3g} at gamma {row[s]['best_gamma']:g} ({time.perf_counter() - t0:.0f}s)", flush=True)
        out['members'].append(row)
        (raw_dir(t) / fname).write_text(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
