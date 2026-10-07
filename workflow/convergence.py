import argparse
import numpy as np
import torch
from common import make, raw_dir, setup
from workflow import config as C
from methods.gi import affine_transport
from methods.tgi import EnsembleRecorder, fit_dictionary
from mcmc import GroupAccAdapter, GroupJAdapter, LatentTarget, make_generator, make_proposal, run_chains
from mcmc.adaptation import J_ROBUST
from train import load
from transports.ensemble import replicate

class TraceRecorder(EnsembleRecorder):

    def __init__(self, *a, j=0, **k):
        super().__init__(*a, **k)
        self.j, self.trace = (j, [])

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        D = self.sub
        Gz, Gp = (self._G(D.he_table(s.z)), self._G(D.he_table(sp.z)))
        EG = self._G(D.cond_table(m, tau))
        F = self.target.F(s.x)
        H1 = alpha[:, None] * (Gp - Gz)
        H2 = Gp - EG
        j = self.j
        w = torch.ones_like(alpha) if mask is None else mask
        self.trace.append(torch.stack([F[:, j] * w, H1[:, j] * w, H2[:, j] * w, w], -1).cpu())
        self.t += 1

def running(F, H1, H2, theta, w=None):
    t = np.arange(1, len(F) + 1) if w is None else np.maximum(np.cumsum(w), 1)
    mF = np.cumsum(F) / t
    if theta == 'fixed':
        return (mF, np.cumsum(F + H1 - H2) / t)
    c = lambda x: np.cumsum(x) / t
    m1, m2 = (c(H1), c(H2))
    k11, k22, k12 = (c(H1 * H1) - m1 * m1, c(H2 * H2) - m2 * m2, c(H1 * H2) - m1 * m2)
    q1, q2 = (c(F * H1) - mF * m1, c(F * H2) - mF * m2)
    det = k11 * k22 - k12 ** 2
    ok = det > 1e-300
    safe = np.where(ok, det, 1.0)
    th1 = np.where(ok, -(k22 * q1 - k12 * q2) / safe, 0.0)
    th2 = np.where(ok, -(k11 * q2 - k12 * q1) / safe, 0.0)
    est = mF + th1 * m1 + th2 * m2
    est[:20] = np.nan
    return (mF, est)

def mh_traces(target, T, sampler, adapt, j, cfg, seed, theta):
    R, n, burn = (C.CONV['chains'], C.CONV['steps'], C.CONV['burn'])
    dfit = fit_dictionary(T, target, K=C.DICT['K'], p=C.DICT['p'], lam=C.DICT['lam'], n_select=C.DICT['n_select'], n_mc=C.DICT['n_mc'], n_resid=C.DICT['n_mc'])
    groups = torch.zeros(R, dtype=torch.long)
    if adapt == 'J':
        adapter = GroupJAdapter(groups, dfit['V0'][:, j], 0.05, robust=J_ROBUST)
    else:
        adapter = GroupAccAdapter(groups, adapt[1], bounds=(0.0001, 1.99))
    holder = {}

    def factory(g):
        holder['r'] = TraceRecorder(target, dfit['sub'], dfit['coef'], g, 1, R, n, j=j)
        return holder['r']
    gen = make_generator(seed)
    out = run_chains(LatentTarget(target, T), make_proposal(sampler), torch.randn(R, target.dim, generator=gen), torch.full((R,), 0.5), n=n, burn=burn, adapter=adapter, per_chain=True, generator=gen, recorder_factory=factory, indep=C.indep(target.name) if adapt == 'J' else None)
    tr = torch.stack(holder['r'].trace, 1).numpy()
    plain, cv = zip(*(running(*tr[r, :, :3].T, theta, w=tr[r, :, 3]) for r in range(R)))
    return (np.array(plain), np.array(cv), float(out['gamma'][0]))

def nuts_traces(target, j, seed):
    from methods.nuts import run_nuts
    from methods.zvcv import fit_estimate_chunked
    R, n = (C.CONV['chains'], C.CONV['steps'])
    runs, _ = run_nuts(target.name, target, R, C.CONV['nuts_budget'], seed=seed)
    n = min(n, *(r['x'].shape[0] for r in runs))
    m, s = target.laplace()
    ts = np.unique(np.geomspace(min(200, n), n, 50).astype(int))
    plain, cv = ([], [])
    for r in runs:
        x = torch.as_tensor(r['x'][:n])
        g = torch.as_tensor(r['grad'][:n])
        F = target.F(x)[:, j].cpu().numpy()
        plain.append(np.cumsum(F) / np.arange(1, n + 1))
        z = []
        for t in ts:
            try:
                z.append(float(fit_estimate_chunked(x[:t], g[:t], target.F(x[:t]), m, s, 2)[0][j]))
            except RuntimeError:
                z.append(np.nan)
        cv.append(np.interp(np.arange(1, n + 1), ts, z, left=np.nan))
    return (np.array(plain), np.array(cv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    t = ap.parse_args().target
    setup(C.DTYPE)
    target = make(t)
    j = C.TARGETS[t]['headline']
    flow, _ = load(t)
    res = {}
    p, c, g = mh_traces(target, replicate(flow, 1), 'rwm', 'J', j, C.TARGETS[t], 1, 'fixed')
    res.update(tgi_plain=p, tgi_cv=c, tgi_gamma=g)
    print(f'[convergence {t}] TGI-RWM gamma {g:.3f}', flush=True)
    p, c, g = mh_traces(target, affine_transport(target, 1), 'mala', ('acc', 0.8), j, C.TARGETS[t], 2, 'fit')
    res.update(gi_plain=p, gi_cv=c, gi_gamma=g)
    print(f'[convergence {t}] GI-MALA gamma {g:.3f}', flush=True)
    p, c = nuts_traces(target, j, 3)
    res.update(nuts_plain=p, nuts_cv=c)
    print(f'[convergence {t}] NUTS done', flush=True)
    np.savez(raw_dir(t) / 'convergence.npz', **res)
if __name__ == '__main__':
    main()
