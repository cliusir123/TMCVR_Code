import argparse
import json
import time
import numpy as np
import torch
from common import RESULT_TAG, make, raw_dir, setup, truth
from workflow import config as C
from benchmark import BENCH_DICT
from methods.tgi import fit_dictionary, run_gti
from mcmc import make_generator
from train import load
from transports.ensemble import SharedFlow, replicate
from wallclock import run_snis
GRIDS = {'rwm': [0.6, 0.8, 0.9, 0.95, 0.99, 1.0], 'mala': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]}

def fit(flow, target, R, **kw):
    d1 = fit_dictionary(replicate(flow, 1), target, **{**BENCH_DICT, **kw})
    return {k: v.expand(R, *v.shape[1:]) if torch.is_tensor(v) and v.dim() >= 1 and (v.shape[0] == 1) else v for k, v in d1.items()}

def var_groups(chains, G, sd, Cn):
    m = chains.detach().cpu().numpy()
    Rp = m.shape[0] // G
    out = []
    for g in range(G):
        x = m[g * Rp:(g + 1) * Rp].reshape(-1, m.shape[-1])
        x = x[np.isfinite(x).all(1)]
        out.append(x.var(0, ddof=1) / Cn / sd ** 2)
    return np.array(out)

def mse_groups(chains, groups, G, mu, sd, Cn):
    m = chains.detach().cpu().numpy()
    Rp = m.shape[0] // G
    out = []
    for g in range(G):
        x = m[g * Rp:(g + 1) * Rp].reshape(-1, m.shape[-1])
        x = x[np.isfinite(x).all(1)]
        out.append(((x.mean(0) - mu) / sd) ** 2 + x.var(0, ddof=1) / Cn / sd ** 2)
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--what', required=True, choices=['gamma', 'dict'])
    ap.add_argument('--R', type=int, default=10)
    ap.add_argument('--N', type=float, default=1000000.0)
    ap.add_argument('--chain-len', type=float, default=10000.0)
    ap.add_argument('--n-select', nargs='+', type=float, default=[10000.0, 100000.0, 1000000.0])
    ap.add_argument('--K', nargs='+', type=int, default=[4])
    ap.add_argument('--configs', nargs='*', default=None, help='dictionary settings "K,n_select,n_mc" (overrides --K/--n-select)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--snis-chunk', type=float, default=2000000.0)
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    target = make(t)
    tr = truth(target)
    mu, sd = (np.array(tr['mean']), np.sqrt(np.array(tr['var'])))
    h = C.TARGETS[t]['headline']
    flow, _ = load(t)
    N, Rp = (int(args.N), args.R)
    Cn = max(1, int(round(N / args.chain_len)))
    kw = dict(burn_frac=0.2, burn_max=C.BURN, indep=dict(C.INDEP, delta=0.0), headline=h, lean=True, adapt='fixed')
    if args.what == 'gamma':
        out = {'target': t, 'R_pilot': Rp, 'N': N, 'chain_len': args.chain_len, 'headline': h}
        for s, grid in GRIDS.items():
            G = len(grid)
            R = G * Rp
            dfit = fit(flow, target, R)
            g0 = torch.tensor(grid).repeat_interleave(Rp)
            t0 = time.perf_counter()
            r = run_gti(target, SharedFlow(flow, R), dfit, s, N, Cn, generator=make_generator(21), gamma0=g0, **kw)
            mse = mse_groups(r['chain_fixed'], None, G, mu, sd, Cn)
            mse0 = mse_groups(r['chain_plain'], None, G, mu, sd, Cn)
            k = int(np.nanargmin(mse[:, h]))
            out[s] = {'grid': grid, 'mse_cv': mse.tolist(), 'mse_plain': mse0.tolist(), 'gamma': grid[k], 'seconds': time.perf_counter() - t0}
            print(f'[pilot {t}] {s}: ' + ' '.join((f'{g}:{v:.2e}' for g, v in zip(grid, mse[:, h]))) + f' -> gamma {grid[k]}', flush=True)
        (raw_dir(t) / f'gamma_pilot{RESULT_TAG}.json').write_text(json.dumps(out, indent=1))
    else:
        gp = raw_dir(t) / f'gamma_pilot{RESULT_TAG}.json'
        g_rwm = json.loads(gp.read_text())['rwm']['gamma'] if gp.exists() else 0.97
        out = {'target': t, 'R_pilot': Rp, 'N': N, 'chain_len': args.chain_len, 'headline': h, 'gamma_rwm': g_rwm, 'runs': []}
        configs = [tuple((float(v) for v in c.split(','))) for c in args.configs] if args.configs else [(K, ns, C.DICT['n_mc']) for K in args.K for ns in args.n_select]
        for K, ns, nmc in configs:
            K = int(K)
            t0 = time.perf_counter()
            dfit = fit(flow, target, Rp, K=K, n_select=int(ns), n_mc=int(nmc), n_resid=int(nmc))
            fit_sec = time.perf_counter() - t0
            r = run_snis(target, flow, dfit['sub'], N, Rp, make_generator(5), chunk=int(args.snis_chunk), blocks=100)

            def mse_rep(e, v):
                e, v = (e.cpu().numpy(), v.cpu().numpy())
                return ((e.mean(0) - mu) / sd) ** 2 + v.mean(0) / sd ** 2
            s_plain, s_cv = (mse_rep(r['plain'], r['var_plain']), mse_rep(r['cv'], r['var_cv']))
            rr = run_gti(target, SharedFlow(flow, Rp), dfit, 'rwm', N, Cn, generator=make_generator(21), gamma0=g_rwm, cv_reg=True, **kw)
            g = lambda k: mse_groups(rr[k], None, 1, mu, sd, Cn)[0]
            t_plain, t_fix, t_reg = (g('chain_plain'), g('chain_fixed'), g('chain_reg'))
            gv = lambda k: var_groups(rr[k], 1, sd, Cn)[0]
            v_plain, v_fix, v_reg = (gv('chain_plain'), gv('chain_fixed'), gv('chain_reg'))
            sv_plain, sv_cv = (r['var_plain'].cpu().numpy().mean(0), r['var_cv'].cpu().numpy().mean(0))
            row = {'K': K, 'n_select': ns, 'n_mc': nmc, 'terms': len(dfit['sub']), 'fit_seconds': fit_sec, 'var_ratio': {'snis': (sv_plain / sv_cv).tolist(), 'tgi_fixed': (v_plain / v_fix).tolist(), 'tgi_reg': (v_plain / v_reg).tolist()}, 'snis_vrf': (s_plain / s_cv).tolist(), 'tgi_vrf_fixed': (t_plain / t_fix).tolist(), 'tgi_vrf_reg': (t_plain / t_reg).tolist(), 'mse': {'snis_cv': s_cv.tolist(), 'tgi_fixed': t_fix.tolist(), 'tgi_reg': t_reg.tolist(), 'tgi_plain': t_plain.tolist()}}
            out['runs'].append(row)
            med = lambda a: float(np.median(np.asarray(a)))
            print(f"[dict {t}] K={K} n_select={ns:.0e} n_mc={nmc:.0e}: {len(dfit['sub'])} terms ({fit_sec:.0f}s) | VRF main fn: SNIS+CV {s_plain[h] / s_cv[h]:.0f}, TGI fixed {t_plain[h] / t_fix[h]:.0f}, TGI least-squares {t_plain[h] / t_reg[h]:.0f} | median over fns: {med(s_plain / s_cv):.0f} / {med(t_plain / t_fix):.0f} / {med(t_plain / t_reg):.0f} | variance-only main fn: {sv_plain[h] / sv_cv[h]:.0f} / {v_plain[h] / v_fix[h]:.0f} / {v_plain[h] / v_reg[h]:.0f}", flush=True)
            (raw_dir(t) / (args.out or f'dict_pilot{RESULT_TAG}.json')).write_text(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
