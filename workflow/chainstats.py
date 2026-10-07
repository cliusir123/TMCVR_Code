import argparse
import json
import time
import numpy as np
import torch
from common import RESULT_TAG, make, raw_dir, setup, truth
from workflow import config as C
from methods.ellipt import run_ellipt
from methods.neutra import run_neutra_multi
from methods.tgi import run_gti
from mcmc import make_generator
from mcmc.adaptation import J_LAMBDA, J_ROBUST
from train import load
from transports.ensemble import SharedFlow
from wallclock import run_snis, single_fit
N_BOOT, BLOCKS = (2000, 100)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--R', type=int, default=100)
    ap.add_argument('--N', type=float, default=1000000.0)
    ap.add_argument('--chain-len', type=float, default=10000.0)
    ap.add_argument('--snis-chunk', type=float, default=2000000.0)
    ap.add_argument('--neutra-L', type=int, default=None, help='default: results/raw/<target>/neutra_L.json')
    ap.add_argument('--neutra-eps-half', action='store_true')
    ap.add_argument('--only', nargs='*', default=None, help='subset of snis tgi_rwm tgi_mala tess neutra')
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    cfg = C.TARGETS[t]
    target = make(t)
    tr = truth(target)
    mu, sd = (np.array(tr['mean']), np.sqrt(np.array(tr['var'])))
    R, N = (args.R, int(args.N))
    Cn = max(1, int(round(N / args.chain_len)))
    h = cfg['headline']
    flow, _ = load(t)
    T = SharedFlow(flow, R)
    dfit = single_fit(flow, target, R)
    jpath, npath = (raw_dir(t) / f'chainstats{RESULT_TAG}.json', raw_dir(t) / f'chainstats{RESULT_TAG}.npz')
    out = json.loads(jpath.read_text()) if args.only and jpath.exists() else {'target': t, 'R': R, 'N': N, 'C': Cn, 'chain_len': args.chain_len, 'headline': h, 'J_lambda': J_LAMBDA, 'cv_terms': len(dfit['sub']), 'runs': []}
    arrays = dict(np.load(npath)) if args.only and npath.exists() else {}
    rng = np.random.default_rng(0)
    want = lambda k: args.only is None or k in args.only

    def ci(vals):
        lo, hi = np.nanpercentile(vals, [2.5, 97.5], axis=0)
        return (lo.tolist(), hi.tolist())

    def chain_stats(m, w):
        ok = np.isfinite(m).all(1) & (w > 0)
        m, w = (m[ok], w[ok] / w[ok].mean())

        def mse_of(mm, ww):
            g = (ww[..., None] * mm).sum(-2) / ww.sum(-1)[..., None]
            v = (ww[..., None] ** 2 * (mm - g[..., None, :]) ** 2).mean(-2) / Cn
            return (((g - mu) / sd) ** 2 + v / sd ** 2, g, v)
        mse, g, v = mse_of(m, w)
        boot = []
        for s in range(0, N_BOOT, 100):
            idx = rng.integers(0, m.shape[0], size=(100, m.shape[0]))
            boot.append(mse_of(m[idx], w[idx])[0])
        return (mse, np.abs(g - mu) / sd, np.sqrt(v) / sd, np.concatenate(boot))

    def rep_stats(e, var=None):
        ok = np.isfinite(e).all(1)
        e = e[ok]
        var = None if var is None else var[ok]

        def mse_of(ee, vv):
            b = ((ee.mean(-2) - mu) / sd) ** 2
            if vv is None:
                return np.nanmean((ee - mu) ** 2, -2) / sd ** 2
            return np.maximum(b - vv.mean(-2) / ee.shape[-2] / sd ** 2, 0) + vv.mean(-2) / sd ** 2
        mse = mse_of(e, var)
        idx = rng.integers(0, e.shape[0], size=(N_BOOT, e.shape[0]))
        boot = mse_of(e[idx], None if var is None else var[idx])
        return (mse, boot)

    def record(method, sec, est, chains=None, w=None, var=None, **extra):
        e = est.detach().cpu().numpy()
        rm_rep = np.sqrt(np.nanmean((e - mu) ** 2, 0)) / sd
        row = {'method': method, 'seconds_batch': sec, 'seconds': sec / R, 'rmse_rep': rm_rep.tolist(), 'bias_rep': (np.abs(np.nanmean(e, 0) - mu) / sd).tolist(), **extra}
        spe = sec / R
        if chains is not None:
            m = chains.detach().cpu().numpy().reshape(-1, chains.shape[-1])
            ww = np.ones(m.shape[0]) if w is None else w.detach().cpu().numpy().reshape(-1).astype(float)
            mse, bias_c, sd_c, boot = chain_stats(m, ww)
            row.update({'bias_chain': bias_c.tolist(), 'sd_chain': sd_c.tolist()})
            arrays[f'{method}_chainmeans'] = m.astype(np.float32)
            arrays[f'{method}_chainweights'] = ww.astype(np.float32)
        else:
            vv = None if var is None else var.detach().cpu().numpy()
            mse, boot = rep_stats(e, vv)
            if vv is not None:
                arrays[f'{method}_blockvar'] = vv.astype(np.float32)
        row['rmse'] = np.sqrt(mse).tolist()
        row['acc_per_s'] = (1.0 / (mse * spe)).tolist()
        row['acc_per_s_ci'] = ci(1.0 / (boot * spe))
        out['runs'] = [r for r in out['runs'] if r['method'] != method] + [row]
        arrays[f'{method}_est'] = e.astype(np.float32)
        lo, hi = row['acc_per_s_ci']
        print(f"[chainstats {t}] {method:14s} {sec:7.1f}s RMSE {row['rmse'][h]:.2e} (rep {rm_rep[h]:.2e}) acc/s {row['acc_per_s'][h]:.2e} [{lo[h]:.1e}, {hi[h]:.1e}]", flush=True)

    def save():
        jpath.write_text(json.dumps(out, indent=1))
        np.savez_compressed(npath, **arrays)
    if want('snis'):
        gen = make_generator(5)
        sync()
        t0 = time.perf_counter()
        r = run_snis(target, flow, dfit['sub'], N, R, gen, chunk=int(args.snis_chunk), blocks=BLOCKS)
        sync()
        sec = time.perf_counter() - t0
        mw = r['max_weight'].cpu().numpy()
        ex = {'steps': {'draws_per_estimate': N, 'blocks': BLOCKS, 'cost_per_estimate': N}, 'max_weight_median': float(np.median(mw)), 'max_weight_max': float(mw.max()), 'max_wN_median': float(np.median(mw) * N), 'max_wN_max': float(mw.max() * N)}
        record('SNIS', sec, r['plain'], var=r['var_plain'], **ex)
        record('SNIS+CV', sec, r['cv'], var=r['var_cv'], **ex)
        save()
    for key, sampler, name in (('tgi_rwm', 'rwm', 'TGI-RWM'), ('tgi_mala', 'mala', 'TGI-MALA')):
        if not want(key):
            continue
        gen = make_generator(11 if sampler == 'rwm' else 12)
        sync()
        t0 = time.perf_counter()
        gp = raw_dir(t) / f'gamma_pilot{RESULT_TAG}.json'
        gpil = json.loads(gp.read_text())[sampler]['gamma'] if gp.exists() else None
        akw = dict(gamma0=gpil, adapt='fixed') if gpil is not None else dict(gamma0=0.5, j_lr=0.05, j_robust=J_ROBUST, adapt='J')
        r = run_gti(target, T, dfit, sampler, N, Cn, generator=gen, headline=h, burn_frac=0.2, burn_max=C.BURN, indep=dict(C.INDEP, delta=0.0), lean=True, **akw)
        sync()
        sec = time.perf_counter() - t0
        acc = r['acc_chain'].detach().cpu().numpy()
        mrej = r['max_rej_chain'].detach().cpu().numpy()
        cost = torch.as_tensor(r['cost']).float()
        ex = {'gamma': float(torch.as_tensor(r['gamma']).float().median()), 'gamma_rule': 'pilot' if gpil is not None else 'J', 'acc_mean': float(acc.mean()), 'steps': {'chains_per_estimate': Cn, 'burn_steps_per_chain': int(r['n_burn']), 'kept_steps_per_chain': int(r['n_eval']), 'total_steps_per_chain': int(r['n_burn']) + int(r['n_eval']), 'kept_draws_per_estimate': Cn * int(r['n_eval']), 'cost_per_estimate': float(cost.mean())}}
        record(name, sec, r['plain'], r['chain_plain'], **ex)
        record(name + '+CV', sec, r['estimate_fixed'], r['chain_fixed'], **ex)
        arrays[f'{name}_acc_chain'] = acc.astype(np.float32)
        arrays[f'{name}_maxrej_chain'] = mrej.astype(np.float32)
        save()
    if want('tess'):
        gen = make_generator(13)
        sync()
        t0 = time.perf_counter()
        r = run_ellipt(target, T, N, Cn, dfit, burn_frac=0.2, burn_max=C.BURN, generator=gen)
        sync()
        sec = time.perf_counter() - t0
        it = r['chain_iters'].float()
        ex = {'steps': {'chains_per_estimate': Cn, 'candidates_per_chain': int(round(r['cost'] / Cn)), 'burn_candidates_per_chain': int(r['n_burn']), 'kept_iterations_per_chain_mean': float(it.mean()), 'kept_iterations_per_chain_min': float(it.min()), 'kept_iterations_per_chain_max': float(it.max()), 'kept_draws_per_estimate': float(it.mean()) * Cn, 'evals_per_iteration': float(r['evals_per_iter']), 'cost_per_estimate': float(r['cost'])}}
        record('TESS', sec, r['plain'], r['chain_plain'], r['chain_iters'], **ex)
        record('TESS+CV', sec, r['estimate_fixed'], r['chain_fixed'], r['chain_iters'], **ex)
        save()
    if want('neutra'):
        L = args.neutra_L
        if L is None:
            p = raw_dir(t) / f'neutra_L{RESULT_TAG}.json'
            L = json.loads(p.read_text())['L'] if p.exists() else None
        if L is None:
            print(f'[chainstats {t}] no NeuTra L (run workflow/neutra_pilot.py)', flush=True)
        else:
            for tag, scale in [('', 1.0)] + ([('[eps/2]', 0.5)] if args.neutra_eps_half else []):
                gen = make_generator(17)
                sync()
                t0 = time.perf_counter()
                r = run_neutra_multi(target, flow, N, R, chain_len=args.chain_len, generator=gen, accept_target=0.8, warmup_min=300, L_fixed=L, per_chain=True, eps_scale=scale)
                sync()
                sec = time.perf_counter() - t0
                n_bad = int((~torch.isfinite(r['zv_latent']).all(1)).sum())
                kept = r['chain_n'].float()
                cost = torch.as_tensor(r['cost']).float()
                ex = {'L': L, 'eps': float(r['eps'].mean()), 'acceptance': float(r['acceptance'].mean()), 'replicates_nonfinite': n_bad, 'steps': {'chains_per_estimate': Cn, 'warmup_iterations': int(r['n_warm']), 'kept_iterations_per_chain_mean': float(kept.mean()), 'kept_iterations_per_chain_min': float(kept.min()), 'kept_draws_per_estimate': float(kept.mean()) * Cn, 'leapfrog_steps_per_estimate': float(cost.mean()) / 2, 'cost_per_estimate': float(cost.mean())}}
                record('NeuTra' + tag, sec, r['plain'], r['chain_plain'], r['chain_n'], **ex)
                record('NeuTra' + tag + '+ZVz', sec, r['zv_latent'], r['chain_zv'], r['chain_n'], **ex)
                save()
if __name__ == '__main__':
    main()
