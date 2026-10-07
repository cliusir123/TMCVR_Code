import argparse
import json
import math
import time
import numpy as np
import torch
from common import flow_dir, make, raw_dir, setup
from workflow import config as C
from methods.ellipt import run_ellipt
from methods.gi import affine_transport
from methods.tgi import fit_dictionary, run_gti
from mcmc import make_generator
from mcmc.adaptation import J_ROBUST
from train import load
from transports.ensemble import replicate
BENCH_DICT = dict(K=C.DICT['K'], p=C.DICT['p'], lam=C.DICT['lam'], n_select=C.DICT['n_select'], n_mc=C.DICT['n_mc'], n_resid=C.DICT['n_mc'])

def done(path):
    return {json.loads(l)['key'] for l in path.read_text().splitlines()} if path.exists() else set()

def write(path, rows):
    import fcntl
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(''.join((json.dumps(r) + '\n' for r in rows)))
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)

def zv2(target, x, g):
    from methods.zvcv import fit_estimate_chunked
    F = target.F(x)
    n, d = x.shape
    if n <= 2 * (d + d * (d + 1) // 2) or not bool(torch.isfinite(g).all()):
        return F.mean(0)
    m, s = target.laplace()
    try:
        return fit_estimate_chunked(x, g, F, m, s, 2)[0]
    except RuntimeError:
        return F.mean(0)
CHAIN_METHODS = {'tgi_rwm': dict(kind='mh', sampler='rwm', on='flow', adapt='J', names=('TGI-RWM', 'TGI-RWM+CV'), theta='fixed'), 'tgi_rwm_d': dict(kind='mh', sampler='rwm', on='flow', adapt='J', names=('TGI-RWM-δ', 'TGI-RWM-δ+CV'), theta='fixed', indep=dict(delta=0.01, df=10)), 'tgi_rwm_g1': dict(kind='mh', sampler='rwm', on='flow', adapt='fixed', gamma0=1.0, names=('TGI-RWM-γ1', 'TGI-RWM-γ1+CV'), theta='fixed', indep=dict(delta=0.0, df=10), only=[1000000.0]), 'tgi_mala': dict(kind='mh', sampler='mala', on='flow', adapt='J', names=('TGI-MALA', 'TGI-MALA+CV'), theta='fixed'), 'gi_mala': dict(kind='mh', sampler='mala', on='laplace', adapt=('acc', 0.8), names=('GI-MALA', 'GI-MALA+CV'), theta='fit', bounds=(0.0001, 1.99)), 'gi_pmala': dict(kind='mh', sampler='pmala', on='laplace', adapt=('acc', 0.8), names=('GI-pMALA', 'GI-pMALA+CV'), theta='fit', bounds=(0.0001, 1.99)), 'gi_pmala_lgm': dict(kind='mh', sampler='pmala', on='laplace', precond='scalar', adapt=('acc', 0.8), names=('GI-pMALA-LGM', 'GI-pMALA-LGM+CV'), theta='fit', bounds=(0.0001, 1.99)), 'mala': dict(kind='mh', sampler='mala_plain', on='identity', adapt=('acc', 0.574), names=('MALA', None), gamma0=0.1, bounds=(1e-05, 10.0)), 'tess': dict(kind='ellipt', on='flow', names=('TESS', 'TESS+CV')), 'ellipt': dict(kind='ellipt', on='laplace', names=('Ellipt', 'Ellipt+CV'))}

def part_chains(t, target, path, method, only=None):
    cfg, spec = (C.TARGETS[t], CHAIN_METHODS[method])
    have = done(path)
    seed0 = 7919 + 1000 * list(CHAIN_METHODS).index(method)
    prefix = cfg.get('prefix') if spec['kind'] == 'mh' else None
    if 'prefix_methods' in cfg and method not in cfg['prefix_methods']:
        prefix = None
    if spec['on'] == 'flow':
        info = json.loads((flow_dir(t) / 'train.json').read_text())
        configs = []
        if prefix:
            configs = [(cfg['ckpts'][-1], max(prefix))]
        else:
            for k in cfg.get('bench_ckpts', cfg['ckpts']):
                configs += [(k, B) for B in cfg['gti_budgets']]
        only = only or spec.get('only')
        if only:
            configs = [(cfg['ckpts'][-1], B) for B in only]
    else:
        configs = [(None, max(prefix))] if prefix else [(None, B) for B in cfg['nuts_budgets']]
    cache = {}
    for k, B in configs:
        key = f'{method}|{k}|{B:.0e}'
        if key in have:
            continue
        if k not in cache:
            cache.clear()
            if spec['on'] == 'flow':
                flow, cost_train = load(t, k)
                T = replicate(flow, C.REPLICATES)
            else:
                cost_train = 0.0
                T = affine_transport(target, C.REPLICATES, identity=spec['on'] == 'identity')
            dfit = fit_dictionary(T, target, **BENCH_DICT) if spec['names'][1] else None
            cache[k] = (T, cost_train, dfit)
        T, cost_train, dfit = cache[k]
        n_chains = C.PREFIX_CHAINS if prefix else max(1, int(round(B / cfg['chain_len'])))
        gen = make_generator(seed0 + (k or 0) + int(math.log10(B) * 7))
        t0 = time.perf_counter()
        if spec['kind'] == 'mh':
            res = run_gti(target, T, dfit, spec['sampler'], B, n_chains, generator=gen, headline=cfg['headline'], gamma0=spec.get('gamma0', 0.5), j_lr=0.05, burn_frac=0.2, burn_max=C.BURN, j_robust=J_ROBUST, adapt=spec['adapt'], gamma_bounds=spec.get('bounds'), precond=spec.get('precond', 'diag'), snapshots=prefix, indep=spec.get('indep', C.indep(t) if spec['on'] == 'flow' else None))
        else:
            res = run_ellipt(target, T, B, n_chains, dfit, burn_frac=0.2, burn_max=C.BURN, generator=gen)
        if res is None:
            continue
        plain_name, cv_name = spec['names']
        parts = [(N, res if N == B else res.get('snapshots', {}).get(N)) for N in sorted(prefix)] if prefix else [(B, res)]
        rows = []
        for N, rs in parts:
            if rs is None:
                print(f'[benchmark {t}] {method}: no estimate at N = {N:.0e} (burn-in exceeds it)', flush=True)
                continue
            for e in range(C.REPLICATES):
                base = {'key': f'{method}|{k}|{N:.0e}', 'ckpt': k, 'budget': N, 'cost_train': cost_train, 'total_cost': cost_train + N, 'replicate': e, 'n_chains': n_chains}
                if prefix:
                    base['prefix'] = True
                pick = lambda v: float(v[e]) if torch.is_tensor(v) and v.dim() else float(v)
                if 'cost_eff' in rs:
                    base['cost_eff'] = pick(rs['cost_eff'])
                if 'indep' in res:
                    base.update(cost_actual=pick(res['cost']), n_ind=pick(res['indep']['n_ind']), acc_ind=pick(res['indep']['acc_ind']))
                if 'max_rej' in res:
                    base['max_rej'] = pick(res['max_rej'])
                if spec['on'] == 'flow':
                    base['ess_is'] = info['ckpts'][str(k)]['ess_is']
                if spec['kind'] == 'mh':
                    base.update(gamma=float(res['gamma'][e]), acceptance=float(res['acceptance'][e]))
                rows.append({**base, 'method': plain_name, 'estimate': rs['plain'][e].tolist()})
                if cv_name:
                    est = rs['estimate'] if spec.get('theta') == 'fit' else rs['estimate_fixed']
                    row = {**base, 'method': cv_name, 'estimate': est[e].tolist()}
                    if spec.get('theta') == 'fit':
                        row['estimate_fixed'] = rs['estimate_fixed'][e].tolist()
                    rows.append(row)
        write(path, rows)
        extra = f"gamma {res['gamma'].median().item():.3g}, acceptance {res['acceptance'].mean().item():.2f}" if spec['kind'] == 'mh' else f"{res['evals_per_iter']:.2f} evaluations / iteration"
        print(f'[benchmark {t}] {method} ckpt {k} budget {B:.0e}: {extra} ({time.perf_counter() - t0:.0f}s)', flush=True)

def part_nuts(t, target, path):
    from methods.nuts import run_nuts
    cfg = C.TARGETS[t]
    have = done(path)
    multi = cfg['scheme'] == 'Aa'
    K = C.MULTISTART['chains'] if multi else 1
    prefix = cfg.get('prefix')
    for B in [max(prefix)] if prefix else cfg['nuts_budgets']:
        key = f'nuts|{B:.0e}'
        if key in have:
            continue
        seed = int(B) % 997 + 11
        init = np.random.default_rng(seed + 17).uniform(-2.0, 2.0, (C.REPLICATES * K, target.dim)) if multi else None
        runs, _ = run_nuts(target.name, target, C.REPLICATES * K, B / K, seed=seed, init=init, chain_method='vectorized' if multi or cfg.get('nuts_vectorized') else 'parallel', cost_guess=cfg.get('ms_cost_guess', 30.0))
        rows = []
        for N in sorted(prefix) if prefix else [B]:
            for e in range(C.REPLICATES):
                cut = [(r, int(np.searchsorted(r['cum_cost'], N / K, side='right')) if prefix else len(r['x'])) for r in runs[e * K:(e + 1) * K]]
                rr = [(r, n) for r, n in cut if n > 0]
                if not rr:
                    continue
                x = torch.cat([torch.as_tensor(r['x'][:n]) for r, n in rr])
                g = torch.cat([torch.as_tensor(r['grad'][:n]) for r, n in rr])
                cost = sum((r['cost_warm'] + (r['cum_cost'][n - 1] - r['cost_warm'] if n else 0.0) for r, n in cut))
                eff = sum((r['cum_cost'][n - 1] - r['cost_warm'] for r, n in rr))
                base = {'key': f'nuts|{N:.0e}', 'budget': N, 'total_cost': cost, 'cost_eff': float(eff), 'replicate': e, 'n_chains': K, 'n_draws': int(x.shape[0])}
                if prefix:
                    base['prefix'] = True
                rows.append({**base, 'method': 'NUTS', 'estimate': target.F(x).mean(0).tolist()})
                rows.append({**base, 'method': 'NUTS+ZV2', 'estimate': zv2(target, x, g).tolist()})
        write(path, rows)
        print(f'[benchmark {t}] NUTS budget {B:.0e} ({K} chain(s) per replicate)', flush=True)

def part_snis(t, target, path, only=None):
    from wallclock import run_snis, single_fit
    cfg = C.TARGETS[t]
    have = done(path)
    configs = [(cfg['ckpts'][-1], B) for B in only] if only else [(k, B) for k in cfg.get('bench_ckpts', cfg['ckpts']) for B in cfg['gti_budgets']]
    fits = {}
    for k, B in configs:
        key = f'snis|{k}|{B:.0e}'
        if key in have:
            continue
        flow, cost_train = load(t, k)
        if k not in fits:
            fits = {k: single_fit(flow, target, C.REPLICATES)}
        t0 = time.perf_counter()
        out = run_snis(target, flow, fits[k]['sub'], int(B), C.REPLICATES, make_generator(53 + k + int(math.log10(B) * 7)))
        rows = []
        for e in range(C.REPLICATES):
            base = {'key': key, 'ckpt': k, 'budget': B, 'cost_train': cost_train, 'replicate': e, 'n_chains': 0, 'total_cost': cost_train + B, 'max_weight': float(out['max_weight'][e])}
            rows.append({**base, 'method': 'SNIS', 'estimate': out['plain'][e].tolist()})
            rows.append({**base, 'method': 'SNIS+CV', 'estimate': out['cv'][e].tolist()})
        write(path, rows)
        print(f"[benchmark {t}] SNIS ckpt {k} budget {B:.0e}: max w*N median {(out['max_weight'] * B).median().item():.1f} ({time.perf_counter() - t0:.0f}s)", flush=True)

def part_neutra(t, target, path, only=None):
    from methods.neutra import run_neutra_multi
    cfg = C.TARGETS[t]
    have = done(path)
    configs = [(cfg['ckpts'][-1], B) for B in only] if only else [(k, B) for k in cfg.get('bench_ckpts', cfg['ckpts']) for B in cfg['gti_budgets']]
    L = json.loads((raw_dir(t) / 'neutra_L_c5k.json').read_text())['L']
    for k, B in configs:
        key = f'neutra|{k}|{B:.0e}'
        if key in have:
            continue
        flow, cost_train = load(t, k)
        out = run_neutra_multi(target, flow, B, C.REPLICATES, chain_len=cfg['chain_len'], generator=make_generator(31 + k + int(math.log10(B) * 7)), L_fixed=L, jitter=True, accept_target=0.8, warmup_min=300)
        rows = []
        for e in range(C.REPLICATES):
            base = {'key': key, 'ckpt': k, 'budget': B, 'cost_train': cost_train, 'replicate': e, 'n_chains': out['C'], 'eps': float(out['eps'][e]), 'L': float(out['L'][e]), 'acceptance': float(out['acceptance'][e])}
            rows.append({**base, 'method': 'NeuTra', 'total_cost': cost_train + float(out['cost'][e]), 'estimate': out['plain'][e].tolist()})
            rows.append({**base, 'method': 'NeuTra+ZVz', 'total_cost': cost_train + float(out['cost'][e]), 'estimate': out['zv_latent'][e].tolist()})
        write(path, rows)
        print(f"[benchmark {t}] NeuTra ckpt {k} budget {B:.0e}: {out['C']} chains, eps {out['eps'].median().item():.2f}, L {out['L'].median().item():.0f}, acceptance {out['acceptance'].mean().item():.2f} ({out['seconds']:.0f}s)", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--methods', nargs='+', default=None, choices=[*CHAIN_METHODS, 'nuts', 'neutra', 'snis'], help='default: C.BENCH_METHODS')
    ap.add_argument('--only-budgets', nargs='+', type=float, default=None, help='flow-based methods: only the last checkpoint at these budgets')
    args = ap.parse_args()
    setup(C.DTYPE)
    target = make(args.target)
    path = raw_dir(args.target) / 'benchmark.jsonl'
    methods = args.methods or C.BENCH_METHODS
    if C.TARGETS[args.target]['skip_pmala']:
        methods = [m for m in methods if m not in ('gi_pmala', 'gi_pmala_lgm')]
    for m in methods:
        if m == 'nuts':
            part_nuts(args.target, target, path)
        elif m == 'neutra':
            part_neutra(args.target, target, path, only=args.only_budgets)
        elif m == 'snis':
            part_snis(args.target, target, path, only=args.only_budgets)
        else:
            part_chains(args.target, target, path, m, only=args.only_budgets)
if __name__ == '__main__':
    main()
