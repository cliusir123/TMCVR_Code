import argparse
import json
import types
import numpy as np
import torch
from common import make, raw_dir, setup, truth
from workflow import config as C
from experiments.runner import run_transport
from hermite import HermiteDictionary, fit_dictionary, full, magnitude, structural
from mcmc import GroupAccAdapter, JAdapter, LatentTarget, make_generator, make_proposal, run_chains
from methods.gi import affine_transport
from mcmc.adaptation import J_ROBUST
from train import load

def protocol(gammas):
    return types.SimpleNamespace(CHAINS=C.CHAINS, N_STEPS=C.N_STEPS, FLOW_BURN=C.BURN, GAMMAS=list(gammas), SAMPLERS=C.SAMPLERS, SEED=0, N_SELECT=C.DICT['n_select'], N_MC=C.DICT['n_mc'], N_RESID=C.DICT['n_mc'], N_DIAG=20000, FLOW_STATE_THIN=20)
GI_SAMPLERS, GI_ACC = (['mala', 'pmala'], 0.8)

def transport(t, target, name, ckpt=None):
    if name == 'flow':
        return (load(t)[0], False, 'flow') if ckpt is None else (load(t, ckpt)[0], False, f'flow_ck{ckpt}')
    if name == 'laplace':
        return (affine_transport(target, 1), False, 'Laplace affine')
    family, value = C.TARGETS[t]['analytic'][name]
    return (getattr(target, f'{family}_transport')(value), True, f'{family}={value:g}')

def j_gamma(target, T, sampler, indep=None):
    D = HermiteDictionary(target.dim, C.DICT['K'], max_support=C.DICT['p'])
    fit = fit_dictionary(D, T, [f for _, f, _ in target.test_functions], {'a': magnitude(C.DICT['K'], C.DICT['p'], C.DICT['lam'])}, C.DICT['n_select'], C.DICT['n_mc'], C.DICT['n_mc'])
    j = C.TARGETS[target.name]['headline']
    lat = LatentTarget(target, T)
    out = []
    for seed in range(C.J_ADAPT_RUNS):
        gen = make_generator(100 + seed)
        z0 = torch.randn(C.CHAINS, target.dim, generator=gen)
        r = run_chains(lat, make_proposal(sampler), z0, 0.5, n=1, burn=C.BURN, adapter=JAdapter(fit.V0('a')[j].item(), 0.05, (0.001, 1.0), robust=J_ROBUST), generator=gen, indep=indep)
        g = r['gamma']
        out.append(float(g.flatten()[0]) if torch.is_tensor(g) else float(g))
    return out

def acc_gamma(target, T, sampler):
    lat = LatentTarget(target, T)
    out = []
    for seed in range(C.J_ADAPT_RUNS):
        gen = make_generator(100 + seed)
        z0 = torch.randn(C.CHAINS, target.dim, generator=gen)
        ad = GroupAccAdapter(torch.zeros(C.CHAINS, dtype=torch.long), GI_ACC, bounds=(0.0001, 1.99))
        r = run_chains(lat, make_proposal(sampler), z0, torch.full((C.CHAINS,), 0.5), n=1, burn=C.BURN, adapter=ad, per_chain=True, generator=gen)
        out.append(float(r['gamma'][0]))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--exp', required=True, choices=['gamma', 'pruning', 'vrf'])
    ap.add_argument('--transport', nargs='+', default=None, help='flow, laplace (GI-MALA / GI-pMALA, exp vrf) and/or the analytic transports of the target (default: all)')
    ap.add_argument('--sampler', nargs='+', default=None, help='default: SAMPLERS (laplace: GI_SAMPLERS)')
    ap.add_argument('--delta', type=float, default=None, help="defensive independence step with this delta instead of the target's (TGI only); outputs <exp>_<transport>_d_<sampler>.jsonl (delta 0: no suffix)")
    ap.add_argument('--ckpt', type=int, default=None, help='flow checkpoint (default: the last); outputs <exp>_flow_ck<ckpt>_<sampler>.jsonl')
    ap.add_argument('--gamma1', action='store_true', help='exp vrf: also gamma = 1 (no adaptation) in the same batched run; outputs <exp>_<transport>_g1_<sampler>.jsonl')
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    target = make(t)
    tr = truth(target)
    means, var_pi = (torch.tensor(tr['mean']), torch.tensor(tr['var']))
    names = args.transport or ['flow', *C.TARGETS[t]['analytic']] + (['laplace'] if args.exp == 'vrf' else [])
    jpath = raw_dir(t) / 'j_gamma.json'
    for name in names:
        T, exact, label = transport(t, target, name, args.ckpt)
        indep = None if name == 'laplace' else C.indep(t)
        tag = label if name == 'flow' else name
        if args.delta is not None and name != 'laplace':
            indep = dict(C.INDEP, delta=args.delta)
            if args.delta > 0:
                tag = f'{tag}_d'
        base_tag = tag
        if args.gamma1:
            tag = f'{tag}_g1'
        samplers = args.sampler or (GI_SAMPLERS if name == 'laplace' else C.SAMPLERS)
        if C.TARGETS[t]['skip_pmala']:
            samplers = [s for s in samplers if s != 'pmala']
        for s in samplers:
            out = raw_dir(t) / f'{args.exp}_{tag}_{s}.jsonl'
            jg = None
            if args.exp in ('gamma', 'vrf'):
                js = json.loads(jpath.read_text()) if jpath.exists() else {}
                jg = (js.get(f'{base_tag}:{s}') if args.gamma1 else None) or (acc_gamma(target, T, s) if name == 'laplace' else j_gamma(target, T, s, indep))
                js = json.loads(jpath.read_text()) if jpath.exists() else {}
                js[f'{tag}:{s}'] = jg
                jpath.write_text(json.dumps(js, indent=1))
                print(f"[chains {t}] {name} {s}: {('acceptance' if name == 'laplace' else 'J')}'s gamma {np.round(jg, 3).tolist()}", flush=True)
            if args.exp == 'gamma':
                E = protocol(C.GAMMA_GRID)
                D = HermiteDictionary(target.dim, C.DICT['K'], max_support=C.DICT['p'])
                arms = {'main': magnitude(C.DICT['K'], C.DICT['p'], C.DICT['lam'])}
            elif args.exp == 'vrf':
                E = protocol([float(np.median(jg))] + ([1.0] if args.gamma1 else []))
                D = HermiteDictionary(target.dim, C.DICT['K'], max_support=C.DICT['p'])
                arms = {'main': magnitude(C.DICT['K'], C.DICT['p'], C.DICT['lam'])}
            else:
                E = protocol(C.PRUNE_GRID)
                small = target.dim <= 10
                D = HermiteDictionary(target.dim, max(C.PRUNE_KS), max_support=None if small else 2)
                arms = {}
                for K in C.PRUNE_KS:
                    if small:
                        arms[f'full{K}'] = full(K)
                    arms[f'p1_K{K}'] = structural(K, 1)
                    arms[f'p2_K{K}'] = structural(K, 2)
                    arms[f'l2_K{K}'] = magnitude(K, 2, 2.0)
                    arms[f'llog_K{K}'] = magnitude(K, 2, '2log')
            print(f'[chains {t}] {args.exp} {name} ({label}) {s}', flush=True)
            run_transport(target, T, exact, D, arms, out, E, var_pi, means, (0.01, 2.0), samplers=[s], extra={'family': 'analytic' if exact else 'flow', 'label': label, 'target': t, 'param': 0, 'experiment': args.exp, 'Ks': C.PRUNE_KS if args.exp == 'pruning' else None}, x0_fn=None, indep=indep)
if __name__ == '__main__':
    main()
