import argparse
import copy
import time
import numpy as np
import torch
from common import flow_dir, make, save_json, setup
from workflow import config as C
from transports.ensemble import EnsembleFlow
from transports.linear import FixedLinear
from transports.training import WhitenedTarget, flow_diagnostics, train_combined, train_reverse_kl

def group_chains(means, thr):
    K = means.shape[0]
    lab = list(range(K))

    def root(i):
        while lab[i] != i:
            i = lab[i]
        return i
    D = torch.cdist(means, means)
    for i in range(K):
        for j in range(i + 1, K):
            if D[i, j] < thr:
                lab[root(j)] = root(i)
    return [root(i) for i in range(K)]

def multistart_pilot(target, cfg, seed):
    from methods.nuts import run_nuts
    MS = {**C.MULTISTART, **cfg.get('multistart', {})}
    K, d, w = (MS['chains'], target.dim, MS.get('init_width', 2.0))
    init = np.random.default_rng(seed).uniform(-w, w, (K, d))
    runs, _ = run_nuts(target.name, target, K, cfg['pilot'] / K, seed=seed, init=init, chain_method='vectorized', cost_guess=cfg['ms_cost_guess'])
    n = min((r['x'].shape[0] for r in runs))
    if n < 2 * d:
        raise RuntimeError(f'pilot too short ({n} draws per chain)')
    xs = [torch.as_tensor(r['x'][:n]) for r in runs]
    Cw = torch.stack([torch.cov(x.T) for x in xs]).mean(0)
    L = torch.linalg.cholesky(Cw + 1e-10 * torch.eye(d))
    Li = torch.linalg.inv(L)
    lab = group_chains(torch.stack([x.mean(0) for x in xs]) @ Li.T, MS['group_threshold'])
    groups = sorted(set(lab))
    g = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
    per = K * n // len(groups)
    parts = []
    for c in groups:
        xc = torch.cat([x for x, l in zip(xs, lab) if l == c])
        parts.append(xc[torch.randint(xc.shape[0], (per,), generator=g)])
    pilot = torch.cat(parts) @ Li.T
    info = {'pilot_cost': float(sum((r['cost_warm'] + r['cost_sampling'] for r in runs))), 'pilot_draws_per_chain': n, 'pilot_chains_per_mode': [lab.count(c) for c in groups]}
    return (L, pilot.unsqueeze(0), info)

def train(t, ckpt0_only=False, variant=None):
    cfg = C.TARGETS[t]
    if variant == 'B':
        cfg = {**cfg, 'tail': False, 'train': {}, 'ckpts': [0, 1000, 10000], 'scheme': 'Ba' if cfg['scheme'] == 'Aa' else 'B'}
    target = make(t)
    TR = {**C.TRAIN, **cfg.get('train', {})}
    TL = {**C.TAIL, **cfg.get('tail_cfg', {})}
    A = {**C.ADAPTIVE, **cfg.get('adaptive', {})}
    seed = TR['seed']
    torch.manual_seed(seed)
    m, s = target.laplace()
    info = {'target': t, 'scheme': cfg['scheme'], 'arch': 'FINE' if cfg['arch'] is C.FINE else 'SMALL' if cfg['arch'] is C.SMALL else cfg['arch'], 'steps': TR['steps'], 'batch': TR['batch'], 'pilot_cost': 0.0}
    L = pilot = None
    if cfg['scheme'] in ('Aa', 'Ba'):
        L, pilot, pinfo = multistart_pilot(target, cfg, seed + 1)
        info.update(pinfo)
        pilot = pilot.to(torch.get_default_device())
        m, s = (pilot.mean(1, keepdim=True), pilot.std(1).unsqueeze(1))
        L = L.unsqueeze(0).to(torch.get_default_device())
    flow = EnsembleFlow(1, target.dim, m, s, **cfg['arch'])
    ckpts = {}

    def keep(step, fl):
        if step in cfg['ckpts']:
            f = copy.deepcopy(fl).requires_grad_(False)
            ckpts[step] = FixedLinear(f, L) if L is not None else f
    keep(0, flow)
    if ckpt0_only:
        import json
        out = flow_dir(t)
        old = json.loads((out / 'train.json').read_text())
        f = ckpts[0]
        ess = flow_diagnostics(target, f)['ess_is'][0]
        torch.save(f, out / 'ckpt_0.pt')
        old['ckpts']['0'] = {'cost': info['pilot_cost'], 'ess_is': ess}
        old['ckpts'] = dict(sorted(old['ckpts'].items(), key=lambda kv: int(kv[0])))
        save_json(out / 'train.json', old)
        print(f"[train {t}] ckpt 0: training cost {info['pilot_cost']:.2e}, ESS_IS {ess:.3f}")
        return
    t0 = time.perf_counter()
    common = dict(steps=TR['steps'], batch=TR['batch'], lr=TR['lr'], lr_min_ratio=TR['lr_min_ratio'], clip=TR['clip'], callback=keep, tail=TL if cfg.get('tail', True) else None)
    info['tail'] = TL if cfg.get('tail', True) else None
    if cfg['scheme'] == 'B':
        train_reverse_kl(flow, target, **common)
        per_step = 2.0 * TR['batch']
    elif cfg['scheme'] == 'Ba':
        train_combined(flow, WhitenedTarget(target, L), pilot=pilot, adaptive=False, rkl_weight=1.0, fkl_weight=1.0, seed=seed + 3, **common)
        per_step = 2.0 * TR['batch']
    else:
        tt = WhitenedTarget(target, L) if L is not None else target
        r = train_combined(flow, tt, pilot=pilot, adaptive=pilot is not None, pilot_steps=max(50, int(0.05 * TR['steps'])), chain_kind='gti', flush=True, rkl_weight=1.0, fkl_weight=1.0, chains=A['chains'], refresh=A['refresh'], chain_steps=A['chain_steps'], buffer_size=A['buffer'], acc_target=A['acc_target'], seed=seed + 3, anchor=cfg.get('anchor', C.MULTISTART['anchor']) if L is not None else 0.0, **common)
        info['chain_gamma'] = float(r['chain_gamma'][0])
        info['chain_acceptance'] = float(r['chain_acceptance'][0])
        per_step = 2.0 * TR['batch'] + A['chains'] * (A['chain_steps'] + 1) / A['refresh']
    info['seconds'] = time.perf_counter() - t0
    out = flow_dir(t)
    out.mkdir(parents=True, exist_ok=True)
    info['ckpts'] = {}
    for k, f in sorted(ckpts.items()):
        cost = info['pilot_cost'] + per_step * k
        if cfg.get('tail', True):
            cost += 2.0 * TL['batch'] * max(0, k - int(TL['start'] * TR['steps']))
        ess = flow_diagnostics(target, f)['ess_is'][0]
        torch.save(f, out / f'ckpt_{k}.pt')
        info['ckpts'][str(k)] = {'cost': cost, 'ess_is': ess}
        print(f'[train {t}] ckpt {k}: training cost {cost:.2e}, ESS_IS {ess:.3f}', flush=True)
    save_json(out / 'train.json', info)

def load(t, step=None):
    import json
    info = json.loads((flow_dir(t) / 'train.json').read_text())
    if step is None:
        step = max((int(k) for k in info['ckpts']))
    f = torch.load(flow_dir(t) / f'ckpt_{step}.pt', weights_only=False, map_location=torch.get_default_device())
    return (f.requires_grad_(False).eval(), info['ckpts'][str(step)]['cost'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--ckpt0', action='store_true', help='only add the step-0 (initial) checkpoint')
    ap.add_argument('--pilot', type=float, default=None, help="multi-start pilot budget (overrides the target's)")
    ap.add_argument('--ms-chains', type=int, default=None, help="multi-start pilot chains (overrides the target's)")
    ap.add_argument('--variant', default=None, choices=['B'], help='B: reverse KL only (multimodal: + fixed pilot anchor); run with GTI_FLOWS=flows_B to save elsewhere')
    args = ap.parse_args()
    setup(C.DTYPE)
    if args.pilot is not None:
        C.TARGETS[args.target]['pilot'] = args.pilot
    if args.ms_chains is not None:
        C.TARGETS[args.target]['multistart'] = {**C.TARGETS[args.target].get('multistart', {}), 'chains': args.ms_chains}
    train(args.target, ckpt0_only=args.ckpt0, variant=args.variant)
if __name__ == '__main__':
    main()
