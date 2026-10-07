import argparse
import json
import math
import numpy as np
import torch
import common
from common import make, setup, truth, save_json
from workflow import config as C
from hermite import CVRecorder, HermiteDictionary, fit_dictionary, magnitude
from mcmc import GroupJAdapter, LatentTarget, make_generator, make_proposal, run_chains
from mcmc.adaptation import J_ROBUST
from transports.diagnostics import fisher, log_importance_weights
CKPTS = [0, 100, 250, 500, 1000, 2000, 5000, 10000]
SHORT = [0, 100, 250, 500, 1000]
SHORT3 = [0, 250, 500, 1000, 2000, 3000]
VARIANTS = {'banana': {'B': ('B', dict(scheme='B', tail=False), False), 'B_tail': ('B + tail', dict(scheme='B', tail=True), False), 'A': ('A', dict(scheme='A', tail=False), False), 'A_tail': ('A + tail (default)', {}, True), 'A_tail_small': ('A + tail, SMALL', dict(arch=C.SMALL), False), 'A_tail_10': ('A + tail, 1/10 cost', dict(train=dict(steps=1000), ckpts=SHORT), False)}, 'gmix_d10_h6': {'B': ('B', dict(scheme='B'), False), 'A': ('A', dict(scheme='A'), False), 'Aa_noanchor': ('A$^a$ without anchor', dict(anchor=0.0), False), 'Aa': ('A$^a$ (default)', {}, True), 'Aa_notail': ('A$^a$ without tail', dict(tail=False), False), 'Aa_10': ('A$^a$, 1/10 cost', dict(train=dict(steps=1000), pilot=80000.0, ckpts=SHORT), False)}}
ROOT = common.RESULTS / 'train_ablation'

def variant_name(t, v):
    return f'{t}@{v}'

def register(t):
    for v, (_, over, prod) in VARIANTS[t].items():
        cfg = dict(C.TARGETS[t], base=C.TARGETS[t].get('base', t), **over)
        if not prod and 'ckpts' not in over:
            cfg['ckpts'] = CKPTS
        C.TARGETS[variant_name(t, v)] = cfg

def flow_root(t, v):
    return common.RESULTS if VARIANTS[t][v][2] else ROOT

def load_variant(t, v, step=None):
    import train
    old = common.RESULTS
    common.RESULTS = flow_root(t, v)
    try:
        name = t if VARIANTS[t][v][2] else variant_name(t, v)
        info = json.loads((common.flow_dir(name) / 'train.json').read_text())
        return (train.load(name, step), info)
    finally:
        common.RESULTS = old

def part_train(t):
    import train
    old = common.RESULTS
    common.RESULTS = ROOT
    try:
        for v, (_, _, prod) in VARIANTS[t].items():
            if prod:
                continue
            name = variant_name(t, v)
            if (common.flow_dir(name) / 'train.json').exists():
                print(f'[ablation {t}] {v}: trained', flush=True)
                continue
            print(f'[ablation {t}] training {v}', flush=True)
            train.train(name)
    finally:
        common.RESULTS = old

def band(n, d, gen):
    lo, hi = C.TAIL['band']
    u = torch.randn(n, d, generator=gen)
    u = u / u.norm(dim=-1, keepdim=True)
    return (d ** 0.5 + lo + (hi - lo) * torch.rand(n, 1, generator=gen)) * u

@torch.no_grad()
def ckpt_metrics(target, T, gen, n=20000):
    d = target.dim
    lw0 = log_importance_weights(target, T, torch.randn(n, d, generator=gen))
    lwb = log_importance_weights(target, T, band(n, d, gen))
    ref = torch.nanmedian(lw0[torch.isfinite(lw0)])
    rel = (lwb[torch.isfinite(lwb)] - ref).cpu().numpy()
    x = target.sample(n, generator=gen)
    z = T.inverse(x)[0]
    ok = torch.isfinite(z).all(-1)
    out = {'tail_q999': float(np.quantile(rel, 0.999)), 'tail_max': float(rel.max()), 'fisher': float(fisher(LatentTarget(target, T), z[ok]))}
    xq = T(torch.randn(n, d, generator=gen))[0]
    out['mode_weight'] = float((xq[:, 0] > 0).double().mean())
    return out

def chain_eval(target, T, sampler, j, seed):
    F_fns = [f for _, f, _ in target.test_functions]
    D = HermiteDictionary(target.dim, C.DICT['K'], max_support=C.DICT['p'])
    fit = fit_dictionary(D, T, F_fns, {'main': magnitude(C.DICT['K'], C.DICT['p'], C.DICT['lam'])}, C.DICT['n_select'], C.DICT['n_mc'], C.DICT['n_mc'])
    R = 2000
    holder = {}

    def factory(g):
        holder['r'] = CVRecorder(fit, F_fns, ['main'], g)
        return holder['r']
    gen = make_generator(500 + seed)
    out = run_chains(LatentTarget(target, T), make_proposal(sampler), torch.randn(R, target.dim, generator=gen), torch.full((R,), 0.5), n=C.N_STEPS, burn=C.BURN, per_chain=True, generator=gen, adapter=GroupJAdapter(torch.zeros(R, dtype=torch.long), fit.V0('main')[j:j + 1], 0.05, robust=J_ROBUST), recorder_factory=factory)
    res = holder['r'].results()
    return (res['fbar'].cpu().numpy(), res['mu_fixed'].cpu().numpy(), float(out['gamma'][0]), float(out['acc'].mean()))

def part_eval(t):
    target = make(t)
    tr = truth(target)
    j = C.TARGETS[t]['headline']
    samplers = ['rwm'] + (['mala'] if t == 'banana' else [])
    p = ROOT / f'{t}_eval.json'
    res = json.loads(p.read_text()) if p.exists() else {}
    for v, (label, _, _) in VARIANTS[t].items():
        if v in res:
            res[v]['label'] = label
            continue
        (T, cost), info = load_variant(t, v)
        gen = torch.Generator(device=torch.get_default_device()).manual_seed(7)
        curve = []
        for k in sorted((int(s) for s in info['ckpts'])):
            (Tk, ck), _ = load_variant(t, v, k)
            m = ckpt_metrics(target, Tk, gen)
            curve.append(dict(step=k, cost=ck, ess_is=info['ckpts'][str(k)]['ess_is'], **m))
        chains = {}
        for s in samplers:
            per = []
            for seed in (1, 2):
                f, mfix, g, acc = chain_eval(target, T, s, j, seed)
                sd = [math.sqrt(x) for x in tr['var']]
                zc = (mfix[:, j] - np.median(mfix[:, j])) / (1.4826 * np.median(np.abs(mfix[:, j] - np.median(mfix[:, j]))))
                per.append(dict(gamma=g, acceptance=acc, ess_n=[float(tr['var'][i] / (C.N_STEPS * f[:, i].var(ddof=1))) for i in range(f.shape[1])], vrf=[float(f[:, i].var(ddof=1) / mfix[:, i].var(ddof=1)) for i in range(f.shape[1])], bias=[float((mfix[:, i].mean() - tr['mean'][i]) / sd[i]) for i in range(f.shape[1])], bias_se=[float(mfix[:, i].std(ddof=1) / math.sqrt(len(mfix)) / sd[i]) for i in range(f.shape[1])], max_z=float(np.abs(zc).max()), z_sorted=np.sort(np.abs(zc))[::-1][:200].tolist()))
            chains[s] = per
            print(f"[ablation {t}] {v} {s}: gamma {per[0]['gamma']:.3f}, VRF {per[0]['vrf'][j]:.1f} / {per[1]['vrf'][j]:.1f}, max z {per[0]['max_z']:.1f} / {per[1]['max_z']:.1f}, bias {per[0]['bias'][j]:+.2e}", flush=True)
        res[v] = dict(label=label, train_cost=cost, curve=curve, chains=chains)
        save_json(p, res)
    save_json(p, {v: res[v] for v in VARIANTS[t] if v in res})

def part_figures(t):
    from figures import FAMILY, plt_
    res = json.loads((ROOT / f'{t}_eval.json').read_text())
    res = {v: res[v] for v in VARIANTS[t] if v in res}
    out = common.OUTPUT / 'train' / t
    out.mkdir(parents=True, exist_ok=True)
    j = C.TARGETS[t]['headline']
    plt = plt_()
    cols = ['#7570b3', '#a6761d', '#e7298a', '#1b9e77', '#66a61e']
    keys = [('ess_is', 'ESS$_{\\rm IS}/N$', False), ('tail_q999', 'tail $\\log w$ (99.9%)', False), ('fisher', 'Fisher divergence $\\mathcal{F}$', True)] if t == 'banana' else [('mode_weight', 'mode weight $q(x_1>0)$', False), ('ess_is', 'ESS$_{\\rm IS}/N$', False), ('tail_q999', 'tail $\\log w$ (99.9%)', False)]
    if TABLE_ONLY:
        return part_table(t, res, out, j)
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.3))
    for (k, lab, logy), ax in zip(keys, axs):
        for c, (v, r) in zip(cols, res.items()):
            cv = [p for p in r['curve'] if p['step'] > 0]
            ax.plot([p['cost'] for p in cv], [p[k] for p in cv], color=c, marker='o', ms=2, lw=1.2 if 'default' in r['label'] else 0.8, label=r['label'])
        if k == 'mode_weight':
            ax.axhline(0.5, color='black', ls=':', lw=0.6)
        ax.set(xscale='log', yscale='log' if logy else 'linear', xlabel='training cost', ylabel=lab)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'curves.{ext}', dpi=150)
    plt.close(fig)
    if t == 'banana':
        fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.7), sharey=True)
        for ax, s in zip(axs, ('rwm', 'mala')):
            for c, (v, r) in zip(cols, res.items()):
                zs = r['chains'][s][0]['z_sorted']
                ax.plot(np.arange(1, len(zs) + 1), zs, color=c, lw=1.2 if 'default' in r['label'] else 0.8, label=r['label'])
            ax.set(xscale='log', yscale='log', xlabel='rank among 2000 chains', title=f'TGI-{s.upper()}')
        axs[0].set_ylabel('|robust z| of the CV estimate')
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=5.5, columnspacing=1.0, handlelength=1.5)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
        for ext in ('pdf', 'png'):
            fig.savefig(out / f'chains.{ext}', dpi=150)
        plt.close(fig)
    part_table(t, res, out, j)
TABLE_ONLY = True

def part_table(t, res, out, j):
    from figures import num
    rows = []
    for v, r in res.items():
        last = r['curve'][-1]
        cells = [r['label'], num(r['train_cost']), f"{last['ess_is']:.3f}", num(last['fisher']), num(last['tail_q999'])]
        if t != 'banana':
            cells.append(f"{last['mode_weight']:.2f}")
        for s in [x for x in r['chains'] if x == 'rwm']:
            a, b = r['chains'][s]
            c = 1 if s == 'rwm' else 2
            eff = [x['ess_n'][j] * x['vrf'][j] / c for x in (a, b)]
            stuck = any((abs(x['bias'][j]) > 4 * x['bias_se'][j] for x in (a, b)))
            cells += [f"{num(a['vrf'][j])} / {num(b['vrf'][j])}", f"{max(a['max_z'], b['max_z']):.0f}", num(a['bias'][j], True) + ('$^*$' if stuck else '')]
        rows.append(' & '.join(cells) + ' \\\\')
    head = ['Variant', 'train cost', 'ESS$_{\\rm IS}/N$', '$\\mathcal F$', 'tail $\\log w$']
    if t != 'banana':
        head.append('$q(x_1>0)$')
    for s in [x for x in res[next(iter(res))]['chains'] if x == 'rwm']:
        S = s.upper()
        head += ['VRF', 'max $|z|$', 'bias']
    lines = ['\\begin{table*}[t]', '\\centering', '\\caption{Training ablation, ' + ('banana' if t == 'banana' else 'Gaussian mixture') + ': every variant trained once for $10^4$ steps, except the 1/10-cost one, a complete training of $10^3$ steps with every schedule scaled to its length' + ('' if t == 'banana' else ' (pilot $8\\times10^4$)') + "; flow diagnostics at the last checkpoint (tail $\\log w$: 99.9\\% quantile over the radial band relative to the median over $\\mathcal N(0,I)$) and TGI-RWM chains at $J$'s $\\gamma$ (two seeds of 2000 chains, 500 + 5000 steps; headline function). max $|z|$: the largest $|\\hat\\mu_r-\\mathrm{median}\\,\\hat\\mu|/(1.4826\\,\\mathrm{MAD})$ over the chains' control-variate estimates $\\hat\\mu_r$ (about 4 for Gaussian spread; larger values mean that a few chains stuck). bias: $(\\bar\\mu-\\mu)/\\sigma_\\pi$ of the pooled estimate; $^*$: more than four standard errors $\\mathrm{sd}(\\hat\\mu_r)/\\sqrt R$.}", f'\\label{{tab:train-{t}}}', '\\setlength{\\tabcolsep}{3pt}', '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'c' * (len(head) - 1) + '@{}}', '\\toprule', ' & '.join(head) + ' \\\\', '\\midrule', *rows, '\\bottomrule', '\\end{tabular*}', '\\end{table*}']
    (out / 'table.tex').write_text('\n'.join(lines) + '\n')
    print(f'  output/train/{t}/table.tex')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(VARIANTS))
    ap.add_argument('--part', nargs='+', default=['train', 'eval', 'figures'], choices=['train', 'eval', 'figures'])
    args = ap.parse_args()
    setup(C.DTYPE)
    register(args.target)
    for p in args.part:
        {'train': part_train, 'eval': part_eval, 'figures': part_figures}[p](args.target)
if __name__ == '__main__':
    main()
