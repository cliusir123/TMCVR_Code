import argparse
import json
import numpy as np
from common import flow_dir, raw_dir, setup
from workflow import config as C
from figures import GREEN, GREY, draw_benchmark, draw_flow, draw_gamma, draw_pruning, out_dir, plt_

def draw_family(ax, t, ms=2.5):
    D = json.loads((raw_dir(t) / 'samplers_family_main.json').read_text())
    M = sorted(D['members'], key=lambda r: r['fisher'])
    F = np.array([r['fisher'] for r in M])
    for s, col, mk, name in (('rwm', GREEN, 'o', 'TGI-RWM'), ('mala', '#d95f02', 's', 'TGI-MALA')):
        ax.plot(F, [r[s]['best_ess_per_cost'] for r in M], color=col, marker=mk, ms=ms, lw=1.0, label=name)
    E = json.loads((raw_dir(t) / 'family_ess_is_repeats.json').read_text())['members']
    key = lambda c: E[min(E, key=lambda k: abs(float(k) - c))]
    med = [key(r['param'])['median'] for r in M]
    ax.fill_between(F, [key(r['param'])['q10'] for r in M], [key(r['param'])['q90'] for r in M], color=GREY, alpha=0.18, lw=0)
    ax.plot(F, med, color=GREY, ls='-', lw=0.8, marker='x', ms=ms, label='ESS$_{\\rm IS}/N$')
    ax.set(xscale='log', yscale='log', ylabel='ESS / cost')
    return ax.get_legend_handles_labels()

def fig_a1(t='banana'):
    plt = plt_()
    plt.rcParams.update({'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5})
    j = C.TARGETS[t]['main_fn']
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 2.45))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.42, wspace=0.42)
    panels = [(draw_family, (t,), {}, 'Fisher divergence $\\mathcal{F}$', 'shear family $T_c$'), (draw_gamma, (t, 'accurate', 'rwm'), dict(fn=j, ms_best=4), 'step size $\\gamma$', 'TGI-RWM, $c=0.01$'), (draw_gamma, (t, 'accurate', 'mala'), dict(fn=j, ms_best=4), 'step size $\\gamma$', 'TGI-MALA, $c=0.01$'), (draw_pruning, (t, 'accurate', 'mala'), dict(fn=j), 'maximal degree $K$', 'pruning, $c=0.01$')]
    for ax, (f, args, kw, xl, title) in zip(axs, panels):
        h, l = f(ax, *args, **kw)
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=6.5)
        if f is draw_family:
            ax.legend(h, l, frameon=False, fontsize=5, loc='upper center', bbox_to_anchor=(0.5, -0.3))
        elif f is draw_pruning:
            ax.legend(h, l, frameon=False, ncol=2, fontsize=5, loc='upper center', bbox_to_anchor=(0.32, -0.3))
        elif args[2] == 'rwm':
            ax.legend(h, l, frameon=False, ncol=3, fontsize=5, loc='upper center', bbox_to_anchor=(1.21, -0.3))
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir('paper') / f'fig_A1_banana_family.{ext}', dpi=200)
    print(out_dir('paper') / 'fig_A1_banana_family.pdf')
A2_TARGETS = ['banana', 'funnel', 'gmix_d10_h6']
A2_METHODS = ['SNIS+CV', 'TGI-RWM+CV', 'TESS+CV', 'NeuTra+ZVz', 'NUTS+ZV2']
A2_NAME = {'banana': 'Banana', 'funnel': 'Funnel', 'gmix_d10_h6': 'Mixture', 'tmix_d10_h6': '$t$ mixture'}
A2_STEPS = (100, 1000, 10000)

def fig_a2():
    t = 'banana'
    plt = plt_()
    plt.rcParams.update({'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 6})
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 2.1), constrained_layout=True)
    j = C.TARGETS[t]['main_fn']
    h, l = draw_benchmark(axs[0], t, A2_METHODS, fn=j)
    H = dict(zip(l, h))
    axs[0].set_xlabel('total target cost')
    scratch = plt.figure()
    two = np.array([[scratch.add_subplot(1, 3, c + 1) for c in range(3)], list(axs[1:])])
    draw_flow(t, two, list(A2_STEPS))
    plt.close(scratch)
    info = json.loads((flow_dir(t) / 'train.json').read_text())['ckpts']
    for c, k in enumerate(A2_STEPS):
        a = axs[c + 1]
        a.set_title(f"{k} steps, ESS$_{{\\rm IS}}/N={info[str(k)]['ess_is']:.2f}$", fontsize=6)
        if c:
            a.set_ylabel('')
    order = [m for m in ('SNIS + CV', 'TGI-RWM + CV', 'TESS + CV', 'NeuTra + ZV-CV', 'NUTS + ZV-CV') if m in H]
    fig.legend([H[m] for m in order], order, loc='outside lower center', ncol=5, frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir('paper') / f'fig_A2_flow_benchmark.{ext}', dpi=200)
    print(out_dir('paper') / 'fig_A2_flow_benchmark.pdf')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a1', action='store_true')
    ap.add_argument('--a2', action='store_true')
    a = ap.parse_args()
    setup(C.DTYPE)
    if a.a1:
        fig_a1()
    if a.a2:
        fig_a2()
if __name__ == '__main__':
    main()
