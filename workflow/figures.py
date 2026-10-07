import argparse
import re
import json
import math
import numpy as np
from pathlib import Path
from common import out_dir, raw_dir, setup
from workflow import config as C
from experiments.metrics import SynthData, grid_metrics
PANEL_W, PANEL_AX_H = (1.69, 1.0)
PANEL_L, PANEL_R, PANEL_T, PANEL_B = (0.42, 0.06, 0.13, 0.77)
GREEN, PURPLE, GREY = ('#1b9e77', '#7570b3', '#666666')
PRUNE = [('full{K}', 'black', 'full'), ('p1_K{K}', '#2a78d6', 'struct. $p=1$'), ('p2_K{K}', '#e87ba4', 'struct. $p=2$'), ('l2_K{K}', '#a6761d', '$\\tau=2$'), ('llog_K{K}', GREEN, '$\\tau=2\\log|\\mathcal{K}|$')]

def plt_():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6, 'legend.fontsize': 5.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.size': 2, 'ytick.major.size': 2, 'axes.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
    return plt

def panel(letter=None):
    plt = plt_()
    H = PANEL_T + PANEL_AX_H + PANEL_B
    fig = plt.figure(figsize=(PANEL_W, H))
    ax = fig.add_axes([PANEL_L / PANEL_W, PANEL_B / H, (PANEL_W - PANEL_L - PANEL_R) / PANEL_W, PANEL_AX_H / H])
    if letter:
        ax.text(-0.02, 1.02, f'({letter})', transform=ax.transAxes, fontsize=7, fontweight='bold', ha='right', va='bottom')
    return (plt, fig, ax)

def save(fig, ax, t, stem, xlabel, handles, labels):
    ax.set_xlabel(xlabel)
    H = PANEL_T + PANEL_AX_H + PANEL_B
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, (PANEL_B - 0.3) / H), ncol=2, frameon=False, handlelength=1.6, handletextpad=0.3, columnspacing=0.6, borderaxespad=0.0, borderpad=0.0, labelspacing=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'{stem}.{ext}', dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'  output/{t}/{stem}.pdf')

def load(t, exp, transport, sampler):
    p = raw_dir(t) / f'{exp}_{transport}_{sampler}.jsonl'
    D = SynthData(p) if p.exists() else None
    if not D:
        return None
    label = D.labels()[0]
    return (D.meta[label], D.runs[label, sampler])

def pos(v):
    v = np.asarray(v, dtype=float)
    return np.where(v > 0, v, np.nan)

def fig_pruning(t, transport='flow', sampler='rwm', letter=None):
    if load(t, 'pruning', transport, sampler) is None:
        return
    plt, fig, ax = panel(letter)
    h, l = draw_pruning(ax, t, transport, sampler)
    save(fig, ax, t, f'pruning_{transport}_{sampler}', 'maximal degree $K$', h, l)

def draw_pruning(ax, t, transport, sampler, ms=2, fn=None):
    meta, runs = load(t, 'pruning', transport, sampler)
    j = C.TARGETS[t]['headline'] if fn is None else fn
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax2 = ax.twinx()
    ax2.axis('off')
    Ks = meta['Ks']
    top = 1.0
    for pat, col, lab in PRUNE:
        if pat.format(K=Ks[0]) not in meta['arms']:
            continue
        vrf, kept = ([], [])
        for K in Ks:
            gm = grid_metrics(meta, runs, j, pat.format(K=K))
            v = np.where(np.isfinite(gm['eff']), gm['eff'], -np.inf)
            vrf.append(gm['vrf'][int(np.argmax(v))])
            kept.append(meta['arms'][pat.format(K=K)]['kept'][j])
        top = max(top, np.nanmax(vrf))
        ax.plot(Ks, pos(vrf), color=col, marker='o', ms=ms, lw=0.9, label=lab)
        ax2.plot(Ks, kept, color=col, marker='x', ms=ms, lw=0.5, ls=':')
    ax2.set(yscale='log', ylim=(0.7, None))
    ax.set(yscale='log', xticks=Ks, ylabel='VRF')
    lo, hi = ax.get_ylim()
    if hi / lo < 3:
        ax.set_ylim(1, 3 * hi)
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=GREY, marker='o', ms=ms, lw=0.9), Line2D([], [], color=GREY, marker='x', ms=ms, lw=0.5, ls=':')]
    l += ['VRF', '# terms']
    return (h, l)

def gamma_curves(meta, runs, j, arm='main'):
    gm = grid_metrics(meta, runs, j, arm)
    return (gm['gamma'], gm['ess_n'], gm['vrf'])

def fig_gamma(t, transport='flow', sampler='mala', letter=None):
    if load(t, 'gamma', transport, sampler) is None:
        return
    plt, fig, ax = panel(letter)
    h, l = draw_gamma(ax, t, transport, sampler)
    save(fig, ax, t, f'gamma_{transport}_{sampler}', 'step size $\\gamma$', h, l)

def draw_gamma(ax, t, transport, sampler, ms=1.0, ms_best=4, fn=None):
    meta, runs = load(t, 'gamma', transport, sampler)
    j = C.TARGETS[t]['headline'] if fn is None else fn
    g, ess, vrf = gamma_curves(meta, runs, j)
    eff = ess * vrf
    ib = int(np.nanargmax(eff))
    jp = raw_dir(t) / 'j_gamma.json'
    jg = json.loads(jp.read_text()).get(f'{transport}:{sampler}') if jp.exists() else None
    for y, col, mk, lw, name in ((ess, '#2a78d6', 'o', 0.8, 'ESS'), (vrf, GREEN, 's', 0.8, 'VRF'), (eff, 'black', 'o', 1.3, 'ESS$\\times$VRF')):
        ax.plot(g, y / np.nanmax(y), color=col, marker=mk, ms=ms, lw=lw, label=name)
    if jg:
        ax.axvspan(min(jg), max(jg), color='#d95f02', alpha=0.15, lw=0)
        ax.axvline(float(np.median(jg)), color='#d95f02', ls='--', lw=0.9, label='$\\gamma$ by $J$')
    gs = np.array([r['gamma'] for r in runs])
    speed = np.log(2 * gs - gs ** 2) + np.array([r['min0'] for r in runs])
    if sampler == 'pmala':
        speed = speed + np.array([r['log_a'] for r in runs])
    ax.axvline(float(gs[int(np.nanargmax(speed))]), color='#555555', ls=':', lw=1.0, label='$\\gamma$ by speed')
    ax.plot([g[ib]], [1.0], marker='^', color='black', ms=ms_best, ls='none', label='best $\\gamma$')
    ax.set(xlim=(0, 1.02), ylim=(0, 1.1), ylabel='relative to max.', xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=['0', '.25', '.5', '.75', '1'])
    return ax.get_legend_handles_labels()

def fig_samplers(t):
    p = raw_dir(t) / 'samplers_family.json'
    if not p.exists():
        return
    plt, fig, ax = panel()
    h, l = draw_samplers(ax, t)
    save(fig, ax, t, f'samplers_{t}', 'Fisher divergence $\\mathcal{F}$', h, l)

def draw_samplers(ax, t, ms=2.5):
    D = json.loads((raw_dir(t) / 'samplers_family.json').read_text())
    M = sorted(D['members'], key=lambda r: r['fisher'])
    F = np.array([r['fisher'] for r in M])
    for s, col, mk, name in (('rwm', GREEN, 'o', 'TGI-RWM'), ('mala', '#d95f02', 's', 'TGI-MALA'), ('pmala', PURPLE, '^', 'TGI-pMALA')):
        ax.plot(F, [r[s]['best_ess_per_cost'] for r in M], color=col, marker=mk, ms=ms, lw=1.0, label=name)
    ax.plot(F, [r['ess_is'] for r in M], color=GREY, ls='-', lw=0.8, marker='x', ms=ms, label='ESS$_{\\rm IS}/N$')
    dsp = np.array([r['dispersion'] for r in M])
    keep = dsp > 1e-06
    if keep.any():
        ax.plot(F[keep], dsp[keep], color='black', ls='--', lw=0.8, label='dispersion of $\\delta$')
    ax.set(xscale='log', yscale='log', ylabel='ESS / cost')
    return ax.get_legend_handles_labels()

def fig_analytic_overall(t):
    if not (raw_dir(t) / 'samplers_family.json').exists():
        return
    plt = plt_()
    fs = dict(lab=8, tick=7, leg=6.5)
    plt.rcParams.update({'axes.labelsize': fs['lab'], 'xtick.labelsize': fs['tick'], 'ytick.labelsize': fs['tick'], 'legend.fontsize': fs['leg']})
    W, AX_W, AX_H, L, GAP, TOP, BOT = (7.0, 1.15, 1.25, 0.55, 0.5, 0.08, 1.12)
    H = TOP + AX_H + BOT
    fig = plt.figure(figsize=(W, H))
    panels = [(draw_samplers, (t,), 'Fisher divergence $\\mathcal{F}$'), (draw_gamma, (t, 'accurate', 'rwm'), 'step size $\\gamma$ (TGI-RWM)'), (draw_gamma, (t, 'accurate', 'mala'), 'step size $\\gamma$ (TGI-MALA)'), (draw_pruning, (t, 'accurate', 'mala'), 'maximal degree $K$')]
    for i, (fn, args, xl) in enumerate(panels):
        ax = fig.add_axes([(L + i * (AX_W + GAP)) / W, BOT / H, AX_W / W, AX_H / H])
        h, l = fn(ax, *args, **{'ms_best': 5} if fn is draw_gamma else {})
        ax.set_xlabel(xl)
        last = i == len(panels) - 1
        ax.legend(h, l, loc='upper right' if last else 'upper center', bbox_to_anchor=(1.0 if last else 0.5, -0.42 / AX_H - 0.02), ncol=2, frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=0.8, labelspacing=0.35, borderaxespad=0.0)
    stem = f'{fig_name(t)}_analytic_overall'
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'{stem}.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/{stem}.pdf')

def bench_rows(t):
    p = raw_dir(t) / 'benchmark.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

def rmse(R, mu, sd, j, ci=False, n_boot=2000):
    e = np.array([r['estimate'][j] for r in R], dtype=float)
    e = e[np.isfinite(e)]
    if not e.size:
        return (np.nan, np.nan, np.nan) if ci else np.nan
    v = float(np.sqrt(((e - mu) ** 2).mean()) / sd)
    if not ci:
        return v
    idx = np.random.default_rng(0).integers(0, e.size, (n_boot, e.size))
    b = np.sqrt(((e[idx] - mu) ** 2).mean(1)) / sd
    return (v, float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975)))

def curve(rows, method, mu, sd, j, envelope):
    g = {}
    for r in rows:
        if r['method'] == method:
            g.setdefault(r['key'], []).append(r)
    pts = sorted(((float(np.median([r['total_cost'] for r in R])), *rmse(R, mu, sd, j, ci=True)) for R in g.values()))
    pts = [p for p in pts if np.isfinite(p[1])]
    if not envelope:
        return pts
    env, best = ([], np.inf)
    for p in pts:
        if p[1] < best:
            best = p[1]
            env.append(p)
    return env
FAMILY = {'TGI-RWM': (GREEN, 'D', 'TGI-RWM', True), 'TGI-RWM-δ': ('#1b7837', 's', 'TGI-RWM (δ)', True), 'TGI-MALA': ('#66a61e', 'd', 'TGI-MALA', True), 'TESS': ('#e6ab02', '^', 'TESS', True), 'NeuTra': (PURPLE, 'v', 'NeuTra', True), 'SNIS': ('#2a78d6', 'o', 'SNIS', True), 'GI-MALA': ('#d95f02', 'p', 'GI-MALA', False), 'GI-pMALA': ('#e7298a', 'P', 'GI-pMALA', False), 'GI-pMALA-LGM': ('#b3007a', '*', 'GI-pMALA (LGM)', False), 'Ellipt': ('#a6761d', '<', 'Ellipt', False), 'MALA': ('#2a78d6', 'x', 'MALA', False), 'NUTS': ('black', 's', 'NUTS', False)}

def method_style(m):
    fam, _, cv = m.partition('+')
    return (fam, bool(cv))

def draw_benchmark(ax, t, methods=None, fn=None):
    from common import make, truth
    rows = bench_rows(t)
    if not rows:
        return None
    j = C.TARGETS[t]['headline'] if fn is None else fn
    tr = truth(make(t))
    mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
    from matplotlib.ticker import LogLocator
    have = list(dict.fromkeys((r['method'] for r in rows)))
    order = [f + s for f in FAMILY for s in ('+CV', '+ZV2', '+ZVz', '')]
    methods = [m for m in order if m in have and (methods is None or m in methods)]
    for m in methods:
        fam, cv = method_style(m)
        col, mk, name, env = FAMILY[fam]
        pts = curve(rows, m, mu, sd, j, env)
        if not pts:
            continue
        c, e, lo, hi = map(np.array, zip(*pts))
        ax.fill_between(c, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(c, e, color=col, ls='-' if cv else '--', marker=mk, ms=1.8, lw=1.0, mfc=col if cv else 'white', label=name + (' + CV' if m.endswith('+CV') else ' + ZV-CV' if cv else ''))
    ax.set(xscale='log', yscale='log', ylabel='RMSE / sd')
    ax.xaxis.set_major_locator(LogLocator(numticks=4))
    return ax.get_legend_handles_labels()

def fig_benchmark(t, letter=None, methods=None, stem='benchmark', big=False):
    if not bench_rows(t):
        return
    if big:
        plt = plt_()
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
    else:
        plt, fig, ax = panel(letter)
    h, l = draw_benchmark(ax, t, methods)
    if big:
        ax.set_xlabel('total target cost')
        ax.legend(h, l, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=5.5)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(out_dir(t) / f'{stem}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  output/{t}/{stem}.pdf')
    else:
        save(fig, ax, t, stem, 'total target cost', h, l)

def target_background(ax, q, lo, hi, bins=200):
    from scipy.ndimage import gaussian_filter
    H, ex, ey = np.histogram2d(q[:, 0], q[:, 1], bins=bins, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
    H = gaussian_filter(H, 1.5)
    H = H / H.sum()
    top = np.log(H.max())
    ax.imshow(np.log(H.T + 1e-12), origin='lower', aspect='auto', cmap='Greys', extent=(ex[0], ex[-1], ey[0], ey[-1]), vmin=top - 8, vmax=top + 1.5, zorder=0, interpolation='bilinear')
    hs = np.sort(H.ravel())[::-1]
    cm = np.cumsum(hs)
    lev = sorted({float(hs[min(np.searchsorted(cm, a), hs.size - 1)]) for a in (0.5, 0.9, 0.99)})
    xc, yc = (0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1]))
    ax.contour(xc, yc, H.T, levels=lev, colors='#6baed6', linewidths=0.6, zorder=1)

def draw_flow(t, axs, steps=None, pair=(0, 1)):
    import torch
    from common import draws, make
    from train import load as load_flow
    from common import flow_dir
    info = json.loads((flow_dir(t) / 'train.json').read_text())
    have = sorted((int(k) for k in info['ckpts']))
    steps = [k for k in steps or C.TARGETS[t].get('flow_steps', C.FLOW_STEPS) if k in have] or have[::2]
    target = make(t)
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(1)
    x = draws(target, 4000, gen)
    xbig = draws(target, 1000000, gen).cpu().numpy()[:, list(pair)]
    th = np.linspace(0, 2 * np.pi, 200)
    i, k2 = pair
    xn = x.cpu().numpy()
    for c, k in enumerate(steps):
        T, _ = load_flow(t, k)
        with torch.no_grad():
            z = T.inverse(x)[0]
        zn = z.cpu().numpy()
        ok = np.isfinite(zn).all(1)
        A = np.c_[zn[ok], np.ones(ok.sum())]
        coef = np.linalg.lstsq(A, xn[ok][:, [i, k2]], rcond=None)[0][:-1]
        E, Rq = np.linalg.qr(coef)
        E = E * np.sign(np.diag(Rq))
        P = zn @ E
        ax = axs[0, c]
        ax.scatter(P[:, 0], P[:, 1], s=0.3, color='#2a78d6', alpha=0.5, rasterized=True)
        for r in (1, 2, 3):
            ax.plot(r * np.cos(th), r * np.sin(th), color='black', lw=0.5, ls=':')
        ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect='equal', xlabel=f'$z$ along $x_{i + 1}$')
        ax.set_title(f"{k} steps, ESS$_{{\\rm IS}}/N={info['ckpts'][str(k)]['ess_is']:.2f}$", fontsize=6)
        ax = axs[1, c]
        lo, hi = np.quantile(xbig, [0.002, 0.998], axis=0)
        pad = 0.15 * (hi - lo)
        lo, hi = (lo - pad, hi + pad)
        target_background(ax, xbig, lo, hi)
        s_ = np.linspace(-3, 3, 200)
        Et = torch.as_tensor(E, dtype=z.dtype, device=z.device)
        for gv in np.linspace(-3, 3, 13):
            for line in (np.stack([s_, np.full_like(s_, gv)], 1), np.stack([np.full_like(s_, gv), s_], 1)):
                zz = torch.as_tensor(line, dtype=z.dtype, device=z.device) @ Et.T
                with torch.no_grad():
                    y = T(zz)[0].cpu().numpy()
                ax.plot(y[:, i], y[:, k2], color='#d95f02', lw=0.4, zorder=2)
        ax.set(xlabel=f'$x_{i + 1}$', xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]))
        ax.grid(False)
    axs[0, 0].set_ylabel(f'$z$ along $x_{k2 + 1}$')
    axs[1, 0].set_ylabel(f'$x_{k2 + 1}$')
    return steps

def fig_flow(t, steps=None, pair=(0, 1)):
    from common import flow_dir
    info = json.loads((flow_dir(t) / 'train.json').read_text())
    have = sorted((int(k) for k in info['ckpts']))
    n = len([k for k in steps or C.TARGETS[t].get('flow_steps', C.FLOW_STEPS) if k in have]) or len(have[::2])
    plt = plt_()
    fig, axs = plt.subplots(2, n, figsize=(1.7 * n, 3.6), squeeze=False)
    draw_flow(t, axs, steps, pair)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'flow.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/flow.pdf')

def draw_convergence(t, axs):
    p = raw_dir(t) / 'convergence.npz'
    if not p.exists():
        return None
    from common import make, truth
    D = np.load(p)
    j = C.TARGETS[t]['headline']
    tr = truth(make(t))
    mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
    panels = [('tgi', 'TGI-RWM (+CV)', f"$\\gamma={float(D['tgi_gamma']):.2f}$, cost 1 / iteration"), ('gi', 'GI-MALA (+CV)', f"$\\gamma={float(D['gi_gamma']):.2f}$, cost 2 / iteration"), ('nuts', 'NUTS (+ZV-CV)', 'cost $\\approx 2\\times$ leapfrog steps')]
    cols = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    lim = 0.0
    for ax, (key, title, sub_) in zip(axs, panels):
        plain, cv = (D[f'{key}_plain'], D[f'{key}_cv'])
        it = np.arange(1, plain.shape[1] + 1)
        for r in range(plain.shape[0]):
            ax.plot(it, (plain[r] - mu) / sd, color=cols[r % 5], lw=0.6, ls='--')
            ax.plot(it, (cv[r] - mu) / sd, color=cols[r % 5], lw=0.9)
        ax.axhline(0.0, color='black', lw=0.6, ls=':')
        k0 = min(100, plain.shape[1] // 10)
        tail = np.concatenate([plain[:, k0:], cv[:, k0:]], 1)
        lim = max(lim, np.nanmax(np.abs((tail - mu) / sd)))
        ax.set(xscale='log', xlabel='iteration', title=f'{title}\n{sub_}')
        ax.title.set_fontsize(6)
    for ax in axs:
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
    return lim

def conv_legend():
    from matplotlib.lines import Line2D
    return ([Line2D([], [], color='grey', ls='--', lw=0.6), Line2D([], [], color='grey', lw=0.9)], ['plain average', 'with control variates'])

def fig_convergence(t):
    if not (raw_dir(t) / 'convergence.npz').exists():
        return
    plt = plt_()
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.2), sharey=True)
    draw_convergence(t, axs)
    axs[0].set_ylabel('(ergodic average $-$ truth) / sd')
    fig.legend(*conv_legend(), loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'convergence.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/convergence.pdf')

def fig_overview(t):
    plt = plt_()
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.42, left=0.07, right=0.99, top=0.96, bottom=0.17)
    axf = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)])
    draw_flow(t, axf)
    axb = fig.add_subplot(gs[2, 0])
    hb = draw_benchmark(axb, t)
    axb.set_xlabel('total target cost')
    axb.set_title('all methods', fontsize=6)
    axc = [fig.add_subplot(gs[2, c]) for c in (1, 2, 3)]
    draw_convergence(t, axc)
    axc[0].set_ylabel('(average $-$ truth) / sd')
    for ax in axc[1:]:
        ax.tick_params(labelleft=False)
    h, l = hb if hb else ([], [])
    hc, lc = conv_legend()
    fig.legend(h + hc, l + lc, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=5.5, columnspacing=1.0)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'{fig_name(t)}_overview.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/{fig_name(t)}_overview.pdf')

def mpl_names(t, target):
    names = C.TARGETS[t].get('tex_names')
    if not names:
        return [n for n in target.names]
    out = []
    for n in names:
        n = re.sub('\\\\tfrac(\\d)(\\d)', '\\\\frac{\\1}{\\2}', n)
        n = re.sub('\\\\tfrac(\\d)\\{', '\\\\frac{\\1}{', n)
        n = n.replace('\\mathbb 1\\{', '1\\{')
        out.append(f'${n}$')
    return out

def fig_posterior(t):
    p = raw_dir(t) / 'posterior_samples.npz'
    if not p.exists():
        return
    from common import make, truth
    D = np.load(p)
    target = make(t)
    tr = truth(target)
    names = mpl_names(t, target)
    nF = len(names)
    plt = plt_()
    fig, axs = plt.subplots(1, nF, figsize=(7.0, 1.75), squeeze=False)
    for j, ax in enumerate(axs[0]):
        fc, fr = (D['F_chain'][:, j], D['F_ref'][:, j])
        lo, hi = np.quantile(np.r_[fc, fr], [0.002, 0.998])
        bins = np.linspace(lo, hi, 60)
        discrete = np.unique(fr).size <= 2
        if discrete:
            bins = np.array([-0.25, 0.25, 0.75, 1.25])
        ax.hist(fr, bins=bins, density=True, color='#cccccc', label='reference')
        ax.hist(fc, bins=bins, density=True, histtype='step', color=GREEN, lw=1.0, label='TGI-RWM')
        ax.axvline(tr['mean'][j], color='black', ls='--', lw=0.8, label='true mean' if t in SYNTHETIC else 'reference mean')
        ax.axvline(fc.mean(), color=GREEN, lw=0.8, label='chain mean')
        try:
            ax.set_title(names[j], fontsize=6.5)
            fig.canvas.draw()
        except ValueError:
            ax.set_title(target.names[j], fontsize=6.5)
        ax.set_yticks([])
        ax.tick_params(labelsize=5.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=5.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.6)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'posterior.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/posterior.pdf')

def _est_select(t):
    from common import make, truth
    rows = bench_rows(t)
    if not rows:
        return None
    target = make(t)
    tr = truth(target)
    cfg = C.TARGETS[t]
    omit = set(cfg.get('table_omit', []))
    if cfg.get('skip_pmala'):
        omit |= {'GI-pMALA-LGM', 'GI-pMALA-LGM+CV'}
    jh = cfg['headline']
    names = mpl_names(t, target)
    nF = len(names)
    have = list(dict.fromkeys((r['method'] for r in rows)))
    pairs = []
    for f in [f for f in FAMILY if not f.startswith('GI-')]:
        cv = f + '+CV' if f + '+CV' in have else f + '+ZV2' if f + '+ZV2' in have else None
        if f in have and cv and (f not in omit) and (cv not in omit):
            pairs.append((f, f, cv))
    keys = {}
    real = t not in SYNTHETIC
    Nmax, final = (max(C.PREFIX_N), cfg['ckpts'][-1])
    for f, plain, cv in pairs:
        g = {}
        for r in rows:
            if r['method'] == cv and (not real or (abs(r['budget'] - Nmax) <= 1e-06 * Nmax and r.get('ckpt') in (None, final))):
                g.setdefault(r['key'], []).append(r)
        if real:
            if g:
                keys[f] = next(iter(g))
            continue
        best = None
        for key, R in g.items():
            c = float(np.median([r['total_cost'] for r in R]))
            if c > 1.1 * C.COST_REF:
                continue
            e = np.array([r['estimate'][jh] for r in R], float) - tr['mean'][jh]
            rm = float(np.sqrt(np.nanmean(e ** 2)))
            if best is None or rm < best[0]:
                best = (rm, key)
        if best:
            keys[f] = best[1]
    pairs = [p for p in pairs if p[0] in keys]
    if not pairs:
        return None
    est = lambda m, key, j: np.array([r['estimate'][j] for r in rows if r['method'] == m and r['key'] == key], float)
    return dict(target=target, tr=tr, names=names, nF=nF, pairs=pairs, keys=keys, est=est)

def _est_panel(ax, j, S, fs=5.5):
    data, pos, cols, fills, ticks, tlab = ([], [], [], [], [], [])
    for g, (f, plain, cv) in enumerate(S['pairs']):
        x0 = 3 * g
        for k, m in enumerate((plain, cv)):
            v = S['est'](m, S['keys'][f], j)
            data.append(v[np.isfinite(v)])
            pos.append(x0 + k)
            cols.append(FAMILY[f][0])
            fills.append(k == 1)
        ticks.append(x0 + 0.5)
        tlab.append(FAMILY[f][2].replace(' (LGM)', ''))
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, flierprops=dict(markersize=1.5, markeredgewidth=0.3), medianprops=dict(color='black', lw=0.7), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for patch, col, fl in zip(bp['boxes'], cols, fills):
        patch.set(facecolor=col if fl else 'white', edgecolor=col, lw=0.8)
    ax.axhline(S['tr']['mean'][j], color='black', lw=0.6, ls='--', zorder=0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlab, fontsize=fs)
    ax.set_xlim(min(pos) - 0.8, max(pos) + 0.8)
    ax.tick_params(axis='y', labelsize=fs)
    ax.ticklabel_format(axis='y', useOffset=False)

def _post_panel(ax, j, D, tr, t):
    fc, fr = (D['F_chain'][:, j], D['F_ref'][:, j])
    lo, hi = np.quantile(np.r_[fc, fr], [0.002, 0.998])
    bins = np.linspace(lo, hi, 60)
    if np.unique(fr).size <= 2:
        bins = np.array([-0.25, 0.25, 0.75, 1.25])
    ax.hist(fr, bins=bins, density=True, color='#cccccc', label='reference')
    ax.hist(fc, bins=bins, density=True, histtype='step', color=GREEN, lw=1.0, label='TGI-RWM')
    ax.axvline(tr['mean'][j], color='black', ls='--', lw=0.8, label='true mean' if t in SYNTHETIC else 'reference mean')
    ax.axvline(fc.mean(), color=GREEN, lw=0.8, label='chain mean')
    ax.set_yticks([])
FIG_NAME = {'lotka_volterra': 'lotka_volterra', 'gmix_d10_h6': 'mixture', 'sparse_logistic_german': 'sparse_logistic', 'stochvol_T100': 'stochvol', 'phi4_N32': 'phi4'}

def fig_name(t):
    return FIG_NAME.get(t, t.removeprefix('pdb_'))

def fig_post_est(t):
    p = raw_dir(t) / 'posterior_samples.npz'
    S = _est_select(t)
    if not p.exists() or S is None:
        return
    D = np.load(p)
    names, tr = (S['names'], S['tr'])
    js = C.TARGETS[t].get('fig_fns', list(range(S['nF'])))
    nF = len(js)
    plt = plt_()
    fs = 7
    fig, axs = plt.subplots(2, nF, figsize=(7.0, 3.5), squeeze=False, gridspec_kw=dict(height_ratios=[1, 1.25]))
    for c, j in enumerate(js):
        _post_panel(axs[0, c], j, D, tr, t)
        axs[0, c].tick_params(labelsize=fs)
        from matplotlib.ticker import MaxNLocator
        axs[0, c].xaxis.set_major_locator(MaxNLocator(4))
        try:
            axs[0, c].set_title(names[j], fontsize=fs + 1.5)
            fig.canvas.draw()
        except ValueError:
            axs[0, c].set_title(S['target'].names[j], fontsize=fs + 1.5)
        _est_panel(axs[1, c], j, S, fs)
        axs[1, c].yaxis.set_major_locator(MaxNLocator(4))
    axs[0, 0].set_ylabel('density', fontsize=fs)
    axs[1, 0].set_ylabel('estimate', fontsize=fs)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [Patch(facecolor='white', edgecolor='grey'), Patch(facecolor='grey', edgecolor='grey')]
    zv = any((f == 'NUTS' for f, _, _ in S['pairs']))
    l += ['estimate without control variates', 'with control variates (ZV for NUTS)' if zv else 'with control variates']
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=fs, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4, h_pad=0.6)
    fig.align_ylabels(axs[:, 0])
    stem = f'{fig_name(t)}_overall'
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'{stem}.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/{stem}.pdf')

def fig_estimators(t):
    from common import make, truth
    rows = bench_rows(t)
    if not rows:
        return
    target = make(t)
    tr = truth(target)
    cfg = C.TARGETS[t]
    omit = set(cfg.get('table_omit', []))
    if cfg.get('skip_pmala'):
        omit |= {'GI-pMALA-LGM', 'GI-pMALA-LGM+CV'}
    jh = cfg['headline']
    names = mpl_names(t, target)
    nF = len(names)
    have = list(dict.fromkeys((r['method'] for r in rows)))
    pairs = []
    for f in [f for f in FAMILY if not f.startswith('GI-')]:
        cv = f + '+CV' if f + '+CV' in have else f + '+ZV2' if f + '+ZV2' in have else None
        if f in have and cv and (f not in omit) and (cv not in omit):
            pairs.append((f, f, cv))
    keys = {}
    real = t not in SYNTHETIC
    Nmax, final = (max(C.PREFIX_N), cfg['ckpts'][-1])
    for f, plain, cv in pairs:
        g = {}
        for r in rows:
            if r['method'] == cv and (not real or (abs(r['budget'] - Nmax) <= 1e-06 * Nmax and r.get('ckpt') in (None, final))):
                g.setdefault(r['key'], []).append(r)
        if real:
            if g:
                keys[f] = next(iter(g))
            continue
        best = None
        for key, R in g.items():
            c = float(np.median([r['total_cost'] for r in R]))
            if c > 1.1 * C.COST_REF:
                continue
            e = np.array([r['estimate'][jh] for r in R], float) - tr['mean'][jh]
            rm = float(np.sqrt(np.nanmean(e ** 2)))
            if best is None or rm < best[0]:
                best = (rm, key)
        if best:
            keys[f] = best[1]
    pairs = [p for p in pairs if p[0] in keys]
    if not pairs:
        return
    est = lambda m, key, j: np.array([r['estimate'][j] for r in rows if r['method'] == m and r['key'] == key], float)
    plt = plt_()
    fig, axs = plt.subplots(1, nF, figsize=(7.0, 2.3), squeeze=False)
    for j, ax in enumerate(axs[0]):
        data, pos, cols, fills, ticks, tlab = ([], [], [], [], [], [])
        for g, (f, plain, cv) in enumerate(pairs):
            x0 = 3 * g
            for k, m in enumerate((plain, cv)):
                v = est(m, keys[f], j)
                data.append(v[np.isfinite(v)])
                pos.append(x0 + k)
                cols.append(FAMILY[f][0])
                fills.append(k == 1)
            ticks.append(x0 + 0.5)
            tlab.append(FAMILY[f][2].replace(' (LGM)', ''))
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, flierprops=dict(markersize=1.2, markeredgewidth=0.3), medianprops=dict(color='black', lw=0.7), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for patch, col, fl in zip(bp['boxes'], cols, fills):
            patch.set(facecolor=col if fl else 'white', edgecolor=col, lw=0.8)
        ax.axhline(tr['mean'][j], color='black', lw=0.6, ls='--', zorder=0)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlab, rotation=30, ha='right', fontsize=5.5)
        try:
            ax.set_title(names[j], fontsize=6.5)
            fig.canvas.draw()
        except ValueError:
            ax.set_title(target.names[j], fontsize=6.5)
        ax.tick_params(axis='y', labelsize=5)
        ax.ticklabel_format(axis='y', useOffset=False)
    axs[0, 0].set_ylabel('estimate', fontsize=6)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend([Patch(facecolor='white', edgecolor='grey'), Patch(facecolor='grey', edgecolor='grey'), Line2D([], [], color='black', ls='--', lw=0.6)], ['without control variates', 'with control variates (ZV for NUTS)', 'true / reference mean'], loc='lower center', ncol=3, frameon=False, fontsize=5.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.5)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir(t) / f'estimators.{ext}', dpi=150)
    plt.close(fig)
    print(f'  output/{t}/estimators.pdf  (configurations: {keys})')
TEX_TITLE = {'banana': 'Banana ($d=8$)', 'funnel': 'Funnel ($d=10$)', 'gmix_d10_h6': 'Gaussian mixture ($d=10$, $h=6$)', 'pdb_garch': 'GARCH(1,1)', 'pdb_dugongs': 'Dugongs', 'pdb_one_comp': 'One-compartment model', 'pdb_gp_pois': 'GP Poisson regression', 'pdb_surgical': 'Surgical', 'pdb_pilots': 'Pilots', 'eight_schools': 'Eight schools', 'lotka_volterra': 'Lotka--Volterra', 'sensor': 'Sensor network', 'phi4_N32': '$\\phi^4$ field', 'sparse_logistic_german': 'Sparse logistic regression', 'stochvol_T100': 'Stochastic volatility'}
TEX_METHOD = {'+CV': ' + CV', '+ZV2': ' + ZV-CV', '+ZVz': ' + ZV-CV (latent)', '': ''}
SKIP_PMALA = ' GI-pMALA is not run on this target: its $d$ Hessian-vector products per step make it far more expensive than the other samplers.'

def num(v, signed=False):
    if v is None or not np.isfinite(v):
        return '--'
    if v != 0 and (not 0.01 <= abs(v) < 1000):
        e = int(math.floor(math.log10(abs(v))))
        mnt = v / 10 ** e
        if abs(round(mnt, 1)) >= 10:
            e, mnt = (e + 1, mnt / 10)
        return f"${mnt:{('+' if signed else '')}.1f}\\times10^{{{e}}}$"
    return f"{v:{('+' if signed else '')}.3g}"

def tex_names(t, target):
    names = C.TARGETS[t].get('tex_names')
    return [f'${x}$' for x in names] if names else ['\\texttt{' + n.replace('_', '\\_') + '}' for n in target.names]

def table_env(body, caption, label, wide=False):
    env = 'table*' if wide else 'table'
    return [f'\\begin{{{env}}}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}'] + body + [f'\\end{{{env}}}', '']
SYNTHETIC = ('banana', 'funnel', 'gmix_d10_h6')

def bold(s):
    return '{\\boldmath ' + s + '}' if '$' in s else '\\textbf{' + s + '}'

def bold_min(cells, values):
    fin = [v for v in values if v is not None and np.isfinite(v)]
    if not fin:
        return cells
    best = cells[values.index(min(fin))]
    return [bold(c) if c == best else c for c in cells]

def sci(v):
    if v is None or not np.isfinite(v):
        return '--'
    if v != 0 and abs(v) < 0.01:
        m, e = f'{v:.1e}'.split('e')
        return f'{m}e{int(e)}'
    return f'{v:.3g}'

def prefix_note(rows, meths, Ns):
    pre = [m for m in meths if any((r.get('prefix') for r in rows if r['method'] == m and r['budget'] < max(Ns)))]
    if not pre:
        return 'Every $N$ is a separate run.'
    if len(pre) == len(meths):
        return 'The estimates at $N<10^7$ are prefixes of the runs at $10^7$.'
    fams = list(dict.fromkeys((method_style(m)[0] for m in pre)))
    return ', '.join(fams) + ': the estimates at $N<10^7$ are prefixes of the runs at $10^7$.'

def rmse_by_cost(t, cfg, rows, meths, names, tr, title):
    final = cfg['ckpts'][-1]
    Ns = C.PREFIX_N
    ok = lambda r, m, N: r['method'] == m and abs(r['budget'] - N) <= 1e-06 * N and (r.get('ckpt') in (None, final))
    PER_BLOCK = 4
    nN = len(Ns)
    vals = {}
    for j in range(len(names)):
        mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
        for mth in meths:
            for N in Ns:
                sel = [r for r in rows if ok(r, mth, N)]
                vals[j, mth, N] = rmse(sel, mu, sd, j) if sel else float('nan')
    xnote = ''
    if cfg.get('nuts_extrapolate'):
        ref = json.loads((Path(__file__).resolve().parent.parent / 'results' / 'references' / f'reference_{t}.json').read_text())
        cref = ref['n_draws'] * ref['mean_leapfrog'] * 2.0
        mc = ref.get('mcse_plain', ref['mcse'])
        skip = set(cfg.get('nuts_extrapolate_skip', []))
        for j in range(len(names)):
            for N in Ns:
                vals[j, 'NUTS-X', N] = float('nan') if j in skip else mc[j] / math.sqrt(tr['var'][j]) * math.sqrt(cref / N)
        meths = list(meths) + ['NUTS-X']
        xnote = f" NUTS (extrap.): extrapolated from the reference run ({num(ref['n_draws'])} draws, {ref['mean_leapfrog']:.0f} leapfrog steps per draw)." + (' --: a NUTS chain stays in one mode.' if skip else '')
    cols = list(range(len(names)))
    label = {j: names[j] for j in cols}
    text = lambda j, m, N: sci(vals[j, m, N])
    if cfg.get('table_range'):
        rc = cfg['table_range']
        for m in meths:
            for N in Ns:
                v = [vals[j, m, N] for j in rc['range']]
                fin = [x for x in v if np.isfinite(x)]
                vals['R', m, N] = max(fin) if fin else float('nan')
                vals['Rlo', m, N] = min(fin) if fin else float('nan')
        cols = list(rc['keep']) + ['R']
        label['R'] = rc['name']
        text = lambda j, m, N: sci(vals[j, m, N]) if j != 'R' else '--' if not np.isfinite(vals['R', m, N]) else f"{sci(vals['Rlo', m, N])}--{sci(vals['R', m, N])}"
    k = min(PER_BLOCK, len(cols))
    body = ['\\begin{tabular}{@{}l' + 'c' * nN * k + '@{}}', '\\toprule']
    for b0 in range(0, len(cols), PER_BLOCK):
        js = cols[b0:b0 + PER_BLOCK]
        if b0:
            body.append('\\midrule')
        body.append(' & ' + ' & '.join((f'\\multicolumn{{{nN}}}{{c}}{{{label[j]}}}' for j in js)) + ' & ' * (nN * (k - len(js))) + ' \\\\')
        body.append(''.join((f'\\cmidrule(lr){{{2 + nN * i}-{1 + nN * (i + 1)}}}' for i in range(len(js)))))
        body.append('Method & ' + ' & '.join((f'$10^{{{int(round(math.log10(N)))}}}$' for _ in js for N in Ns)) + ' & ' * (nN * (k - len(js))) + ' \\\\')
        body.append('\\midrule')
        cell = {}
        for j in js:
            for N in Ns:
                col = bold_min([text(j, m, N) for m in meths], [vals[j, m, N] for m in meths])
                for m, c in zip(meths, col):
                    cell[j, m, N] = c
        for mth in meths:
            if mth == 'NUTS-X':
                label = 'NUTS (extrap.)'
            else:
                fam, _ = method_style(mth)
                label = (fam + TEX_METHOD[mth[len(fam):]].replace(' ', '')).replace('-δ', '$^\\delta$')
            body.append(label + ' & ' + ' & '.join((cell[j, mth, N] for j in js for N in Ns)) + ' & ' * (nN * (k - len(js))) + ' \\\\')
    body += ['\\bottomrule', '\\end{tabular}']
    pn = prefix_note(rows, [m for m in meths if m != 'NUTS-X'], Ns)
    cap = f'{title}: RMSE at sampling cost $N$.' + ('' if pn.startswith('Every') else ' ' + pn) + cfg.get('table_note', '') + xnote
    return table_env(body, cap, f'tab:{t}-rmse')

def tables(t):
    from common import make, truth
    target = make(t)
    tr = truth(target)
    cfg = C.TARGETS[t]
    jh = cfg['headline']
    omit = set(cfg.get('table_omit', []))
    if cfg.get('skip_pmala'):
        omit |= {'GI-pMALA', 'GI-pMALA-LGM', 'GI-pMALA-LGM+CV'}
    names = tex_names(t, target)
    nF = len(names)
    title = TEX_TITLE.get(t, t.replace('_', '\\_'))
    lines = []
    rows_v = []
    for transport, s, label in (('flow', 'rwm', 'TGI-RWM'), ('flow_d', 'rwm', 'TGI-RWM'), ('flow', 'mala', 'TGI-MALA'), ('flow_d', 'mala', 'TGI-MALA'), ('laplace', 'mala', 'GI-MALA'), ('laplace', 'pmala', 'GI-pMALA')):
        got = load(t, 'vrf', transport, s)
        if got is None or label in omit:
            continue
        if transport == 'flow_d':
            label += '$^\\delta$'
        meta, runs = got
        r = runs[0]
        m = np.array([x[jh] for x in r['chain']['main']['mu_fixed']], dtype=float)
        dev = m.mean() - tr['mean'][jh]
        signif = abs(dev) > 4 * m.std(ddof=1) / math.sqrt(m.size)
        stuck = signif and abs(dev) / math.sqrt(tr['var'][jh]) > 0.05
        cells = []
        for j in range(nF):
            gm = grid_metrics(meta, runs, j, 'main')
            e = gm['ess_n'][0]
            cells.append('-- & --' if stuck else f'{e:.2f}' if e >= 0.1 else f'{e:.2g}' if e >= 0.01 else sci(e))
            if not stuck:
                cells[-1] += f" & {num(gm['vrf'][0])}"
        bias = dev / math.sqrt(tr['var'][jh])
        bias_s = f'{bias:+.3g}' if abs(bias) >= 0.01 or bias == 0 else ('+' if bias > 0 else '') + sci(bias)
        rows_v.append(f"{label} & {r['gamma']:.2f} & " + ' & '.join(cells) + f" & {bias_s}{('$^*$' if signif else '')} \\\\")
    if rows_v:
        R, n = (C.CHAINS, C.N_STEPS)
        body = ['\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}lc' + 'cc' * nF + 'c@{}}', '\\toprule', ' & & ' + ' & '.join((f'\\multicolumn{{2}}{{c}}{{{nm}}}' for nm in names)) + ' & \\\\', ''.join((f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(nF))), 'Method & $\\gamma$ & ' + ' & '.join(['ESS$/n$ & VRF'] * nF) + ' & bias$/$sd \\\\', '\\midrule', *rows_v, '\\bottomrule', '\\end{tabular}']
        cap = f'{title}: ESS$/n$ and VRF in the chain experiment; last column: bias of the control-variate estimate of {names[jh]}.'
        if 'TGI-MALA' in omit:
            cap += ' TGI-MALA is omitted: its chains stall at the seam between the modes.'
        if cfg.get('skip_pmala'):
            cap += SKIP_PMALA
        lines += table_env(body, cap, f'tab:{t}-vrf', wide=nF > 4)
    rows = bench_rows(t)
    if rows:
        have = list(dict.fromkeys((r['method'] for r in rows)))
        meths = [f + x for f in FAMILY for x in ('+CV', '+ZV2', '+ZVz', '') if f + x in have and f + x not in omit]
        rmse_rows, eff_rows = ([], [])
        for mth in meths:
            fam, _ = method_style(mth)
            at, best = ([], [])
            for j in range(nF):
                mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
                within = [e for c, e, _, _ in curve(rows, mth, mu, sd, j, False) if c <= 1.1 * C.COST_REF]
                at.append(min(within) if within else float('nan'))
                allp = [(c, e) for c, e, _, _ in curve(rows, mth, mu, sd, j, False) if c <= C.COST_CAP]
                best.append(max((1 / (c * e * e) for c, e in allp)) if allp else float('nan'))
            lab = (fam + TEX_METHOD[mth[len(fam):]]).replace('-δ', '$^\\delta$')
            rmse_rows.append((lab, at))
            eff_rows.append(lab + ' & ' + ' & '.join((num(v) for v in best)) + ' \\\\')
        cols = [bold_min([num(at[j]) for _, at in rmse_rows], [at[j] for _, at in rmse_rows]) for j in range(nF)]
        rmse_rows = [lab + ' & ' + ' & '.join((cols[j][i] for j in range(nF))) + ' \\\\' for i, (lab, _) in enumerate(rmse_rows)]
        head = ['Method & ' + ' & '.join(names) + ' \\\\', '\\midrule']
        spec = '\\begin{tabular}{@{}l' + 'c' * nF + '@{}}'
        e10 = int(round(math.log10(C.COST_REF)))
        common_txt = f' over {C.REPLICATES} independent estimates; the cost counts target evaluations (a joint evaluation of $\\log\\pi$ and its gradient counts 2) and includes the flow training for TGI, at the best training checkpoint and sampling budget (lower envelope).'
        if t not in SYNTHETIC:
            (out_dir(t) / 'tables.tex').write_text('\n'.join(rmse_by_cost(t, cfg, rows, meths, names, tr, title)) + '\n')
            print(f'  output/{t}/tables.tex')
            return
        else:
            lines += table_env([spec, '\\toprule', *head, *rmse_rows, '\\bottomrule', '\\end{tabular}'], f'{title}: RMSE at total cost $10^{{{e10}}}$.' + cfg.get('table_note', '') + (SKIP_PMALA if cfg.get('skip_pmala') else ''), f'tab:{t}-rmse', wide=nF > 4)
        if False:
            lines += table_env([spec, '\\toprule', *head, *eff_rows, '\\bottomrule', '\\end{tabular}'], f'{title}: best efficiency $\\mathrm{{Var}}_\\pi F/(\\text{{cost}}\\times\\mathrm{{MSE}})$ over the configurations with total cost at most {num(C.COST_CAP)}' + common_txt + cfg.get('table_note', '') + (SKIP_PMALA if cfg.get('skip_pmala') else ''), f'tab:{t}-eff', wide=nF > 4)
    tp = raw_dir(t) / 'timing.json'
    if tp.exists() and False:
        T = json.loads(tp.read_text())
        trows = []
        for r in T['rows']:
            if r['method'] in omit:
                continue
            signif = abs(r.get('bias_z', 0.0)) > 4
            stuck = signif and abs(r['bias']) > 0.05
            sh = lambda v: '--' if stuck else num(v)
            trows.append(f"{r['method']} & {r['gamma']:.2f} & {r['seconds']:.1f} & {('--' if stuck else format(r['ess_n'], '.3f'))} & {sh(r['vrf'])} & {sh(r['ess_per_s'])} & {sh(r['effvrf_per_s'])} & {num(r['bias'], True)}{('$^*$' if signif else '')} \\\\")
        body = ['\\begin{tabular}{@{}lccccccc@{}}', '\\toprule', 'Method & $\\gamma$ & time (s) & ESS$/n$ & VRF & ESS/s & ESS$\\times$VRF/s & bias$/$sd \\\\', '\\midrule', *trows, '\\bottomrule', '\\end{tabular}']
        cap = f"{title}: wall-clock efficiency of the samplers we implement, headline function {names[jh]}, on a trained flow (training not counted; TGI) or the Laplace affine map (GI): {T['chains']} chains run in one batch on one {T['device']}, {T['burn']} burn-in and {T['steps']} recorded steps, control variates recorded on the fly ($\\theta=(1,-1)$). ESS/s counts the effective draws of all chains per second; $^*$ as in Table~\\ref{{" + f'tab:{t}-vrf' + '}.' + (SKIP_PMALA if cfg.get('skip_pmala') else '')
        lines += table_env(body, cap, f'tab:{t}-time')
    if lines:
        (out_dir(t) / 'tables.tex').write_text('\n'.join(lines) + '\n')
        print(f'  output/{t}/tables.tex')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True, choices=list(C.TARGETS))
    ap.add_argument('--only', nargs='*', default=None, choices=['gamma', 'pruning', 'benchmark', 'flow', 'tables', 'convergence', 'overview', 'posterior', 'estimators', 'post_est', 'samplers', 'analytic'])
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    todo = args.only or ['gamma', 'pruning', 'benchmark', 'flow', 'tables', 'convergence', 'overview', 'posterior', 'estimators', 'post_est']
    names = ['flow', *C.TARGETS[t]['analytic']]
    if 'gamma' in todo:
        for tr in names:
            for s in C.SAMPLERS:
                fig_gamma(t, tr, s)
    if 'pruning' in todo:
        for tr in names:
            for s in C.SAMPLERS:
                fig_pruning(t, tr, s)
    if 'benchmark' in todo:
        fig_benchmark(t, methods=C.BENCH_PLOT)
        fig_benchmark(t, stem='benchmark_all', big=True)
    if 'flow' in todo:
        fig_flow(t)
    if 'tables' in todo:
        tables(t)
    if 'convergence' in todo:
        fig_convergence(t)
    if 'overview' in todo and (raw_dir(t) / 'convergence.npz').exists():
        fig_overview(t)
    if 'posterior' in todo:
        fig_posterior(t)
    if 'estimators' in todo:
        fig_estimators(t)
    if 'post_est' in todo:
        fig_post_est(t)
    if 'samplers' in todo:
        fig_samplers(t)
    if 'analytic' in todo:
        fig_analytic_overall(t)
if __name__ == '__main__':
    main()
