import json
import math
from pathlib import Path
import numpy as np
from common import RESULT_TAG, raw_dir
from figures import GREEN, PURPLE, plt_, out_dir, rmse
from workflow import config as C
TARGETS = ['pdb_garch', 'banana', 'pdb_one_comp', 'pdb_dugongs', 'lotka_volterra', 'gmix_d10_h6', 'funnel', 'pdb_gp_pois', 'sparse_logistic_german', 'phi4_N32', 'eight_schools', 'tmix_d10_h6']
NAME = {'pdb_garch': 'GARCH', 'banana': 'Banana', 'pdb_one_comp': 'One-compartment', 'pdb_dugongs': 'Dugongs', 'lotka_volterra': 'Lotka--Volterra', 'gmix_d10_h6': 'Mixture', 'funnel': 'Funnel', 'pdb_gp_pois': 'GP Poisson', 'sparse_logistic_german': 'Sparse logistic', 'pdb_surgical': 'Surgical', 'phi4_N32': '$\\phi^4$ field', 'eight_schools': 'Eight schools', 'tmix_d10_h6': '$t$ mixture'}
FIGNAME = {**NAME, 'pdb_one_comp': 'One-comp.'}
DAGGER = {'pdb_one_comp': '$^\\dagger$', 'pdb_gp_pois': '$^\\dagger$', 'eight_schools': '$^\\ddagger$'}
MAIN = [('SNIS', 'SNIS+CV'), ('TGI-RWM', 'TGI-RWM+CV'), ('TESS', 'TESS+CV'), ('NeuTra', 'NeuTra+ZVz')]
FULL = ['SNIS', 'SNIS+CV', 'TGI-RWM', 'TGI-RWM+CV', 'TGI-MALA', 'TGI-MALA+CV', 'TESS', 'TESS+CV', 'NeuTra', 'NeuTra+ZVz']
OUT = Path(__file__).resolve().parent.parent

def load(t):
    p = raw_dir(t) / f'chainstats{RESULT_TAG}.json'
    if not p.exists():
        return None
    o = json.loads(p.read_text())
    d = {r['method']: r for r in o['runs']}
    if not all((m in d for pair in MAIN for m in pair)):
        return None
    return (o, d)

def tex_num(v):
    if v is None or not np.isfinite(v) or v <= 0:
        return '--'
    e = int(math.floor(math.log10(v)))
    m = v / 10 ** e
    if m >= 9.95:
        m, e = (1.0, e + 1)
    return f'${m:.1f}{{\\cdot}}10^{{{e}}}$'

def bold_set(d, methods, j):
    vals = {m: d[m]['acc_per_s'][j] for m in methods if m in d}
    vals = {m: v for m, v in vals.items() if v is not None and np.isfinite(v)}
    if not vals:
        return set()
    best = max(vals, key=vals.get)
    blo = d[best]['acc_per_s_ci'][0][j]
    out = set()
    for m, v in vals.items():
        hi = d[m]['acc_per_s_ci'][1][j]
        if v >= 0.9 * vals[best] or (hi is not None and hi >= blo):
            out.add(m)
    return out

def agg_value(d, m, h, agg):
    a = np.asarray(d[m]['acc_per_s'], dtype=float)
    return float(a[h]) if agg == 'headline' else float(np.nanmedian(a))

def agg_ratio(d, num, den, h, agg):
    a = np.asarray(d[num]['acc_per_s'], dtype=float) / np.asarray(d[den]['acc_per_s'], dtype=float)
    return float(a[h]) if agg == 'headline' else float(np.nanmedian(a))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg', choices=['median', 'headline'], default='headline', help='headline: the main test function (config main_fn, the median function); median: per-method median over the functions')
    ap.add_argument('--full', action='store_true', help='only the appendix table of every function')
    ap.add_argument('--boxplots', action='store_true', help='only Figure A3 (box plots)')
    ap.add_argument('--vs-neutra', action='store_true', help='only the scatter against NeuTra+ZV')
    ap.add_argument('--nuts', action='store_true', help='only Table 2 (against NUTS)')
    ap.add_argument('--main5', action='store_true', help='only the main-text table: main function, the five CV methods')
    a = ap.parse_args()
    if a.main5:
        return main_table5()
    if a.full:
        return full_table_wallclock()
    if a.boxplots:
        return fig_boxplots()
    if a.vs_neutra:
        return fig_vs_neutra()
    if a.nuts:
        return table_nuts()
    agg = a.agg
    sfx = RESULT_TAG + ('' if agg == 'headline' else '_median')
    rows = []
    for t in TARGETS:
        x = load(t)
        if x is not None:
            o, d = x
            rows.append((t, o, d, d['SNIS']['max_wN_max']))
    rows.sort(key=lambda r: r[3])
    L = ['\\begin{tabular}{@{}l cccc r@{}}', '\\toprule', 'Target & SNIS+CV & TGI-RWM+CV & TESS+CV & NeuTra+ZV$_z$ & $\\max w\\cdot N$ \\\\', '\\midrule']
    md = ['| target | SNIS+CV | TGI-RWM+CV | TESS+CV | NeuTra+ZVz | max w·N (max / median) |', '|---|---|---|---|---|---|']
    for t, o, d, mw in rows:
        h = C.TARGETS[t].get('main_fn', o['headline'])
        cv = [c for _, c in MAIN]
        if True:
            vals = {c: agg_value(d, c, h, agg) for c in cv}
            vb = max((v for v in vals.values() if np.isfinite(v)))
            b = {c for c, v in vals.items() if np.isfinite(v) and v >= 0.9 * vb}
        cells, mcells = ([], [])
        for p, c in MAIN:
            v = agg_value(d, c, h, agg)
            g = agg_ratio(d, c, p, h, agg)
            s = tex_num(v) + (f' ($\\times{g:.0f}$)' if np.isfinite(g) and g >= 1.5 else f' ($\\times{g:.1f}$)' if np.isfinite(g) else '')
            cells.append(f'\\textbf{{{s}}}' if c in b else s)
            mcells.append(('**' if c in b else '') + f'{v:.1e} (×{g:.1f})' + ('**' if c in b else ''))
        L.append(f"{NAME[t]}{DAGGER.get(t, '')} & " + ' & '.join(cells) + f' & {mw:.0f} \\\\')
        md.append(f'| {t} | ' + ' | '.join(mcells) + f" | {mw:.0f} / {d['SNIS']['max_wN_median']:.0f} |")
    L += ['\\bottomrule', '\\end{tabular}']
    (out_dir('paper') / f'table_samemain{sfx}.tex').write_text('\n'.join(L) + '\n')
    F = ['\\begin{tabular}{@{}ll ' + 'c' * len(FULL) + '@{}}', '\\toprule', 'Target & $f$ & ' + ' & '.join((m.replace('ZVz', 'ZV$_z$') for m in FULL)) + ' \\\\', '\\midrule']
    for t, o, d, mw in rows:
        names = [n for n, _, _ in __import__('common').make(t).test_functions]
        for j, n in enumerate(names):
            b = bold_set(d, FULL, j)
            cells = []
            for m in FULL:
                if m not in d:
                    cells.append('--')
                    continue
                v = d[m]['acc_per_s'][j]
                lo, hi = (d[m]['acc_per_s_ci'][0][j], d[m]['acc_per_s_ci'][1][j])
                s = tex_num(v)
                cells.append(f'\\textbf{{{s}}}' if m in b else s)
            first = f"{NAME[t]}{DAGGER.get(t, '')}" if j == 0 else ''
            F.append(f'{first} & {n} & ' + ' & '.join(cells) + ' \\\\')
        F.append('\\addlinespace')
    F += ['\\bottomrule', '\\end{tabular}']
    (out_dir('paper') / f'table_samefull{RESULT_TAG}.tex').write_text('\n'.join(F) + '\n')
    plt = plt_()
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    EST = [('TGI-RWM+CV', 'TGI-RWM+CV', GREEN, 'D'), ('TGI-MALA+CV', 'TGI-MALA+CV', '#d95f02', 's'), ('TESS+CV', 'TESS+CV', '#e6ab02', '^'), ('NeuTra+ZVz', 'NeuTra+ZV', PURPLE, 'v')]
    for t, o, d, mw in rows:
        h = C.TARGETS[t].get('main_fn', o['headline'])
        a_s = np.asarray(d['SNIS+CV']['acc_per_s'], dtype=float)
        ys = []
        for m, _, col, mk in EST:
            rr = np.asarray(d[m]['acc_per_s'], dtype=float) / a_s
            ys.append(float(rr[h]) if agg == 'headline' else float(np.nanmedian(rr)))
        k = int(np.argmax(ys))
        _, _, col, mk = EST[k]
        ax.plot(mw, ys[k], mk, ms=5, mec=col, mfc=col, alpha=0.9, mew=0.6, ls='none')
        pos = {'pdb_one_comp': 'below', 'gmix_d10_h6': 'below', 'sparse_logistic_german': 'below', 'lotka_volterra': 'left', 'phi4_N32': 'right', 'pdb_garch': 'above_r'}.get(t, 'above')
        off, ha, va = {'below': ((0, -5), 'center', 'top'), 'left': ((-5, 0), 'right', 'center'), 'right': ((5, 0), 'left', 'center'), 'above': ((0, 5), 'center', 'bottom'), 'above_r': ((4, 5), 'center', 'bottom')}[pos]
        ax.annotate(FIGNAME[t].replace('$\\phi^4$ field', '$\\phi^4$').replace('--', chr(8211)), (mw, ys[k]), fontsize=5.5, xytext=off, textcoords='offset points', ha=ha, va=va)
    ax.axhline(1.0, color='k', lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\max\\, w\\cdot N$', fontsize=8)
    ax.set_ylabel('ratio' + ('' if agg == 'headline' else ', median over $f$'), fontsize=8)
    ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker=mk, color=col, ls='none', ms=5) for _, _, col, mk in EST], [n for _, n, _, _ in EST], fontsize=6.5, loc='upper left', frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir('paper') / f'fig_maxw{sfx}.{ext}', dpi=200)
    p = OUT / 'results' / 'revision' / f'final_tables{sfx}.md'
    p.write_text(f'# Final same-flow comparison ({agg} over the test functions, accuracy per second per estimate)\n\n' + '\n'.join(md) + '\n')
    print('\n'.join(md))
FULL2 = [('SNIS', 'SNIS+CV'), ('TGI-RWM', 'TGI-RWM+CV'), ('TGI-MALA', 'TGI-MALA+CV'), ('TESS', 'TESS+CV'), ('NeuTra', 'NeuTra+ZVz')]
FAIL_RMSE = 1.0

def short(v):
    if v is None or not np.isfinite(v) or v <= 0:
        return '--'
    e = int(math.floor(math.log10(v)))
    m = round(v / 10 ** e, 1)
    if m >= 10:
        m, e = (m / 10, e + 1)
    return f'{m:.1f}e{e}'

def ref_mcse(t, sd):
    import pathlib
    p = pathlib.Path(raw_dir(t)).parent.parent / 'references' / f'reference_{t}.json'
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    mc = r.get('mcse')
    return None if mc is None else np.asarray(mc, float) / sd

def full_table_wallclock():
    import csv
    from common import make, truth
    rows = []
    for t in TARGETS:
        x = load(t)
        if x is not None:
            rows.append((t, *x, x[1]['SNIS']['max_wN_max']))
    rows.sort(key=lambda r: r[3])
    methods = [m for pair in FULL2 for m in pair]
    L = ['\\begin{tabular}{@{}ll' + 'r' * len(methods) + 'r@{}}', '\\toprule', ' & & ' + ' & '.join((f'\\multicolumn{{2}}{{c}}{{{p}}}' for p, _ in FULL2)) + ' & \\\\', ''.join((f'\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(FULL2)))), 'Target & $F$ & ' + ' & '.join(('plain & ' + ('+ZV' if 'ZV' in c else '+CV') for _, c in FULL2)) + ' & $\\max w{\\cdot}N$ \\\\', '\\midrule']
    csv_rows, relw, failed = ([], [], set())

    def wn(v):
        return f'{v:.0f}' if v < 100 else short(v)

    def cellval(d, m, j):
        rm = d[m]['rmse'][j]
        if rm is None or not np.isfinite(rm) or rm > FAIL_RMSE:
            return None
        return d[m]['acc_per_s'][j]
    for t, o, d, mw in rows:
        target = make(t)
        sd = np.sqrt(np.asarray(truth(target)['var'], float))
        names = C.TARGETS[t].get('tex_names') or [n.replace('_', '\\_') for n, _, _ in target.test_functions]
        nf = len(names)
        mc = ref_mcse(t, sd)
        for j in range(nf):
            for m in methods:
                lo, hi = (d[m]['acc_per_s_ci'][0][j], d[m]['acc_per_s_ci'][1][j])
                v = cellval(d, m, j)
                if v is None:
                    failed.add((t, m))
                elif lo and hi:
                    relw.append((hi - lo) / 2 / v)
                csv_rows.append([t, names[j], m, d[m]['acc_per_s'][j], lo, hi, d[m]['rmse'][j], d[m].get('bias_chain', d[m].get('bias_rep'))[j], mw])
        trows = []
        if t == 'sparse_logistic_german':
            ws = list(range(1, nf))
            h = C.TARGETS[t].get('main_fn', o['headline'])
            for j in [h, 0, 1]:
                lab = f'${names[j]}$' + ('\\,$\\star$' if j == h else '')
                trows.append((lab, {m: cellval(d, m, j) for m in methods}, j))
            med = {m: float(np.median([cellval(d, m, j) or np.nan for j in ws])) for m in methods}
            trows.append(('median over $w_j$', med, None))
            for lab, fn in (('max over $w_j$', max), ('min over $w_j$', min)):
                ext = {}
                for m in methods:
                    v = [x for x in (cellval(d, m, j) for j in ws) if x]
                    ext[m] = float(fn(v)) if v else None
                trows.append((lab, ext, None))
        else:
            h = C.TARGETS[t].get('main_fn', o['headline'])
            for j in [h] + [k for k in range(nf) if k != h]:
                lab = f'${names[j]}$' + ('\\,$\\star$' if j == h else '')
                trows.append((lab, {m: cellval(d, m, j) for m in methods}, j))
        for i, (lab, vals, j) in enumerate(trows):
            num = {m: v for m, v in vals.items() if isinstance(v, float) and np.isfinite(v)}
            best = max(num.values()) if num else None
            cells = []
            for m in methods:
                v = vals[m]
                if isinstance(v, tuple):
                    cells.append(f'{short(v[0])}--{short(v[1])}')
                    continue
                s = short(v) if v is not None and np.isfinite(v) else '--'
                if j is not None and v is not None and (mc is not None) and (mc[j] >= d[m]['rmse'][j] / math.sqrt(2.0)):
                    s += '$^\\dagger$'
                if v is not None and best and np.isfinite(v) and (v >= 0.9 * best):
                    s = f'\\textbf{{{s}}}'
                cells.append(s)
            tname = NAME[t] + ('$^\\ddagger$' if t == 'eight_schools' else '') if i == 0 else ''
            L.append(f'{tname} & {lab} & ' + ' & '.join(cells) + (f' & {wn(mw)}' if i == 0 else ' & ') + ' \\\\')
        L.append('\\addlinespace')
    L[-1] = '\\bottomrule'
    L.append('\\end{tabular}')
    hw = float(np.median(relw)) if relw else float('nan')
    cap = f"\\caption{{Precision per second, $1/(\\mathrm{{RMSE}}^2\\times\\text{{seconds per estimate}})$ with the RMSE in posterior standard deviations, of every method on every test function: $N=10^6$ target evaluations, $R=100$ estimates on the final flow, $200$ chains of cost $5000$ per estimate. Targets are ordered by the largest normalized importance weight of the flow times $N$ (last column). $\\star$ marks the main test function: the median over the test functions of the ratio of the best of SNIS+CV, TGI-RWM+CV and TESS+CV to NeuTra+ZV (the upper median for an even number of functions). For the sparse logistic regression: the main function, $\\log\\tau$, $w_1$ and the median, maximum and minimum over the $25$ coefficients $w_j$. Entries within $10\\%$ of the best in a row are in bold. $\\dagger$: the reference value's Monte Carlo error accounts for at least half of the mean squared error; $\\ddagger$: the flow-based estimators share a bias in the latent space, which dominates the error in $\\log\\tau$ (about $0.01$ posterior standard deviations) and makes up $25$--$50\\%$ of the mean squared error of the chains on $\\theta_1$; --: the control variates of TGI-MALA diverge at the seam between the modes. Bootstrap $95\\%$ intervals (median relative half-width ${100 * hw:.0f}\\%$) are given in the supplementary CSV.}}"
    L = ['\\addlinespace[1.5pt]' if l == '\\addlinespace' else l for l in L]
    env = ['\\begin{table}[p]', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\renewcommand{\\arraystretch}{0.93}', cap, '\\label{tab:wallclock-full}'] + L + ['\\end{table}']
    (out_dir('paper') / f'table_wallclock_full{RESULT_TAG}.tex').write_text('\n'.join(env) + '\n')
    with open(out_dir('paper') / f'table_wallclock_full{RESULT_TAG}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['target', 'function', 'method', 'precision_per_second', 'ci95_low', 'ci95_high', 'rmse_over_sd', 'bias_over_sd', 'max_wN'])
        w.writerows(csv_rows)
    nrow = sum((1 for l in L if l.endswith('\\\\'))) - 2
    print(f'rows: {nrow}, median relative half-width of the 95% intervals: {hw:.2f}, failures: {sorted(failed)}')

def fig_boxplots():
    from common import make, truth
    rows = []
    for t in TARGETS:
        x = load(t)
        if x is not None:
            rows.append((t, *x, x[1]['SNIS']['max_wN_max']))
    rows.sort(key=lambda r: r[3])
    M = [('SNIS', 'SNIS+CV', 'SNIS', '#d95f02'), ('TGI-RWM', 'TGI-RWM+CV', 'TGI-RWM', GREEN), ('TESS', 'TESS+CV', 'TESS', '#e6ab02'), ('NeuTra', 'NeuTra+ZVz', 'NeuTra', PURPLE)]
    pos = [k + 1 for k in range(len(M))]
    cols = [col for _, _, _, col in M]
    alphas = [0.75] * len(M)
    keys = [c for _, c, _, _ in M]
    plt = plt_()
    nc = 4
    nr = math.ceil(len(rows) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(6.75, 1.8 * nr), constrained_layout=True)
    for ax, (t, o, d, mw) in zip(axes.flat, rows):
        z = np.load(raw_dir(t) / f'chainstats{RESULT_TAG}.npz')
        tr = truth(make(t))
        h = C.TARGETS[t].get('main_fn', o['headline'])
        mu, sd = (tr['mean'][h], math.sqrt(tr['var'][h]))
        data = [(z[f'{m}_est'][:, h] - mu) / sd for m in keys]
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=True, flierprops=dict(marker='o', ms=1.5, mew=0.4, alpha=0.7), medianprops=dict(color='k', lw=0.7), whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
        for patch, col, a in zip(bp['boxes'], cols, alphas):
            patch.set_facecolor(col)
            patch.set_alpha(a)
        for fl, col in zip(bp['fliers'], cols):
            fl.set_markeredgecolor(col)
        ax.axhline(0.0, color='k', lw=0.6, ls='--')
        names = C.TARGETS[t].get('tex_names')
        lab = names[h] if names else ''
        for x, y in (('\\mathbf 1', '1'), ('\\mathbb 1', '1'), ('\\tfrac', '\\frac')):
            lab = lab.replace(x, y)
        fn = f'${lab}$' if lab else ''
        ax.set_title(f"{FIGNAME[t].replace('--', chr(8211))}: {fn}\n$\\max\\, w\\cdot N={mw:.0f}$", fontsize=5.5)
        ax.set_xticks([])
        ax.set_xlim(0.5, len(M) + 0.5)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_fontsize(5)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis('off')
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.8) for _, _, _, c in M]
    labels = ['SNIS+CV', 'TGI-RWM+CV', 'TESS+CV', 'NeuTra+ZV']
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.supylabel('(estimate $-$ reference) / posterior sd', fontsize=6)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir('paper') / f'fig_boxplots{RESULT_TAG}.{ext}', dpi=200)
    print(out_dir('paper') / f'fig_boxplots{RESULT_TAG}.pdf')

def fig_vs_neutra():
    rows = []
    for t in TARGETS:
        x = load(t)
        if x is not None:
            rows.append((t, *x, x[1]['SNIS']['max_wN_max']))
    rows.sort(key=lambda r: r[3])
    OURS = [('SNIS+CV', 'SNIS+CV', '#d95f02', 'o'), ('TGI-RWM+CV', 'TGI-RWM+CV', GREEN, 'D'), ('TESS+CV', 'TESS+CV', '#e6ab02', '^')]
    plt = plt_()
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for t, o, d, mw in rows:
        h = C.TARGETS[t].get('main_fn', o['headline'])
        z = d['NeuTra+ZVz']['acc_per_s'][h]
        ys = [d[m]['acc_per_s'][h] / z for m, _, _, _ in OURS]
        k = int(np.argmax(ys))
        _, _, col, mk = OURS[k]
        ax.plot(mw, ys[k], mk, ms=5, mec=col, mfc=col, alpha=0.9, mew=0.6, ls='none')
        pos = {'pdb_one_comp': 'below', 'gmix_d10_h6': 'below', 'sparse_logistic_german': 'below', 'lotka_volterra': 'left', 'phi4_N32': 'right', 'pdb_garch': 'above_r'}.get(t, 'above')
        off, ha, va = {'below': ((0, -5), 'center', 'top'), 'left': ((-5, 0), 'right', 'center'), 'right': ((5, 0), 'left', 'center'), 'above': ((0, 5), 'center', 'bottom'), 'above_r': ((4, 5), 'center', 'bottom')}[pos]
        ax.annotate(FIGNAME[t].replace('$\\phi^4$ field', '$\\phi^4$').replace('--', chr(8211)), (mw, ys[k]), fontsize=5.5, xytext=off, textcoords='offset points', ha=ha, va=va)
    ax.axhline(1.0, color='k', lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\max\\, w\\cdot N$ (SNIS, largest over replicates)')
    ax.set_ylabel('best of ours / NeuTra+ZV (precision per s)')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], marker=mk, color=col, ls='none', ms=4) for _, _, col, mk in OURS], [n for _, n, _, _ in OURS], fontsize=5, loc='upper left', frameon=False)
    for ext in ('pdf', 'png'):
        fig.savefig(out_dir('paper') / f'fig_vs_neutra{RESULT_TAG}.{ext}', dpi=200)
    print(out_dir('paper') / f'fig_vs_neutra{RESULT_TAG}.pdf')

def table_nuts():
    from common import make, truth
    import revision as RV
    POST = ['pdb_garch', 'pdb_one_comp', 'lotka_volterra', 'pdb_dugongs', 'pdb_gp_pois', 'eight_schools', 'sparse_logistic_german', 'phi4_N32', None, 'banana', 'funnel', 'gmix_d10_h6']
    M = [('SNIS+CV', 'SNIS+CV'), ('TGI-RWM+CV', 'TGI-RWM+CV'), ('TESS+CV', 'TESS+CV'), ('NeuTra+ZVz', 'NeuTra+ZV')]

    def sci(v):
        if v is None or not np.isfinite(v):
            return '--'
        e = int(math.floor(math.log10(v)))
        m = round(v / 10 ** e, 1)
        if m >= 10:
            m, e = (m / 10, e + 1)
        return f'{m:.1f}e{e}'
    L = ['\\begin{tabular}{@{}ll cccccc@{}}', '\\toprule', 'Target & $F$ & NUTS+ZV & GI-MALA+CV & ' + ' & '.join((n for _, n in M)) + ' \\\\', '\\midrule']
    md = ['| target | F | NUTS+ZV | GI-MALA+CV | ' + ' | '.join((n for _, n in M)) + ' |', '|---' * (len(M) + 4) + '|']
    for t in POST:
        if t is None:
            L.append('\\midrule')
            continue
        x = load(t)
        if x is None:
            continue
        o, d = x
        target = make(t)
        tr = truth(target)
        h = C.TARGETS[t].get('main_fn', o['headline'])
        names = C.TARGETS[t].get('tex_names') or [n for n, _, _ in target.test_functions]
        cfg, R = (C.TARGETS[t], RV.bench_rows(t))
        final = cfg['ckpts'][-1]
        ok = lambda m, N: [r for r in R if r['method'] == m and abs(r['budget'] - N) <= 1e-06 * N and (r.get('ckpt') in (None, final))]
        b = RV.baseline(t, tr, ok, [1000000.0])[0][h]
        ext = bool(cfg.get('nuts_extrapolate'))
        gi = ok('GI-MALA+CV', 1000000.0)
        g = rmse(gi, tr['mean'][h], math.sqrt(tr['var'][h]), h) if gi else None
        vals = [b, g] + [d[m]['rmse'][h] for m, _ in M]
        fin = [v for v in vals if v is not None and np.isfinite(v)]
        best = min(fin)
        cells = []
        for i, v in enumerate(vals):
            c = sci(v)
            if i == 0 and ext and np.isfinite(v):
                c += '$^\\dagger$'
            if i == 0 and (not np.isfinite(v)):
                c = '--$^\\ddagger$' if t == 'phi4_N32' else '--'
            if v is not None and np.isfinite(v) and (v <= 1.1 * best):
                c = f'\\textbf{{{c}}}'
            cells.append(c)
        L.append(f'{NAME[t]} & ${names[h]}$ & ' + ' & '.join(cells) + ' \\\\')
        md.append(f'| {t} | {names[h]} | ' + ' | '.join((sci(v) for v in vals)) + ' |')
    L += ['\\bottomrule', '\\end{tabular}']
    cap = '\\caption{RMSE in posterior standard deviations of the main test function (as in Table~\\ref{tab:wallclock}) at $N=10^6$ target evaluations, real posteriors (top) and synthetic targets (bottom). NUTS+ZV: NUTS with second-order zero-variance control variates, warm-up included, $50$ estimates (more accurate than plain NUTS on every target where both were run); $\\dagger$: plain NUTS extrapolated from the reference run as its Monte Carlo standard error times $\\sqrt{C_{\\rm ref}/N}$ (warm-up not counted); $\\ddagger$: NUTS stays in one mode; on the funnel the second-order ZV control variates integrate $v$ exactly. GI-MALA+CV: GI-MALA on the Laplace map with its control variates, $50$ estimates. The flow-based estimators ($100$ estimates) run on the final flow, whose training is not included. The smallest error in a row and those within $10\\%$ of it are in bold.}'
    env = ['\\begin{table}[t]', '\\centering', '\\small', cap, '\\label{tab:nuts}'] + L + ['\\end{table}']
    (out_dir('paper') / f'table_nuts{RESULT_TAG}.tex').write_text('\n'.join(env) + '\n')
    print('\n'.join(md))

def main_table5():
    from common import make, truth
    rows = []
    for t in TARGETS:
        x = load(t)
        if x is not None:
            rows.append((t, *x, x[1]['SNIS']['max_wN_max']))
    rows.sort(key=lambda r: r[3])
    methods = [c for _, c in FULL2]
    L = ['\\begin{tabular}{@{}l rrrrr@{}}', '\\toprule', 'Target & ' + ' & '.join((p for p, _ in FULL2)) + ' \\\\', '\\midrule']
    md = ['| target | F | ' + ' | '.join((p for p, _ in FULL2)) + ' | max wN |', '|---' * (len(FULL2) + 3) + '|']
    for t, o, d, mw in rows:
        target = make(t)
        sd = np.sqrt(np.asarray(truth(target)['var'], float))
        names = C.TARGETS[t].get('tex_names') or [n.replace('_', '\\_') for n, _, _ in target.test_functions]
        j = C.TARGETS[t].get('main_fn', o['headline'])
        mc = ref_mcse(t, sd)
        vals = {}
        for m in methods:
            rm = d[m]['rmse'][j]
            vals[m] = None if rm is None or not np.isfinite(rm) or rm > FAIL_RMSE else d[m]['acc_per_s'][j]
        best = max((v for v in vals.values() if v))
        cells = []
        for pm, m in FULL2:
            v = vals[m]
            c = short(v) if v else '--'
            if v and mc is not None and (mc[j] >= d[m]['rmse'][j] / math.sqrt(2.0)):
                c += '$^\\dagger$'
            if v and v >= 0.9 * best:
                c = f'\\textbf{{{c}}}'
            if v:
                vr = (d[pm]['rmse'][j] / d[m]['rmse'][j]) ** 2
                c += f' ({vr:.0f})' if vr >= 10 else f' ({vr:.1f})'
            cells.append(c)
        wn = f'{mw:.0f}' if mw < 100 else short(mw)
        tname = FIGNAME[t] + ('$^\\ddagger$' if t == 'eight_schools' else '')
        L.append(f'{tname} & ' + ' & '.join(cells) + ' \\\\')
        md.append(f'| {t} | {names[j]} | ' + ' | '.join((short(vals[m]) if vals[m] else '--' for m in methods)) + f' | {wn} |')
    L += ['\\bottomrule', '\\end{tabular}']
    cap = '\\caption{Precision per second, $1/(\\mathrm{RMSE}^2\\times\\text{seconds per estimate})$ with the RMSE in posterior standard deviations, of the main test function of each target (listed in Tables~\\ref{tab:exp-synthetic} and~\\ref{tab:exp-real}): $N=10^6$ target evaluations, $100$ estimates on the same trained flow, $200$ chains of cost $5000$ per estimate. Every method uses its control variates: ours (Hermite terms of the latent draws) for SNIS, TGI-RWM, TGI-MALA and TESS, second-order zero-variance control variates in the latent space for NeuTra; in parentheses the variance reduction factor of the control variates, the mean squared error without them over that with them on the same run. Targets are ordered by the largest normalized importance weight of the flow times $N$, from $3$ (GARCH) to $1.3\\times10^5$ ($t$ mixture; Table~\\ref{tab:wallclock-full}). Entries within $10\\%$ of the best in a row are in bold. $\\dagger$: the Monte Carlo error of the reference value accounts for at least half of the mean squared error; $\\ddagger$: see Table~\\ref{tab:wallclock-full}; --: RMSE above one posterior standard deviation.}'
    env = ['\\begin{table*}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{5pt}', cap, '\\label{tab:wallclock}'] + L + ['\\end{table*}']
    (out_dir('paper') / f'table_main{RESULT_TAG}.tex').write_text('\n'.join(env) + '\n')
    print('\n'.join(md))
if __name__ == '__main__':
    main()
