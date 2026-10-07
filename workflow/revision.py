import argparse
import math
from pathlib import Path
import numpy as np
from common import make, raw_dir, setup, truth
from workflow import config as C
from figures import bench_rows, load, rmse
from experiments.metrics import grid_metrics
OUT = Path(__file__).resolve().parent.parent / 'results' / 'revision'
T1 = ['banana', 'pdb_garch', 'pdb_dugongs', 'pdb_one_comp']
T2 = ['pdb_garch', 'pdb_dugongs', 'pdb_one_comp', 'pdb_gp_pois', 'eight_schools', 'sparse_logistic_german', 'phi4_N32', 'lotka_volterra']

def nuts_extrapolated(t, tr, N):
    import json
    cfg = C.TARGETS[t]
    ref = json.loads((Path(__file__).resolve().parent.parent / 'results' / 'references' / f'reference_{t}.json').read_text())
    cref = ref['n_draws'] * ref['mean_leapfrog'] * 2.0
    mc = ref.get('mcse_plain', ref['mcse'])
    skip = set(cfg.get('nuts_extrapolate_skip', []))
    return np.array([np.nan if j in skip else mc[j] / math.sqrt(tr['var'][j]) * math.sqrt(cref / N) for j in range(len(tr['mean']))])

def baseline(t, tr, ok, N_list):
    if C.TARGETS[t].get('nuts_extrapolate'):
        return np.array([nuts_extrapolated(t, tr, N) for N in N_list])
    out = []
    for N in N_list:
        row = []
        for j in range(len(tr['mean'])):
            mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
            v = [rmse(ok(m, N), mu, sd, j) for m in ('NUTS', 'NUTS+ZV2') if ok(m, N)]
            row.append(min(v) if v else np.nan)
        out.append(row)
    return np.array(out)
T5 = ['pdb_garch', 'pdb_dugongs', 'pdb_one_comp']
T5_CKPTS = [500, 1000, 2000, 5000, 10000]

def f3(v):
    return '--' if v is None or not np.isfinite(v) else f'{v:.3g}' if 0.001 <= abs(v) < 10000.0 else f'{v:.2e}'

def md(head, rows):
    return '\n'.join(['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]) + '\n'

def pooled_z(run, j, mu, key, ref_se=0.0):
    m = np.array([x[j] for x in run['chain']['main'][key]], dtype=float)
    return (m.mean() - mu) / math.sqrt(m.var(ddof=1) / m.size + ref_se ** 2)

def vrf_ci(run, j, n_boot=2000):
    c = run['chain']['main']
    f = np.array([x[j] for x in c['fbar']], dtype=float)
    m = np.array([x[j] for x in c['mu_fixed']], dtype=float)
    i = np.random.default_rng(0).integers(0, f.size, (n_boot, f.size))
    b = f[i].var(1) / m[i].var(1)
    return (np.quantile(b, 0.025), np.quantile(b, 0.975))

def ref_mcse(t):
    import json
    p = Path(__file__).resolve().parent.parent / 'results' / 'references' / f'reference_{t}.json'
    return json.loads(p.read_text()).get('mcse') if p.exists() else None

def task1():
    out = OUT / 'task1'
    out.mkdir(parents=True, exist_ok=True)
    head = ['target', 'test fn', 'γ (J)', 'acc (J)', 'ESS/n (J)', 'VRF (J)', 'ESS×VRF (J)', 'RMSE/sd (J)', 'acc (γ=1)', 'ESS/n (γ=1)', 'VRF (γ=1)', 'ESS×VRF (γ=1)', 'RMSE/sd (γ=1)']
    rows, checks, ratios, coarse = ([], [], [], [])
    for t, transport in [(t, 'flow') for t in T1] + [('banana', 'coarse')]:
        got = load(t, 'vrf', f'{transport}_g1', 'rwm')
        if got is None:
            checks.append(f'{t} {transport}: no run')
            continue
        meta, runs = got
        rj, r1 = (runs[0], runs[-1])
        target = make(t)
        tr = truth(target)
        names = meta['names']
        ref = load(t, 'vrf', transport, 'rwm') if transport == 'flow' else None
        brows = bench_rows(t) if transport == 'flow' else []
        final = C.TARGETS[t]['ckpts'][-1]
        mcse = ref_mcse(t)
        for j, nm in enumerate(names):
            gm = grid_metrics(meta, runs, j)
            ej, e1 = (gm['ess_n'][0], gm['ess_n'][-1])
            vj, v1 = (gm['vrf'][0], gm['vrf'][-1])
            mu, sd = (tr['mean'][j], math.sqrt(tr['var'][j]))
            rm = {}
            for key, m in (('J', 'TGI-RWM+CV'), ('1', 'TGI-RWM-γ1+CV')):
                sel = [r for r in brows if r['method'] == m and abs(r['budget'] - 1000000.0) < 1 and (r.get('ckpt') == final)]
                rm[key] = rmse(sel, mu, sd, j) if sel else float('nan')
            label = t + (' (coarse, F=%.2f)' % meta['F'] if transport == 'coarse' else '')
            rows.append([label, nm, f"{rj['gamma']:.3f}", f"{rj['acc']:.2f}", f3(ej), f3(vj), f3(ej * vj), f3(rm['J']), f"{r1['acc']:.2f}", f3(e1), f3(v1), f3(e1 * v1), f3(rm['1'])])
            (coarse if transport == 'coarse' else ratios).append(e1 * v1 / (ej * vj))
            if ref is not None:
                g0 = grid_metrics(*ref, j)
                lo, hi = vrf_ci(rj, j)
                for what, a, b in (('ESS/n', ej, g0['ess_n'][0]), ('VRF', vj, g0['vrf'][0])):
                    if abs(a / b - 1) > 0.1:
                        checks.append(f"(a) {t} {nm} {what}: {a:.3g} here, {b:.3g} in the tables' run ({100 * (a / b - 1):+.0f}%)" + (f"; bootstrap 95% interval of this run's VRF {lo:.3g}–{hi:.3g}" if what == 'VRF' else ''))
            for g, run in (('J', rj), ('γ=1', r1)):
                for key, est in (('fbar', 'plain'), ('mu_fixed', 'CV')):
                    z = pooled_z(run, j, mu, key, mcse[j] if mcse else 0.0)
                    if abs(z) >= 4:
                        checks.append(f'(b) {t} {transport} {nm} γ {g} {est}: pooled bias {z:+.1f} standard errors')
    r = np.array(ratios)
    readme = ["# Task 1: γ = 1 against J's γ (TGI-RWM, δ = 0, θ = (1, −1))\n", "Final flow (10⁴ steps). Chain experiment: 500 chains from z ~ N(0, I), 500 burn-in, 5000 recorded steps; J's γ is the median of the five adaptive runs of the tables (j_gamma.json), and both step sizes run in one batch with common random numbers. CV: K = 4, p = 2, N₀ = 10⁴, N = 10⁵; at γ = 1, β = 0 and η_κ = ĉ_κ. Benchmark: 50 independent estimates at sampling cost N = 10⁶ (training excluded); the J column is the tables' run, the γ = 1 one has no adaptation.\n", md(head, rows), f"\n**Conclusion.** On the trained flows ESS×VRF at γ = 1 is {r.min():.2f}–{r.max():.2f} times that at J's γ (median {np.median(r):.2f}); on the banana coarse member (F = 0.63) it is {min(coarse):.2f}–{max(coarse):.2f} times.\n" if coarse else f"\n**Conclusion.** ESS×VRF at γ = 1 / at J's γ: {r.min():.2f}–{r.max():.2f}.\n", '\n## Checks\n', "(a) J column within ±10% of the tables' run (vrf_flow_rwm.jsonl), (b) pooled bias below 4 standard errors (500 chains; for the real posteriors the standard error includes the reference's MCSE).\n", '\n'.join((f'- {c}' for c in checks)) if checks else '- all passed']
    (out / 'table.md').write_text(md(head, rows))
    (out / 'README.md').write_text('\n'.join(readme) + '\n')
    print('\n'.join(readme))

def task2():
    out = OUT / 'task2'
    out.mkdir(parents=True, exist_ok=True)
    head = ['posterior', 'd', 'training cost', 'ours at N=10⁶', 'baseline at N=10⁶', 'baseline at N=10⁶+training', 'ratio (baseline matched / ours)']
    rows, log, below = ([], [], [])
    rng = lambda v: f'{f3(np.nanmin(v))}–{f3(np.nanmax(v))}'
    for t in T2:
        target = make(t)
        tr = truth(target)
        cfg, R = (C.TARGETS[t], bench_rows(t))
        final = cfg['ckpts'][-1]
        ok = lambda m, N: [r for r in R if r['method'] == m and abs(r['budget'] - N) <= 1e-06 * N and (r.get('ckpt') in (None, final))]
        ct = [r['cost_train'] for r in R if r.get('ckpt') == final and 'cost_train' in r]
        if not ct or not ok('TGI-RWM+CV', 1000000.0):
            log.append(f'{t}: no TGI-RWM+CV run at N = 1e6 on the final flow; n/a')
            rows.append([t, str(target.dim), '--', '--', '--', 'n/a', 'n/a'])
            continue
        ct = ct[0]
        o = np.array([rmse(ok('TGI-RWM+CV', 1000000.0), tr['mean'][j], math.sqrt(tr['var'][j]), j) for j in range(len(tr['mean']))])
        if cfg.get('nuts_extrapolate'):
            b6 = nuts_extrapolated(t, tr, 1000000.0)
            bm = nuts_extrapolated(t, tr, 1000000.0 + ct)
            log.append(f"{t}: baseline = NUTS extrapolated from the reference run's MCSE")
        else:
            a6, a7 = baseline(t, tr, ok, [1000000.0, 10000000.0])
            b6 = a6
            bm = np.where(np.isfinite(a7), np.exp(np.log(a6) + (np.log(a7) - np.log(a6)) * math.log10(1 + ct / 1000000.0)), np.nan)
        if not np.isfinite(bm).any():
            log.append(f"{t}: baseline only up to N = {max((r['budget'] for r in R if r['method'] == 'NUTS')):.0e}; n/a")
            rows.append([t, str(target.dim), f3(ct), rng(o), rng(b6), 'n/a', 'n/a'])
            continue
        q = bm / o
        rows.append([t, str(target.dim), f3(ct), rng(o), rng(b6), rng(bm), rng(q)])
        if np.nanmin(q) < 1:
            below.append(f"{t} ({', '.join((f'{target.names[j]} {q[j]:.2f}' for j in np.where(q < 1)[0]))})")
    n_ok = sum((1 for r in rows if r[-1] not in ('n/a',) and float(r[-1].split('–')[0]) >= 1))
    n_all = sum((1 for r in rows if r[-1] != 'n/a'))
    mins = [float(r[-1].split('–')[0]) for r in rows if r[-1] != 'n/a']
    readme = ["# Task 2: training cost added to the baseline's budget\n", 'Ours: TGI-RWM+CV at sampling cost N = 10⁶ on the final flow (training excluded). Baseline: per test function the better of NUTS and NUTS+ZV2, its RMSE/sd at 10⁶ and 10⁷ interpolated linearly in log-log to 10⁶ + training cost. Ranges over the test functions; the ratio is per test function.\n', md(head, rows), f'\nRatio ≥ 1 for every test function on {n_ok} of {n_all} posteriors; smallest ratio {min(mins):.2f}.' + (f" Ratio below 1: {'; '.join(below)}." if below else ''), '\n## Log\n', '\n'.join((f'- {l}' for l in log)) or '- none']
    (out / 'table.md').write_text(md(head, rows))
    (out / 'README.md').write_text('\n'.join(readme) + '\n')
    print('\n'.join(readme))

def task5():
    import torch
    from train import load as load_flow
    from transports.diagnostics import ess_is
    out = OUT / 'task5'
    out.mkdir(parents=True, exist_ok=True)
    head = ['posterior', 'checkpoint', 'ESS_IS/N', 'F', 'γ (J)', 'ESS/n', 'VRF', 'ESS×VRF']
    rows, pts, notes = ([], [], [])
    for t in T5:
        target = make(t)
        j = C.TARGETS[t]['headline']
        series = []
        for k in T5_CKPTS:
            tag = 'flow' if k == C.TARGETS[t]['ckpts'][-1] else f'flow_ck{k}'
            got = load(t, 'vrf', tag, 'rwm')
            if got is None:
                notes.append(f'{t} checkpoint {k}: no run')
                continue
            meta, runs = got
            gm = grid_metrics(meta, runs, j)
            torch.manual_seed(5)
            e_is = ess_is(target, load_flow(t, k)[0], 100000)
            e, v = (gm['ess_n'][0], gm['vrf'][0])
            rows.append([t, str(k), f'{e_is:.4f}', f3(meta['F']), f"{runs[0]['gamma']:.3f}", f3(e), f3(v), f3(e * v)])
            pts.append(f"{t},{k},{e_is:.6g},{meta['F']:.6g},{runs[0]['gamma']:.6g},{e:.6g},{v:.6g},{e * v:.6g}")
            series.append((k, e_is, v))
        s = sorted(series, key=lambda x: x[1])
        bad = [(a, b) for a, b in zip(s, s[1:]) if b[2] < a[2]]
        notes.append(f'{t} ({target.names[j]}): VRF ' + ('rises monotonically with ESS_IS/N' if not bad else 'not monotone in ESS_IS/N: ' + '; '.join((f'checkpoint {b[0]} (ESS_IS {b[1]:.4f}, VRF {b[2]:.3g}) below checkpoint {a[0]} (ESS_IS {a[1]:.4f}, VRF {a[2]:.3g})' for a, b in bad))))
    readme = ['# Task 5: VRF against ESS_IS/N over the training checkpoints\n', "TGI-RWM, δ = 0, J's γ (median of five adaptive runs per checkpoint), CV refitted per checkpoint (K = 4, p = 2, N₀ = 10⁴, N = 10⁵); 500 chains from z ~ N(0, I), 500 burn-in, 5000 recorded steps. ESS_IS/N from 10⁵ draws of ρ; F on the chain states. Headline test function of each posterior.\n", md(head, rows), '\n', '\n'.join((f'- {n}' for n in notes))]
    (out / 'table.md').write_text(md(head, rows))
    (out / 'points.csv').write_text('posterior,checkpoint,ess_is,F,gamma,ess_n,vrf,eff\n' + '\n'.join(pts) + '\n')
    (out / 'README.md').write_text('\n'.join(readme) + '\n')
    print('\n'.join(readme))

def table2():
    out = OUT / 'table2'
    out.mkdir(parents=True, exist_ok=True)
    tex = {'pdb_garch': 'GARCH(1,1)', 'pdb_dugongs': 'Dugongs', 'pdb_one_comp': 'One-compartment PK', 'pdb_gp_pois': 'Poisson GP', 'eight_schools': 'Eight schools', 'pdb_surgical': 'Surgical', 'sparse_logistic_german': 'Sparse logistic', 'phi4_N32': '$\\phi^4$ field', 'lotka_volterra': 'Lotka--Volterra', 'sensor': 'Sensor network'}
    from figures import num
    sci = lambda v: num(v)
    lines, md_rows = ([], [])
    for t in T2:
        target = make(t)
        tr = truth(target)
        cfg, R = (C.TARGETS[t], bench_rows(t))
        final = cfg['ckpts'][-1]
        ok = lambda m, N: [r for r in R if r['method'] == m and abs(r['budget'] - N) <= 1e-06 * N and (r.get('ckpt') in (None, final))]
        if not ok('TGI-RWM+CV', 1000000.0):
            continue
        o = np.array([rmse(ok('TGI-RWM+CV', 1000000.0), tr['mean'][j], math.sqrt(tr['var'][j]), j) for j in range(len(tr['mean']))])
        b = baseline(t, tr, ok, [1000000.0])[0]
        bl = f'{sci(np.nanmin(b))}--{sci(np.nanmax(b))}' if np.isfinite(b).any() else '--'
        mark = '$^\\dagger$' if cfg.get('nuts_extrapolate') else ''
        lines.append(f'{tex.get(t, t)} & {target.dim} & {bl}{mark} & {sci(np.nanmin(o))}--{sci(np.nanmax(o))} \\\\')
        md_rows.append(f"| {t} | {target.dim} | {np.nanmin(b):.1e}–{np.nanmax(b):.1e}{(' (extrap.)' if mark else '')} | {np.nanmin(o):.1e}–{np.nanmax(o):.1e} | {np.nanmin(b / o):.2g}–{np.nanmax(b / o):.2g} |")
    body = '\n'.join(['\\begin{tabular}{@{}lccc@{}}', '\\toprule', 'Posterior & $d$ & Baseline & Ours \\\\', '\\midrule', *lines, '\\bottomrule', '\\end{tabular}'])
    (out / 'table2.tex').write_text(body + '\n')
    (out / 'table2.md').write_text('| posterior | d | baseline RMSE/sd at 1e6 | ours | baseline/ours per fn |\n|---|---|---|---|---|\n' + '\n'.join(md_rows) + '\n\n† NUTS extrapolated from the reference run (no NUTS benchmark).\n')
    print(body)
    print('\n'.join(md_rows))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', nargs='+', default=['1', '2', '5'], help='1, 2, 5 or table2')
    tasks = ap.parse_args().task
    setup(C.DTYPE)
    for k in tasks:
        {'1': task1, '2': task2, '5': task5, 'table2': table2}[str(k)]()
if __name__ == '__main__':
    main()
