import json
import math
import time
import numpy as np
import torch
from common import OUTPUT, make, raw_dir, setup, truth
from workflow import config as C
from methods.tgi import run_gti
from mcmc import make_generator
from train import load
from transports.ensemble import SharedFlow
from wallclock import single_fit
TARGETS = ['banana', 'funnel', 'gmix_d10_h6', 'tmix_d10_h6']
NAME = {'banana': 'Banana', 'funnel': 'Funnel', 'gmix_d10_h6': 'Mixture', 'tmix_d10_h6': '$t$ mixture'}
R, N, CHAIN = (100, 10 ** 6, 5000)

def chain_mse(m, mu, sd, Cn):
    m = m.detach().cpu().numpy().reshape(-1, m.shape[-1])
    m = m[np.isfinite(m).all(1)]
    g = m.mean(0)
    return ((g - mu) / sd) ** 2 + m.var(0, ddof=1) / Cn / sd ** 2

def main():
    setup(C.DTYPE)
    Cn = N // CHAIN
    rows = []
    for t in TARGETS:
        target = make(t)
        tr = truth(target)
        mu, sd = (np.array(tr['mean']), np.sqrt(np.array(tr['var'])))
        flow, _ = load(t)
        dfit = single_fit(flow, target, R)
        g = json.loads((raw_dir(t) / 'gamma_pilot_c5k.json').read_text())['rwm']['gamma']
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = run_gti(target, SharedFlow(flow, R), dfit, 'rwm', N, Cn, generator=make_generator(11), headline=C.TARGETS[t]['headline'], gamma0=g, adapt='fixed', burn_frac=0.2, burn_max=C.BURN, indep=dict(C.INDEP), lean=True)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        mse = chain_mse(r['chain_fixed'], mu, sd, Cn)
        acc = 1.0 / (mse * sec / R)
        ref = {x['method']: x for x in json.loads((raw_dir(t) / 'chainstats_c5k.json').read_text())['runs']}
        a0 = np.asarray(ref['TGI-RWM+CV']['acc_per_s'], float)
        r0 = np.asarray(ref['TGI-RWM+CV']['rmse'], float)
        out = {'target': t, 'gamma': g, 'delta': C.INDEP['delta'], 'df': C.INDEP['df'], 'seconds_batch': sec, 'rmse_delta': np.sqrt(mse).tolist(), 'acc_delta': acc.tolist(), 'rmse_0': r0.tolist(), 'acc_0': a0.tolist(), 'n_ind': float(torch.as_tensor(r['indep']['n_ind']).float().mean()) if 'indep' in r else None}
        (raw_dir(t) / 'delta_c5k.json').write_text(json.dumps(out, indent=1))
        h = C.TARGETS[t].get('main_fn', C.TARGETS[t]['headline'])
        ratio = acc / a0
        rows.append((t, h, a0[h], acc[h], r0[h], np.sqrt(mse)[h], float(np.median(ratio)), float(ratio.min()), float(ratio.max())))
        print(f'[delta {t}] gamma {g} | main fn: acc/s delta0 {a0[h]:.2e} delta {acc[h]:.2e} (ratio {acc[h] / a0[h]:.2f}); RMSE {r0[h]:.2e} -> {np.sqrt(mse)[h]:.2e} | ratio over functions: median {np.median(ratio):.2f} [{ratio.min():.2f}, {ratio.max():.2f}]', flush=True)

    def num(v):
        e = int(math.floor(math.log10(v)))
        return f'{v / 10 ** e:.1f}e{e}'
    L = ['\\begin{table}[t]', '\\centering', '\\small', '\\caption{TGI-RWM+CV without ($\\delta=0$) and with ($\\delta=0.01$, $t_{10}$ independence proposals, the setting of Theorem~\\ref{thm:clt}) the defensive independence step on the synthetic targets: precision per second of the main test function, its RMSE in posterior standard deviations, and the ratio of the precisions (with / without) over all test functions. Same flow, step size, seed and layout as Table~\\ref{tab:wallclock}.}', '\\label{tab:delta}', '\\begin{tabular}{@{}lcccccc@{}}', '\\toprule', ' & \\multicolumn{2}{c}{precision per second} & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{ratio over functions} \\\\', '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}', 'Target & $\\delta=0$ & $\\delta=0.01$ & $\\delta=0$ & $\\delta=0.01$ & median & range \\\\', '\\midrule']
    for t, h, a0, a1, r0, r1, med, lo, hi in rows:
        L.append(f'{NAME[t]} & {num(a0)} & {num(a1)} & {num(r0)} & {num(r1)} & {med:.2f} & {lo:.2f}--{hi:.2f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (OUTPUT / 'paper' / 'table_delta_c5k.tex').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
if __name__ == '__main__':
    main()
