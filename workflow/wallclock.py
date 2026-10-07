import argparse
import json
import math
import time
import numpy as np
import torch
from common import make, raw_dir, setup, truth
from workflow import config as C
from benchmark import BENCH_DICT
from methods.ellipt import run_ellipt
from methods.neutra import run_neutra_multi
from methods.tgi import fit_dictionary, run_gti
from mcmc import make_generator
from mcmc.adaptation import J_ROBUST
from train import load
from transports.ensemble import replicate

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def run_snis(target, flow, sub, N, R, generator, chunk=2000000, cv=True, blocks=None):
    d = target.dim
    nF = target.F(flow(torch.zeros(1, d))[0]).shape[-1]
    p = len(sub)
    shift = torch.full((R,), -float('inf'))
    S_w, S_wF = (torch.zeros(R), torch.zeros(R, nF))
    S_H, S_HH = (torch.zeros(R, p), torch.zeros(R, p, p))
    S_Hw, S_HwF = (torch.zeros(R, p), torch.zeros(R, p, nF))
    per = max(1, chunk // R)
    done = 0
    if blocks:
        K = int(blocks)
        bsize = N / K
        B_w, B_wF = (torch.zeros(R, K), torch.zeros(R, K, nF))
        B_H = torch.zeros(R, K, p)
    while done < N:
        m = int(min(per, N - done))
        z = torch.randn(R * m, d, generator=generator)
        x, ld = flow(z)
        lw = (target.log_prob(x) + ld + 0.5 * (z * z).sum(-1)).view(R, m)
        new = torch.maximum(shift, lw.max(1).values)
        f = torch.exp(shift - new)
        f = torch.where(torch.isfinite(f), f, torch.zeros_like(f))
        S_w, S_wF, S_Hw, S_HwF = (S_w * f, S_wF * f[:, None], S_Hw * f[:, None], S_HwF * f[:, None, None])
        if blocks:
            B_w, B_wF = (B_w * f[:, None], B_wF * f[:, None, None])
        shift = new
        w = torch.exp(lw - shift[:, None])
        F = target.F(x).view(R, m, nF)
        S_w += w.sum(1)
        S_wF += (w[:, :, None] * F).sum(1)
        if blocks:
            bid = torch.clamp(((done + torch.arange(m)) / bsize).long(), max=K - 1)
            B_w.index_add_(1, bid, w)
            B_wF.index_add_(1, bid, w[:, :, None] * F)
        if not cv:
            done += m
            continue
        H = sub.columns(sub.he_table(z)).view(R, m, p)
        S_H += H.sum(1)
        if blocks:
            B_H.index_add_(1, bid, H)
        S_HH += torch.bmm(H.transpose(1, 2), H)
        S_Hw += (H * w[:, :, None]).sum(1)
        S_HwF += torch.bmm(H.transpose(1, 2), w[:, :, None] * F)
        done += m
    n = float(N)
    diag = {'max_weight': 1.0 / S_w}
    nb = n / blocks if blocks else None

    def block_var(est, den, bw=None, bwF=None):
        num_b = B_wF / nb
        den_b = B_w / nb
        Hb = B_H / nb if bw is not None or bwF is not None else None
        if bwF is not None:
            num_b = num_b - torch.einsum('rkp,rpf->rkf', Hb, bwF)
        if bw is not None:
            den_b = den_b - torch.einsum('rkp,rp->rk', Hb, bw)
        psi = (num_b - est[:, None, :] * den_b[:, :, None]) / den[:, None, None]
        return psi.var(1, unbiased=True) / blocks
    if not cv:
        out = {'plain': S_wF / S_w[:, None], **diag}
        if blocks:
            out['var_plain'] = block_var(out['plain'], S_w / n)
        return out
    mw, mwF, mH = (S_w / n, S_wF / n, S_H / n)
    covHH = S_HH / n - mH[:, :, None] * mH[:, None, :]
    covHw = S_Hw / n - mH * mw[:, None]
    covHwF = S_HwF / n - mH[:, :, None] * mwF[:, None, :]
    reg = covHH + 1e-10 * covHH.diagonal(dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(p)
    b_w = torch.linalg.lstsq(reg, covHw[:, :, None]).solution[..., 0]
    b_wF = torch.linalg.lstsq(reg, covHwF).solution
    den = mw - (mH * b_w).sum(1)
    num = mwF - (mH[:, :, None] * b_wF).sum(1)
    out = {'plain': mwF / mw[:, None], 'cv': num / den[:, None], 'cv_num': num / mw[:, None], **diag}
    if blocks:
        out['var_plain'] = block_var(out['plain'], mw)
        out['var_cv'] = block_var(out['cv'], den, b_w, b_wF)
        out['var_cv_num'] = block_var(out['cv_num'], mw, None, b_wF)
    return out

def single_fit(flow, target, R):
    d1 = fit_dictionary(replicate(flow, 1), target, **BENCH_DICT)
    return {k: v.expand(R, *v.shape[1:]) if torch.is_tensor(v) and v.dim() >= 1 and (v.shape[0] == 1) else v for k, v in d1.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--sizes', nargs='+', type=float, default=[100000.0, 1000000.0, 10000000.0])
    ap.add_argument('--snis-chunk', type=float, default=2000000.0, help='flow draws per SNIS batch (all replicates); smaller for posteriors with many data points')
    ap.add_argument('--chain-len', type=float, default=2000, help='cost of one chain of TGI/TESS/NeuTra; short chains in parallel use the GPU best')
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    cfg = C.TARGETS[t]
    target = make(t)
    tr = truth(target)
    mu, sd = (np.array(tr['mean']), np.sqrt(np.array(tr['var'])))
    R = C.REPLICATES
    flow, cost_train = load(t)
    try:
        info = json.loads((raw_dir(t).parent.parent / 'flows' / t / 'train.json').read_text())
    except (OSError, ValueError):
        info = {}
    T = replicate(flow, R)
    out = {'target': t, 'R': R, 'train_seconds': info.get('seconds'), 'train_cost': cost_train, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'runs': []}
    sync()
    t0 = time.perf_counter()
    dfit = single_fit(flow, target, R)
    sync()
    out['cv_fit_seconds'] = time.perf_counter() - t0
    out['cv_terms'] = len(dfit['sub'])
    print(f"[wallclock {t}] dictionary: {len(dfit['sub'])} terms, {out['cv_fit_seconds']:.1f}s", flush=True)

    def record(method, N, sec, est, **extra):
        e = est.detach().cpu().numpy()
        rm = np.sqrt(np.nanmean((e - mu) ** 2, 0)) / sd
        bias = np.abs(np.nanmean(e, 0) - mu) / sd
        out['runs'].append({'method': method, 'N': N, 'seconds': sec, 'rmse': rm.tolist(), 'bias': bias.tolist(), **extra})
        print(f'[wallclock {t}] {method:12s} N={N:.0e} {sec:8.1f}s  RMSE/sd ' + ' '.join((f'{v:.1e}' for v in rm)), flush=True)
    for N in args.sizes:
        Cn = max(1, int(round(N / args.chain_len)))
        gen = make_generator(int(math.log10(N)) * 101)
        sync()
        t0 = time.perf_counter()
        r = run_snis(target, flow, dfit['sub'], int(N), R, gen, chunk=int(args.snis_chunk))
        sync()
        sec = time.perf_counter() - t0
        mw = r['max_weight'].cpu().numpy()
        extra = {'max_weight_median': float(np.median(mw)), 'max_weight_max': float(mw.max())}
        record('SNIS', N, sec, r['plain'], **extra)
        record('SNIS+CV', N, sec, r['cv'], **extra)
        print(f'[wallclock {t}] SNIS largest normalized weight: median {np.median(mw):.1e}, max over replicates {mw.max():.1e}', flush=True)
        sync()
        t0 = time.perf_counter()
        r = run_gti(target, T, dfit, 'rwm', N, Cn, generator=gen, headline=cfg['headline'], gamma0=0.5, j_lr=0.05, burn_frac=0.2, burn_max=C.BURN, j_robust=J_ROBUST, adapt='J', indep=dict(C.INDEP, delta=0.0), lean=True)
        sync()
        sec = time.perf_counter() - t0
        record('TGI-RWM', N, sec, r['plain'])
        record('TGI-RWM+CV', N, sec, r['estimate_fixed'])
        sync()
        t0 = time.perf_counter()
        r = run_ellipt(target, T, N, Cn, dfit, burn_frac=0.2, burn_max=C.BURN, generator=gen)
        sync()
        sec = time.perf_counter() - t0
        record('TESS', N, sec, r['plain'])
        record('TESS+CV', N, sec, r['estimate_fixed'])
        sync()
        t0 = time.perf_counter()
        r = run_neutra_multi(target, flow, N, R, chain_len=args.chain_len, generator=gen)
        sync()
        sec = time.perf_counter() - t0
        record('NeuTra', N, sec, r['plain'])
        record('NeuTra+ZVz', N, sec, r['zv_latent'])
        (raw_dir(t) / 'wallclock.json').write_text(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
