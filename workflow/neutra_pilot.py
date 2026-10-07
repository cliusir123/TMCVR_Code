import argparse
import json
import time
import numpy as np
import torch
from common import RESULT_TAG, make, raw_dir, setup, truth
from workflow import config as C
from methods.neutra import run_neutra_multi
from mcmc import make_generator
from train import load

def chain_mse(m, mu, sd, Cn):
    m = m.detach().cpu().numpy().reshape(-1, m.shape[-1])
    m = m[np.isfinite(m).all(1)]
    if m.shape[0] < 2:
        return np.full(mu.shape, np.nan)
    g = m.mean(0)
    return ((g - mu) / sd) ** 2 + m.var(0, ddof=1) / Cn / sd ** 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True)
    ap.add_argument('--Ls', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument('--R', type=int, default=10)
    ap.add_argument('--N', type=float, default=1000000.0)
    ap.add_argument('--chain-len', type=float, default=10000.0)
    args = ap.parse_args()
    setup(C.DTYPE)
    t = args.target
    target = make(t)
    tr = truth(target)
    mu, sd = (np.array(tr['mean']), np.sqrt(np.array(tr['var'])))
    h = C.TARGETS[t]['headline']
    flow, _ = load(t)
    Cn = max(1, int(round(args.N / args.chain_len)))
    out = {'target': t, 'R': args.R, 'N': args.N, 'chain_len': args.chain_len, 'headline': h, 'runs': []}
    for L in args.Ls:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = run_neutra_multi(target, flow, args.N, args.R, chain_len=args.chain_len, generator=make_generator(100 + L), accept_target=0.8, warmup_min=300, L_fixed=L, per_chain=True)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        row = {'L': L, 'seconds': sec, 'eps': float(r['eps'].mean()), 'acceptance': float(r['acceptance'].mean()), 'warmup_iterations': int(r['n_warm']), 'kept_iterations_per_chain': float(r['chain_n'].float().mean()), 'cost_per_estimate': float(torch.as_tensor(r['cost']).float().mean())}
        if row['kept_iterations_per_chain'] == 0:
            row['infeasible'] = 'no sampling iterations left after warm-up (300 x 2L > chain cost)'
        for name, key in (('NeuTra', 'chain_plain'), ('NeuTra+ZVz', 'chain_zv')):
            mse = chain_mse(r[key], mu, sd, Cn) if 'infeasible' not in row else np.full(mu.shape, np.nan)
            row[name] = {'rmse': np.sqrt(mse).tolist(), 'acc_per_s': (1.0 / (mse * sec / args.R)).tolist()}
        out['runs'].append(row)
        print(f"[neutra L {t}] L={L:2d} {sec:6.1f}s eps {row['eps']:.3f} acc {row['acceptance']:.2f} ZVz RMSE {row['NeuTra+ZVz']['rmse'][h]:.2e} acc/s {row['NeuTra+ZVz']['acc_per_s'][h]:.2e}", flush=True)
    ok = [r for r in out['runs'] if np.isfinite(r['NeuTra+ZVz']['acc_per_s'][h])]
    best = max(ok, key=lambda r: r['NeuTra+ZVz']['acc_per_s'][h]) if ok else None
    out['L'] = best['L'] if best else None
    (raw_dir(t) / f'neutra_L{RESULT_TAG}.json').write_text(json.dumps(out, indent=1))
    print(f"[neutra L {t}] chosen L = {out['L']}", flush=True)
if __name__ == '__main__':
    main()
