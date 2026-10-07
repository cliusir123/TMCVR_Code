import argparse
import json
import time
from common import RESULTS, setup
import numpy as np
import torch
from methods.zvcv import _beta, _decompose, features as zv_features
from distributions import POSTERIORS, make_target
N_BATCHES = 400
ZV2_MAX = 9000
KEEP_DRAWS = 200000

class Accumulator:

    def __init__(self, target, C, draws, sym, m, s, degree):
        self.target, self.C, self.sym = (target, C, sym)
        self.b = max(1, draws // N_BATCHES)
        self.NB = -(-draws // self.b)
        self.m, self.s, self.degree = (m, s, degree)
        self.SF = self.SH = None
        self.HH = self.HF = self.sH = self.sF = self.sF2 = None
        self.n = 0
        self.t = 0
        self.kept = []

    def _F(self, x):
        F = self.target.F(x)
        if not self.sym:
            return (F, F * F)
        Fm = self.target.F(-x)
        return (0.5 * (F + Fm), 0.5 * (F * F + Fm * Fm))

    @torch.no_grad()
    def add(self, xr, g, keep_every, chunk=10000):
        C, n, d = xr.shape
        if keep_every:
            self.kept.append(self.target.reference_to_x(xr[:, ::keep_every]).astype(np.float32))
        x_all = self.target.reference_to_x(xr).reshape(-1, d)
        xr_all, g_all = (xr.reshape(-1, d), g.reshape(-1, d))
        for a in range(0, C * n, chunk):
            x = torch.as_tensor(x_all[a:a + chunk])
            F, F2 = self._F(x)
            H = zv_features(torch.as_tensor(xr_all[a:a + chunk]), torch.as_tensor(g_all[a:a + chunk]), self.m, self.s, self.degree) if self.degree else torch.zeros(F.shape[0], 0)
            if self.HH is None:
                p, k = (H.shape[1], F.shape[1])
                self.HH, self.HF = (torch.zeros(p, p), torch.zeros(p, k))
                self.sH, self.sF, self.sF2 = (torch.zeros(p), torch.zeros(k), torch.zeros(k))
                self.SF = torch.zeros(C * self.NB, k)
                self.SH = torch.zeros(C * self.NB, p)
            self.HH += H.T @ H
            self.HF += H.T @ F
            self.sH += H.sum(0)
            self.sF += F.sum(0)
            self.sF2 += F2.sum(0)
            r = torch.arange(a, a + F.shape[0])
            c, t = (r // n, r % n + self.t)
            idx = c * self.NB + torch.clamp(t // self.b, max=self.NB - 1)
            self.SF.index_add_(0, idx, F)
            self.SH.index_add_(0, idx, H)
        self.n += C * n
        self.t += n

    def result(self):
        n, k = (self.n, self.sF.numel())
        Fbar = self.sF / n
        var = (self.sF2 / n - Fbar ** 2) * n / (n - 1)
        full = self.t // self.b
        SF = self.SF.reshape(self.C, self.NB, -1)[:, :full]
        SH = self.SH.reshape(self.C, self.NB, -1)[:, :full]
        out = {'mean_plain': Fbar.tolist(), 'var': var.tolist(), 'mcse_plain': _mcse(SF / self.b).tolist(), 'rhat': _rhat(SF / self.b).tolist(), 'n_draws': n, 'batch': self.b, 'zv_degree': self.degree}
        if self.degree:
            dec = _decompose((self.HH, self.HF, self.sH, self.sF, n))
            beta = _beta(dec)
            mean = Fbar - self.sH / n @ beta
            adj = (SF - SH @ beta) / self.b
            out.update(mean=mean.tolist(), mcse=_mcse(adj).tolist(), n_cv=int(SH.shape[-1]))
        else:
            out.update(mean=out['mean_plain'], mcse=out['mcse_plain'])
        return out

def _mcse(bm):
    C, nb, _ = bm.shape
    v = bm.var(1, unbiased=True) / nb
    return (v.sum(0).sqrt() / C).cpu()

def _rhat(bm):
    C, nb, _ = bm.shape
    h = nb // 2
    S = torch.cat([bm[:, :h], bm[:, h:2 * h]], 0)
    W = S.var(1).mean(0)
    Bv = S.mean(1).var(0)
    r = torch.sqrt(((h - 1) / h * W + Bv) / W)
    return torch.where(W > 0, r, torch.ones_like(r)).cpu()

def run_nuts(target, args, acc):
    import numpyro
    numpyro.set_host_device_count(args.chains)
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    from distributions.backend import jax_backend
    jax.config.update('jax_enable_x64', True)
    B = jax_backend()
    logp = lambda x: target.reference_logp(x, B)
    init = target.reference_init()
    rng = np.random.default_rng(args.seed)
    x0 = init[None, :] + 0.05 * rng.standard_normal((args.chains, init.size))
    kernel = NUTS(potential_fn=lambda x: -logp(x), target_accept_prob=args.accept, max_tree_depth=10)
    block = min(args.block, args.draws)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=block, num_chains=args.chains, chain_method='parallel', progress_bar=False)
    mcmc.warmup(jax.random.PRNGKey(args.seed), init_params=jnp.asarray(x0))
    keep_every = max(1, args.chains * args.draws // KEEP_DRAWS)
    steps, div, done = (0.0, 0, 0)
    t0 = time.time()
    while done < args.draws:
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=('num_steps', 'diverging', 'z_grad'))
        xr = np.asarray(mcmc.get_samples(group_by_chain=True))
        ex = mcmc.get_extra_fields(group_by_chain=True)
        g = -np.asarray(ex['z_grad'])
        steps += float(np.asarray(ex['num_steps']).sum())
        div += int(np.asarray(ex['diverging']).sum())
        if acc is None:
            sd = xr.reshape(-1, xr.shape[-1]).std(0)
            mu = xr.reshape(-1, xr.shape[-1]).mean(0)
            p2 = target.dim + target.dim * (target.dim + 1) // 2
            degree = 0 if args.zv == 0 else args.zv if args.zv > 0 else 2 if p2 <= ZV2_MAX else 1
            acc = Accumulator(target, args.chains, args.draws, target.reference_method == 'nuts_sym', torch.as_tensor(mu), torch.as_tensor(sd), degree)
        acc.add(xr, g, keep_every)
        done += block
        print(f'    {done}/{args.draws} draws per chain, {time.time() - t0:.0f}s', flush=True)
    return (acc, {'mean_leapfrog': steps / (args.chains * done), 'divergences': div, 'accept_target': args.accept})

def run_pt_ref(target, args):
    from methods.pt import run_pt
    from mcmc import make_generator
    gen = make_generator(args.seed)
    x0 = torch.as_tensor(target.pt_init(args.pt_ensembles, args.seed))
    res = run_pt(target, x0, args.pt_iter, args.pt_burn, generator=gen)
    x = res.pop('x')
    E, n, d = x.shape
    g = target.grad_log_prob(x.reshape(-1, d)).reshape(E, n, d)
    mu, sd = (x.reshape(-1, d).mean(0), x.reshape(-1, d).std(0))
    acc = Accumulator(target, E, n, False, mu, sd, 2 if args.zv != 0 else 0)
    acc.add(x.cpu().numpy(), g.cpu().numpy(), max(1, E * n // KEEP_DRAWS))
    return (acc, res)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=list(POSTERIORS), required=True)
    ap.add_argument('--chains', type=int, default=24)
    ap.add_argument('--warmup', type=int, default=3000)
    ap.add_argument('--draws', type=int, default=200000, help='draws per chain')
    ap.add_argument('--block', type=int, default=20000)
    ap.add_argument('--accept', type=float, default=0.9)
    ap.add_argument('--zv', type=int, default=-1, help='control-variate degree (-1: auto, 0: none)')
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--pt-ensembles', type=int, default=256)
    ap.add_argument('--pt-iter', type=int, default=100000)
    ap.add_argument('--pt-burn', type=int, default=10000)
    ap.add_argument('--force', action='store_true', help='overwrite a reference with more draws')
    args = ap.parse_args()
    setup(torch.float64)
    target = make_target(args.target)
    if target.reference_path.exists() and (not args.force):
        old = json.loads(target.reference_path.read_text())
        new_n = args.pt_ensembles * args.pt_iter // 10 if target.reference_method == 'pt' else args.chains * args.draws
        if old.get('n_draws', 0) > new_n:
            raise SystemExit(f"{target.reference_path.name} has {old['n_draws']} draws > {new_n}; --force to overwrite")
    method = target.reference_method
    t0 = time.time()
    if method == 'pt':
        acc, info = run_pt_ref(target, args)
    else:
        acc, info = run_nuts(target, args, None)
    res = acc.result()
    for j, v in enumerate(target.analytic_truth()['mean']):
        if v is not None:
            res['mean'][j], res['mcse'][j] = (v, 0.0)
    out = {'target': args.target, 'method': method, 'chains': acc.C, 'seconds': time.time() - t0, 'names': target.names, **res, **info}
    path = target.reference_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    kept = np.concatenate(acc.kept, 1)
    np.savez_compressed(path.with_name(f'reference_{args.target}_draws.npz'), x=kept)
    rel = [m / np.sqrt(v) if v > 0 else 0.0 for m, v in zip(out['mcse'], out['var'])]
    relp = [m / np.sqrt(v) if v > 0 else 0.0 for m, v in zip(out['mcse_plain'], out['var'])]
    print(f"[reference] {args.target} ({method}): {acc.C} chains, {out['n_draws']} draws in {out['seconds']:.0f}s; max Rhat {max(out['rhat']):.4f}; MCSE/sd max {max(rel):.2e} (plain {max(relp):.2e}, ZV degree {out['zv_degree']})", flush=True)
    for n, m, s, sp in list(zip(target.names, out['mean'], out['mcse'], out['mcse_plain']))[:6]:
        print(f'    {n:>10}: {m:+.7f}  (MCSE {s:.1e}, plain {sp:.1e})')
if __name__ == '__main__':
    main()
