import math
import torch

def _parts(target):
    if hasattr(target, 'log_lik') and hasattr(target, 'log_prior'):
        from distributions.backend import TORCH
        return (lambda x: target.log_prior(x, TORCH), lambda x: target.log_lik(x, TORCH))
    return (lambda x: torch.zeros(x.shape[:-1], device=x.device), target.log_prob)

@torch.no_grad()
def run_pt(target, x0, n_iter, burn, K=24, beta_min=0.001, n_moves=5, thin=10, acc_target=0.3, scale0=0.1, generator=None):
    gen = generator
    E, d = x0.shape
    lprior, llik = _parts(target)
    betas = torch.logspace(0, math.log10(beta_min), K)
    x = x0[:, None, :].expand(E, K, d).clone()
    lp0, ll = (lprior(x), llik(x))
    log_sc = torch.full((K,), math.log(scale0))
    acc_mv = torch.zeros(K)
    acc_sw = torch.zeros(K - 1)
    n_sw = torch.zeros(K - 1)
    keep = []
    for it in range(burn + n_iter):
        for _ in range(n_moves):
            y = x + log_sc.exp()[None, :, None] * torch.randn(x.shape, generator=gen)
            lpy, lly = (lprior(y), llik(y))
            logr = lpy - lp0 + betas * (lly - ll)
            ok = torch.log(torch.rand(logr.shape, generator=gen)) < logr
            ok = ok & torch.isfinite(logr)
            x = torch.where(ok[..., None], y, x)
            lp0 = torch.where(ok, lpy, lp0)
            ll = torch.where(ok, lly, ll)
            a = ok.double().mean(0)
            if it < burn:
                log_sc += (a - acc_target) / math.sqrt(it + 1.0)
            else:
                acc_mv += a
        k0 = it % 2
        lo = torch.arange(k0, K - 1, 2)
        hi = lo + 1
        logr = (betas[lo] - betas[hi]) * (ll[:, hi] - ll[:, lo])
        ok = torch.log(torch.rand(logr.shape, generator=gen)) < logr
        for arr in (x, lp0, ll):
            a_lo, a_hi = (arr[:, lo].clone(), arr[:, hi].clone())
            m = ok if arr.dim() == 2 else ok[..., None]
            arr[:, lo] = torch.where(m, a_hi, a_lo)
            arr[:, hi] = torch.where(m, a_lo, a_hi)
        if it >= burn:
            acc_sw[lo] += ok.double().mean(0)
            n_sw[lo] += 1
            if (it - burn) % thin == 0:
                keep.append(x[:, 0].clone())
    return {'x': torch.stack(keep, 1), 'betas': betas.tolist(), 'swap_acceptance': (acc_sw / n_sw.clamp(min=1)).tolist(), 'move_acceptance': (acc_mv / (n_iter * n_moves)).tolist(), 'scales': log_sc.exp().tolist(), 'cost': float((burn + n_iter) * n_moves * K)}
