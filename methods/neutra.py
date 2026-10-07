import math
import time
import numpy as np
import torch
from mcmc import LatentTarget

def run_neutra(target, flow, budget, R, generator=None, warmup_frac=0.2, warmup_max=1000, accept_target=0.9, eps0=0.5, l_max=100, zv=True, zv_max=100000):
    from methods.zvcv import fit_estimate_chunked
    t0 = time.perf_counter()
    d = target.dim
    lat = LatentTarget(target, flow)
    dev, dt = (torch.get_default_device(), torch.get_default_dtype())
    z = torch.randn(R, d, generator=generator)
    s = lat.evaluate(z, 1)
    spent = torch.full((R,), 2.0)
    log_eps = torch.full((R,), math.log(eps0))
    log_eps_bar = torch.zeros(R)
    h_bar = torch.zeros(R)
    mu = math.log(10 * eps0)
    gam, t_0, kappa = (0.05, 10.0, 0.75)

    def n_leap(le):
        return torch.clamp(torch.ceil(d ** 0.25 / torch.exp(le)), 1, l_max)
    L = n_leap(log_eps)
    n_warm = int(min(warmup_max, warmup_frac * budget / (2.0 * L.mean().item())))
    running = torch.ones(R, dtype=torch.bool)
    nF = target.F(s.x[:1]).shape[-1]
    sum_F = torch.zeros(R, nF)
    n_kept = torch.zeros(R)
    acc_sum = torch.zeros(R)
    keep_x = []
    thin = None
    it = 0
    while running.any():
        warm = it < n_warm
        eps = torch.exp(log_eps)
        L = draw_L(log_eps)
        Lmax = int(L[running].max().item())
        p = torch.randn(R, d, generator=generator)
        H0 = -s.lp + 0.5 * (p * p).sum(-1)
        zq, gq, sq = (s.z.clone(), s.grad.clone(), s)
        pq = p + 0.5 * eps[:, None] * gq
        for l in range(Lmax):
            act = (l < L)[:, None]
            zq = torch.where(act, zq + eps[:, None] * pq, zq)
            sq = lat.evaluate(zq, 1)
            last = (l == L - 1)[:, None]
            pq = torch.where(act & ~last, pq + eps[:, None] * sq.grad, pq)
            pq = torch.where(act & last, pq + 0.5 * eps[:, None] * sq.grad, pq)
        H1 = -sq.lp + 0.5 * (pq * pq).sum(-1)
        log_a = torch.where(torch.isfinite(H1), H0 - H1, torch.full_like(H1, -float('inf')))
        a = torch.exp(torch.clamp(log_a, max=0.0))
        u = torch.rand(R, generator=generator)
        acc = (u < a) & running
        s = s.where(acc, sq)
        spent = spent + torch.where(running, 2.0 * L, torch.zeros_like(L))
        if warm:
            k = it + 1
            h_bar = (1 - 1 / (k + t_0)) * h_bar + (accept_target - a) / (k + t_0)
            log_eps = mu - math.sqrt(k) / gam * h_bar
            w = k ** (-kappa)
            log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
            if it == n_warm - 1:
                log_eps = log_eps_bar.clone() + math.log(eps_scale)
        else:
            Fx = target.F(s.x)
            sum_F += torch.where(running[:, None], Fx, torch.zeros_like(Fx))
            n_kept += running.double()
            acc_sum += torch.where(running, a, torch.zeros_like(a))
            if zv:
                if thin is None:
                    left = (budget - spent.mean().item()) / (2.0 * L.mean().item())
                    thin = max(1, math.ceil(left / zv_max))
                if (it - n_warm) % thin == 0:
                    keep_x.append(torch.where(running[:, None], s.x, torch.nan).cpu())
        it += 1
        running = running & (spent + 2.0 * n_leap(log_eps) <= budget)
    plain = sum_F / n_kept.clamp_min(1)[:, None]
    out = {'plain': plain, 'cost': spent, 'eps': torch.exp(log_eps), 'L': n_leap(log_eps), 'acceptance': acc_sum / n_kept.clamp_min(1), 'n_draws': n_kept, 'n_warm': n_warm}
    if zv and keep_x:
        X = torch.stack(keep_x, 1).to(dev)
        m, sd = target.laplace()
        est = torch.full_like(plain, float('nan'))
        cost_zv = torch.zeros(R)
        for r in range(R):
            x = X[r][torch.isfinite(X[r]).all(1)]
            if x.shape[0] < 2 * (d + d * (d + 1) // 2):
                continue
            g = target.grad_log_prob(x)
            cost_zv[r] = 2.0 * x.shape[0]
            try:
                est[r] = fit_estimate_chunked(x, g, target.F(x), m, sd, 2)[0]
            except RuntimeError:
                pass
        out['zv'] = est
        out['cost_zv'] = spent + cost_zv
    out['seconds'] = time.perf_counter() - t0
    return out

@torch.no_grad()
def run_neutra_multi(target, flow, budget, R, chain_len=20000.0, generator=None, warmup_frac=0.2, warmup_max=1000, accept_target=0.9, eps0=0.5, l_max=100, ridge=1e-10, zv=True, L_fixed=None, jitter=True, per_chain=False, eps_scale=1.0, warmup_min=0):
    from methods.zvcv import features
    t0 = time.perf_counter()
    d = target.dim
    lat = LatentTarget(target, flow)
    C = max(1, int(round(budget / chain_len)))
    b = budget / C
    B = R * C
    rep = torch.arange(B) // C
    z = torch.randn(B, d, generator=generator)
    with torch.enable_grad():
        s = lat.evaluate(z, 1)
    spent = torch.full((B,), 2.0)
    log_eps = torch.full((B,), math.log(eps0))
    log_eps_bar = torch.zeros(B)
    h_bar = torch.zeros(B)
    mu = math.log(10 * eps0)
    gam, t_0, kappa = (0.05, 10.0, 0.75)
    if L_fixed is not None:
        Lr = torch.as_tensor(L_fixed, dtype=torch.get_default_dtype()).reshape(-1)
        if Lr.numel() > 1 and bool((Lr != Lr[0]).any()):
            raise NotImplementedError('one L per batch (run the L grid as separate batches)')
        Lc = (Lr.expand(R) if Lr.numel() == 1 else Lr)[rep]
        lo_j = torch.clamp(torch.ceil(Lc / 2), min=1) if jitter else Lc
        hi_j = torch.clamp(torch.floor(3 * Lc / 2), min=1) if jitter else Lc

    def n_leap(le):
        if L_fixed is not None:
            return Lc.clone()
        return torch.clamp(torch.ceil(d ** 0.25 / torch.exp(le)), 1, l_max)

    def draw_L(le):
        if L_fixed is None:
            return n_leap(le)
        u = torch.rand(1, generator=generator).expand(B)
        return torch.floor(lo_j + u * (hi_j - lo_j + 1)).clamp(max=hi_j)
    L_hi = hi_j if L_fixed is not None else None
    L = n_leap(log_eps)
    n_warm = int(max(min(warmup_max, warmup_frac * b / (2.0 * L.mean().item())), warmup_min))
    running = torch.ones(B, dtype=torch.bool)
    nF = target.F(s.x[:1]).shape[-1]
    p = d + d * (d + 1) // 2
    zero, one = (torch.zeros(d), torch.ones(d))
    S_F = torch.zeros(R, nF)
    S_H = torch.zeros(R, p)
    S_HH = torch.zeros(R, p, p)
    S_HF = torch.zeros(R, p, nF)
    n_kept = torch.zeros(R)
    acc_sum = torch.zeros(B)
    if per_chain:
        c_F, c_H, c_n = (torch.zeros(B, nF), torch.zeros(B, p), torch.zeros(B))
    it = 0
    while running.any():
        warm = it < n_warm
        eps = torch.exp(log_eps)
        L = n_leap(log_eps)
        Lmax = int(L[running].max().item())
        pm = torch.randn(B, d, generator=generator)
        H0 = -s.lp + 0.5 * (pm * pm).sum(-1)
        zq, sq = (s.z.clone(), s)
        pq = pm + 0.5 * eps[:, None] * s.grad
        for l in range(Lmax):
            act = (l < L)[:, None]
            zq = torch.where(act, zq + eps[:, None] * pq, zq)
            with torch.enable_grad():
                sq = lat.evaluate(zq, 1)
            last = (l == L - 1)[:, None]
            pq = torch.where(act & ~last, pq + eps[:, None] * sq.grad, pq)
            pq = torch.where(act & last, pq + 0.5 * eps[:, None] * sq.grad, pq)
        H1 = -sq.lp + 0.5 * (pq * pq).sum(-1)
        log_a = torch.where(torch.isfinite(H1), H0 - H1, torch.full_like(H1, -float('inf')))
        a = torch.exp(torch.clamp(log_a, max=0.0))
        u = torch.rand(B, generator=generator)
        acc = (u < a) & running
        s = s.where(acc, sq)
        spent = spent + torch.where(running, 2.0 * L, torch.zeros_like(L))
        if warm:
            k = it + 1
            h_bar = (1 - 1 / (k + t_0)) * h_bar + (accept_target - a) / (k + t_0)
            log_eps = mu - math.sqrt(k) / gam * h_bar
            w = k ** (-kappa)
            log_eps_bar = w * log_eps + (1 - w) * log_eps_bar
            if it == n_warm - 1:
                log_eps = log_eps_bar.clone()
        else:
            on = running.to(s.z.dtype)[:, None]
            Fx = target.F(s.x) * on
            Fr = Fx.view(R, C, nF)
            S_F += Fr.sum(1)
            if zv:
                H = features(s.z, s.grad, zero, one, 2) * on
                Hr = H.view(R, C, p)
                S_H += Hr.sum(1)
                S_HH += torch.bmm(Hr.transpose(1, 2), Hr)
                S_HF += torch.bmm(Hr.transpose(1, 2), Fr)
                if per_chain:
                    c_H += H
            if per_chain:
                c_F += Fx
                c_n += on[:, 0]
            n_kept += running.view(R, C).sum(1).to(S_F.dtype)
            acc_sum += torch.where(running, a, torch.zeros_like(a))
        it += 1
        running = running & (spent + 2.0 * (L_hi if L_hi is not None else n_leap(log_eps)) <= b)
    n = n_kept.clamp_min(1)
    mF, mH = (S_F / n[:, None], S_H / n[:, None])
    covHH = S_HH / n[:, None, None] - mH[:, :, None] * mH[:, None, :]
    covHF = S_HF / n[:, None, None] - mH[:, :, None] * mF[:, None, :]
    scale = covHH.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-300)
    reg = covHH + ridge * scale[:, None, None] * torch.eye(p)
    beta = torch.linalg.lstsq(reg, covHF).solution
    zv = mF - (mH[:, :, None] * beta).sum(1)
    extra = {}
    if per_chain:
        nc = c_n.clamp_min(1)[:, None]
        extra['chain_plain'] = (c_F / nc).view(R, C, nF)
        cz = c_F - torch.bmm(c_H.view(R, C, p), beta).view(B, nF)
        extra['chain_zv'] = (cz / nc).view(R, C, nF)
        extra['chain_n'] = c_n.view(R, C)
    return {**extra, 'plain': mF, 'zv_latent': zv, 'cost': torch.zeros(R).index_add_(0, rep, spent), 'C': C, 'eps': torch.zeros(R).index_add_(0, rep, torch.exp(log_eps)) / C, 'L': torch.zeros(R).index_add_(0, rep, n_leap(log_eps)) / C, 'acceptance': torch.zeros(R).index_add_(0, rep, acc_sum) / n, 'n_kept': n_kept, 'n_warm': n_warm, 'seconds': time.perf_counter() - t0}
