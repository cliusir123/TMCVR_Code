import math
import torch

@torch.no_grad()
def log_importance_weights(target, transport, zeta):
    x, ld = transport(zeta)
    return target.log_prob(x) + ld + 0.5 * (zeta ** 2).sum(-1) + 0.5 * zeta.shape[-1] * math.log(2 * math.pi)

def ess_fraction(logw):
    logw = logw[torch.isfinite(logw)] if logw.numel() else logw
    if logw.numel() == 0:
        return 0.0
    n = logw.numel()
    return math.exp(2 * torch.logsumexp(logw, 0).item() - torch.logsumexp(2 * logw, 0).item()) / n

@torch.no_grad()
def ess_is(target, transport, n=20000, chunk=20000):
    logw = torch.cat([log_importance_weights(target, transport, torch.randn(min(chunk, n - s), target.dim)) for s in range(0, n, chunk)])
    return ess_fraction(logw)

def fisher(latent, z, chunk=4096):
    tot = 0.0
    for s in range(0, z.shape[0], chunk):
        g = latent.grad_log_w(z[s:s + chunk])
        tot += (g ** 2).sum().item()
    return tot / z.shape[0]

def delta_dispersion(latent, z, chunk=2048):
    s1 = s2 = 0.0
    for s in range(0, z.shape[0], chunk):
        zz = z[s:s + chunk].detach().requires_grad_(True)
        with torch.enable_grad():
            x, ld = latent.transport(zz)
            lp_fn = getattr(latent.target, 'log_prob_twice_diff', latent.target.log_prob)
            lp = lp_fn(x) + ld
            g, = torch.autograd.grad(lp.sum(), zz, create_graph=True)
            h = latent.hess_diag(zz, g)
        delta = (-h - 1.0).detach()
        s1 = s1 + delta.sum(0)
        s2 = s2 + (delta * delta).sum(0)
    n = z.shape[0]
    var = (s2 - s1 ** 2 / n) / (n - 1)
    return var.clamp_min(0).sqrt().mean().item()

@torch.no_grad()
def logw_gap(target, transport, n=20000):
    lw = log_importance_weights(target, transport, torch.randn(n, target.dim))
    lw = lw[torch.isfinite(lw)]
    return (lw.max() - torch.quantile(lw, 0.999)).item()
