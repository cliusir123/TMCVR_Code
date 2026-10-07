import math
import torch

def batch_size(dim, n_cols, n_mc, budget=32000000):
    return max(1, min(n_mc, budget // max(dim, n_cols, 1)))

def estimate_coefficients(dictionary, transport, F_fn, n_mc, chunk=None):
    single = callable(F_fn)
    F_fns = [F_fn] if single else list(F_fn)
    K, d, nF = (len(dictionary), dictionary.dim, len(F_fns))
    nb = chunk or batch_size(d, K, n_mc)
    dtype, device = (dictionary.degrees.dtype, dictionary.degrees.device)
    sH = torch.zeros(K, dtype=dtype, device=device)
    sH2 = torch.zeros(K, dtype=dtype, device=device)
    sFH = torch.zeros(K, nF, dtype=dtype, device=device)
    sFH2 = torch.zeros(K, nF, dtype=dtype, device=device)
    sF2H2 = torch.zeros(K, nF, dtype=dtype, device=device)
    sF = torch.zeros(nF, dtype=dtype, device=device)
    n = 0
    for start in range(0, n_mc, nb):
        m = min(nb, n_mc - start)
        z = torch.randn(m, d, dtype=dtype, device=device)
        x = transport.push(z, chunk=m)
        F = torch.stack([f(x) for f in F_fns], dim=-1)
        phi = dictionary.columns(dictionary.he_table(z))
        del z, x
        sH += phi.sum(0)
        sFH += phi.T @ F
        phi.mul_(phi)
        sH2 += phi.sum(0)
        sFH2 += phi.T @ F
        sF2H2 += phi.T @ (F * F)
        sF += F.sum(0)
        n += m
        del phi, F
    Fbar = (sF / n).unsqueeze(0)
    c = (sFH - Fbar * sH.unsqueeze(1)) / n
    second = (sF2H2 - 2 * Fbar * sFH2 + Fbar ** 2 * sH2.unsqueeze(1)) / n
    var = (second - c ** 2).clamp_min(0.0) * (n / max(n - 1, 1))
    se = (var / n).sqrt()
    return (c[:, 0], se[:, 0]) if single else (c, se)
