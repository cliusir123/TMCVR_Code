import math
from dataclasses import dataclass, field
import torch
from .coefficients import batch_size, estimate_coefficients
from .dictionary import count_indices

def full(K):
    return lambda D, c0, se0: (D.degrees <= K).unsqueeze(1).expand_as(c0)

def structural(K, p):
    return lambda D, c0, se0: ((D.degrees <= K) & (D.supports <= p)).unsqueeze(1).expand_as(c0)

def magnitude(K, p, lam):

    def rule(D, c0, se0):
        base = structural(K, p)(D, c0, se0)
        lv = 2 * math.log(max(count_indices(D.dim, K, p), 2)) if lam == '2log' else float(lam)
        return base & (c0 ** 2 > lv * se0 ** 2)
    return rule

@dataclass
class DictionaryFit:
    dictionary: object
    names: list
    c: torch.Tensor
    c0: torch.Tensor
    se0: torch.Tensor
    masks: dict
    var_rho: torch.Tensor
    resid: dict = field(default_factory=dict)

    def V0(self, arm):
        return self.var_rho / self.resid[arm].clamp_min(1e-300)

    def n_kept(self, arm):
        return self.masks[arm].sum(0)

def fit_dictionary(dictionary, transport, F_fns, arms, n_select=10000, n_fit=100000, n_resid=200000, names=None, chunk=None):
    c0, se0 = estimate_coefficients(dictionary, transport, F_fns, n_select, chunk=chunk)
    masks = {name: rule(dictionary, c0, se0).clone() for name, rule in arms.items()}
    c, _ = estimate_coefficients(dictionary, transport, F_fns, n_fit, chunk=chunk)
    var_rho, resid = residual_variances(dictionary, transport, F_fns, c, masks, n_resid, chunk)
    return DictionaryFit(dictionary, names or [f'F{j}' for j in range(len(F_fns))], c, c0, se0, masks, var_rho, resid)

def residual_variances(dictionary, transport, F_fns, c, masks, n, chunk=None):
    arms = list(masks)
    nF = len(F_fns)
    union = torch.zeros(len(dictionary), dtype=torch.bool, device=c.device)
    for a in arms:
        union |= masks[a].any(1)
    sub = dictionary.restrict(union)
    C = torch.cat([c[union] * masks[a][union] for a in arms], dim=1)
    d = dictionary.dim
    nb = chunk or batch_size(d, max(len(sub), 1), n)
    sF = sF2 = 0.0
    sR = torch.zeros(len(arms) * nF, dtype=c.dtype, device=c.device)
    sR2 = torch.zeros_like(sR)
    done = 0
    while done < n:
        m = min(nb, n - done)
        z = torch.randn(m, d, dtype=c.dtype, device=c.device)
        x = transport.push(z, chunk=m)
        F = torch.stack([f(x) for f in F_fns], dim=-1)
        G = sub.columns(sub.he_table(z)) @ C if len(sub) else torch.zeros(m, C.shape[1], dtype=c.dtype, device=c.device)
        res = F.repeat(1, len(arms)) - G
        sF = sF + F.sum(0)
        sF2 = sF2 + (F * F).sum(0)
        sR += res.sum(0)
        sR2 += (res * res).sum(0)
        done += m
    var_rho = (sF2 - sF ** 2 / n) / (n - 1)
    var_res = ((sR2 - sR ** 2 / n) / (n - 1)).reshape(len(arms), nF)
    return (var_rho, {a: var_res[i] for i, a in enumerate(arms)})
