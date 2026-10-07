import math
import torch
from mcmc import LatentTarget

def ellipt_lambda(k):
    return 0.0 if k % 2 else math.comb(k, k // 2) / 2.0 ** k

class _G:

    def __init__(self, dictfit, E, C):
        self.sub, self.coef, self.E, self.C = (dictfit['sub'], dictfit['coef'], E, C)
        deg = self.sub.degrees.long()
        self.parts = []
        for k in range(1, self.sub.max_degree + 1):
            idx = (deg == k).nonzero().flatten()
            if idx.numel():
                lam = ellipt_lambda(k)
                self.parts.append((idx, 1.0 / (1.0 - lam), lam / (1.0 - lam)))

    def __call__(self, z, rows):
        nF = self.coef.shape[-1]
        G = torch.zeros(z.shape[0], nF)
        QG = torch.zeros(z.shape[0], nF)
        if not len(self.sub) or z.shape[0] == 0:
            return (G, QG)
        phi = self.sub.columns(self.sub.he_table(z))
        coef = self.coef[rows // self.C]
        for idx, wg, wq in self.parts:
            part = torch.einsum('nk,nkf->nf', phi[:, idx], coef[:, idx])
            G += wg * part
            QG += wq * part
        return (G, QG)

@torch.no_grad()
def run_ellipt(target, transport, budget, C, dictfit=None, burn_frac=0.2, burn_max=1000, generator=None, cv_reg=False, ridge=1e-10):
    E, d = (transport.E, transport.dim)
    B = E * C
    n_att = int(budget // C)
    n_burn = int(min(burn_max, math.floor(burn_frac * n_att)))
    if n_att - n_burn < 4:
        return None
    lat = LatentTarget(target, transport)
    gen = generator
    rows_all = torch.arange(B)

    def log_w(zz):
        st = lat.evaluate(zz, order=0)
        return (st.lp + 0.5 * (zz * zz).sum(-1), st.x)

    def uniform(*shape):
        return torch.rand(*shape, generator=gen)
    z = torch.randn(B, d, generator=gen)
    lw, x = log_w(z)
    Gf = _G(dictfit, E, C) if dictfit is not None else None
    nF = len(target.test_functions)
    sF, sCV = (torch.zeros(B, nF), torch.zeros(B, nF))
    cnt = torch.zeros(B)
    pF, pC = (torch.zeros(B, nF), torch.zeros(B, nF))
    reg = cv_reg and dictfit is not None and (len(dictfit['sub']) > 0)
    if reg:
        sub = dictfit['sub']
        K = len(sub)
        lam = torch.tensor([ellipt_lambda(int(k)) for k in sub.degrees.tolist()])
        pP0, pX2 = (torch.zeros(B, K), torch.zeros(B, K))
        R_XX, R_XF = (torch.zeros(E, 2 * K, 2 * K), torch.zeros(E, 2 * K, nF))
        R_X, c_X = (torch.zeros(E, 2 * K), torch.zeros(B, 2 * K))
    rec = torch.zeros(B, dtype=torch.bool)
    v = torch.zeros(B, d)
    logy, t, tmin, tmax = (torch.zeros(B) for _ in range(4))
    new = torch.ones(B, dtype=torch.bool)
    for step in range(n_att):
        if bool(new.any()):
            k = int(new.sum())
            v[new] = torch.randn(k, d, generator=gen)
            logy[new] = lw[new] + torch.log(uniform(k))
            th = 2 * math.pi * uniform(k)
            t[new], tmin[new], tmax[new] = (th, th - 2 * math.pi, th)
            rec[new] = step >= n_burn
            Fz = target.F(x[new])
            pF[new] = Fz
            if Gf is not None:
                rows, zc = (rows_all[new], z[new])
                yc = zc * torch.cos(th)[:, None] + v[new] * torch.sin(th)[:, None]
                G0, QG0 = Gf(zc, rows)
                Gy, _ = Gf(yc, rows)
                pC[new] = Fz - G0 - Gy + QG0
                if reg:
                    ph0 = sub.columns(sub.he_table(zc))
                    pP0[new] = ph0
                    pX2[new] = sub.columns(sub.he_table(yc)) - lam * ph0
        zc = z * torch.cos(t)[:, None] + v * torch.sin(t)[:, None]
        lwc, xc = log_w(zc)
        ok = torch.isfinite(lwc) & (lwc > logy)
        z = torch.where(ok[:, None], zc, z)
        x = torch.where(ok[:, None], xc, x)
        lw = torch.where(ok, lwc, lw)
        done = ok & rec
        if bool(done.any()):
            sF[done] += pF[done]
            if Gf is not None:
                G1, _ = Gf(z[done], rows_all[done])
                sCV[done] += pC[done] + G1
                if reg:
                    Xd = torch.zeros(B, 2 * K)
                    Xd[done] = torch.cat([sub.columns(sub.he_table(z[done])) - pP0[done], pX2[done]], dim=-1)
                    Fd = torch.zeros(B, nF)
                    Fd[done] = pF[done]
                    Xr = Xd.view(E, C, 2 * K)
                    R_XX += torch.bmm(Xr.transpose(1, 2), Xr)
                    R_XF += torch.bmm(Xr.transpose(1, 2), Fd.view(E, C, nF))
                    R_X += Xr.sum(1)
                    c_X += Xd
            cnt[done] += 1
        neg = t < 0
        tmin = torch.where(~ok & neg, t, tmin)
        tmax = torch.where(~ok & ~neg, t, tmax)
        t = torch.where(ok, t, tmin + (tmax - tmin) * uniform(B))
        new = ok
    cE = cnt.reshape(E, C).sum(1).clamp_min(1)[:, None]
    out = {'plain': sF.reshape(E, C, nF).sum(1) / cE, 'iterations': cnt.reshape(E, C).mean(1), 'evals_per_iter': n_att / cnt.clamp_min(1).mean().item(), 'n_burn': n_burn, 'cost': float(C * n_att), 'C': C}
    nc = cnt.clamp_min(1)[:, None]
    out['chain_iters'] = cnt.reshape(E, C)
    out['chain_plain'] = (sF / nc).reshape(E, C, nF)
    if Gf is not None:
        out['estimate_fixed'] = sCV.reshape(E, C, nF).sum(1) / cE
        out['chain_fixed'] = (sCV / nc).reshape(E, C, nF)
    if reg:
        n = cnt.reshape(E, C).sum(1).clamp_min(1)
        mX, mF = (R_X / n[:, None], sF.reshape(E, C, nF).sum(1) / n[:, None])
        cXX = R_XX / n[:, None, None] - mX[:, :, None] * mX[:, None, :]
        cXF = R_XF / n[:, None, None] - mX[:, :, None] * mF[:, None, :]
        sc = cXX.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-300)
        beta = torch.linalg.lstsq(cXX + ridge * sc[:, None, None] * torch.eye(2 * K), cXF).solution
        out['estimate_reg'] = mF - (mX[:, :, None] * beta).sum(1)
        cz = sF.reshape(E, C, nF) - torch.bmm(c_X.view(E, C, 2 * K), beta)
        out['chain_reg'] = cz / nc.reshape(E, C, 1)
    return out
