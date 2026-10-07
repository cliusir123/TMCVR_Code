import math
import torch

class CVRecorder:

    def __init__(self, fit, F_fns, arms, gamma, n_steps=None):
        D, nF = (fit.dictionary, len(F_fns))
        self.F_fns = F_fns
        self.entries = [(arm, j) for arm in arms for j in range(nF)]
        union = torch.zeros(len(D), dtype=torch.bool, device=fit.c.device)
        for arm in arms:
            union |= fit.masks[arm].any(1)
        self.sub = D.restrict(union)
        self.C = torch.stack([fit.c[union, j] * fit.masks[arm][union, j] for arm, j in self.entries], dim=1)
        self.fidx = torch.tensor([j for _, j in self.entries], device=gamma.device)
        deg = self.sub.degrees.long()
        self.deg_idx = [(k, (deg == k).nonzero().flatten()) for k in range(1, D.max_degree + 1) if bool((deg == k).any())]
        beta = (1.0 - gamma).unsqueeze(1)
        ks = torch.arange(1, D.max_degree + 1, dtype=gamma.dtype, device=gamma.device)
        self.W = 1.0 / (1.0 - beta ** ks)
        B, E = (gamma.shape[0], len(self.entries))
        zeros = lambda: torch.zeros(B, E, dtype=gamma.dtype, device=gamma.device)
        self.S = {k: zeros() for k in ('F', '1', '2', 'FF', '11', '22', 'F1', 'F2', '12')}
        self.n = 0
        self.cnt = self.bcnt = None
        self.a = self.b = None
        if n_steps:
            self.a = max(int(math.isqrt(n_steps)), 1)
            self.b = max(n_steps // self.a, 1)
            self.batch = {k: torch.zeros(self.a, B, E, dtype=gamma.dtype, device=gamma.device) for k in ('F', '1', '2')}

    def _G(self, tab):
        B = tab[0].shape[0]
        out = torch.zeros(B, self.C.shape[1], dtype=tab[0].dtype, device=tab[0].device)
        if not len(self.sub):
            return out
        phi = self.sub.columns(tab)
        for k, idx in self.deg_idx:
            out += self.W[:, k - 1:k] * (phi[:, idx] @ self.C[idx])
        return out

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        D = self.sub
        Gz = self._G(D.he_table(s.z))
        Gp = self._G(D.he_table(sp.z))
        EG = self._G(D.cond_table(m, tau))
        F = torch.stack([f(s.x) for f in self.F_fns], dim=-1)[:, self.fidx]
        H1 = alpha.unsqueeze(1) * (Gp - Gz)
        H2 = Gp - EG
        if mask is not None:
            w = mask.unsqueeze(1)
            F, H1, H2 = (F * w, H1 * w, H2 * w)
            if self.cnt is None:
                self.cnt = torch.zeros_like(mask)
                if self.a is not None:
                    self.bcnt = torch.zeros(self.a, mask.shape[0], dtype=mask.dtype, device=mask.device)
            self.cnt += mask
        S = self.S
        S['F'] += F
        S['1'] += H1
        S['2'] += H2
        S['FF'] += F * F
        S['11'] += H1 * H1
        S['22'] += H2 * H2
        S['F1'] += F * H1
        S['F2'] += F * H2
        S['12'] += H1 * H2
        if self.a is not None and self.n < self.a * self.b:
            i = self.n // self.b
            self.batch['F'][i] += F
            self.batch['1'][i] += H1
            self.batch['2'][i] += H2
            if self.bcnt is not None:
                self.bcnt[i] += mask
        self.n += 1

    def results(self):
        S = self.S
        n = self.n if self.cnt is None else self.cnt.clamp_min(1).unsqueeze(1)
        mF, m1, m2 = (S['F'] / n, S['1'] / n, S['2'] / n)
        cov = lambda xy, x, y: S[xy] / n - x * y
        k11, k22, k12 = (cov('11', m1, m1), cov('22', m2, m2), cov('12', m1, m2))
        q1, q2 = (cov('F1', mF, m1), cov('F2', mF, m2))
        ridge = 1e-12 * (k11 + k22) + 1e-300
        k11, k22 = (k11 + ridge, k22 + ridge)
        det = k11 * k22 - k12 ** 2
        ok = det > 0
        safe = torch.where(ok, det, torch.ones_like(det))
        th1 = torch.where(ok, -(k22 * q1 - k12 * q2) / safe, torch.zeros_like(det))
        th2 = torch.where(ok, -(k11 * q2 - k12 * q1) / safe, torch.zeros_like(det))
        out = {'fbar': mF, 'mu_fit': mF + th1 * m1 + th2 * m2, 'mu_fixed': mF + m1 - m2, 'theta1': th1, 'theta2': th2, 'h1bar': m1, 'h2bar': m2}
        if self.a is not None:
            a = min(self.a, self.n // self.b)
            bF, b1_, b2_ = (self.batch[k][:a] for k in ('F', '1', '2'))
            if self.bcnt is not None:
                sc = self.b / self.bcnt[:a].clamp_min(1).unsqueeze(-1)
                bF, b1_, b2_ = (bF * sc, b1_ * sc, b2_ * sc)

            def bm(x):
                if a < 2:
                    return torch.full(x.shape[1:], float('nan'), dtype=x.dtype, device=x.device)
                return self.b * (x / self.b).var(0, unbiased=True)
            if a >= 3:
                xF, x1, x2 = (bF / self.b, b1_ / self.b, b2_ / self.b)
                c = lambda u, v: ((u - u.mean(0)) * (v - v.mean(0))).mean(0)
                c11, c22, c12 = (c(x1, x1), c(x2, x2), c(x1, x2))
                d1, d2 = (c(xF, x1), c(xF, x2))
                rdg = 1e-12 * (c11 + c22) + 1e-300
                c11, c22 = (c11 + rdg, c22 + rdg)
                dt = c11 * c22 - c12 ** 2
                okb = dt > 0
                sb = torch.where(okb, dt, torch.ones_like(dt))
                tb1 = torch.where(okb, -(c22 * d1 - c12 * d2) / sb, torch.zeros_like(dt))
                tb2 = torch.where(okb, -(c11 * d2 - c12 * d1) / sb, torch.zeros_like(dt))
                out.update(theta_bm1=tb1, theta_bm2=tb2, mu_bm=mF + tb1 * m1 + tb2 * m2, bm_bm=bm(bF + tb1 * b1_ + tb2 * b2_))
            out.update(bm_plain=bm(bF), bm_fit=bm(bF + th1 * b1_ + th2 * b2_), bm_fixed=bm(bF + b1_ - b2_))
        return out
