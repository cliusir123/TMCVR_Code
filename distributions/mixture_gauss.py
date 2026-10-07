import math
import numpy as np
import torch
from transports.base import Transport
from .posterior import Posterior

def random_sigma(d, seed=0, lam_max=25.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d + 2, d))
    W = A.T @ A
    S = np.linalg.inv(W)
    S = 0.5 * (S + S.T)
    return S * (lam_max / np.linalg.eigvalsh(S).max())

def _logaddexp(a, b, B):
    mx = B.where(a > b, a, b)
    return mx + B.log1p(B.exp(-B.abs(a - b)))

def _lae(a, b):
    return torch.maximum(a, b) + torch.log1p(torch.exp(-(a - b).abs()))

class GaussMixture(Posterior):
    names = ['x1', 'x1^2', '1{x1>0}', 'mean x']

    def __init__(self, d=50, h=6.0, seed=0):
        super().__init__(d)
        self.h = h
        self.name = f'gmix_d{d}_h{int(h)}'
        self.Sigma = random_sigma(d, seed)
        self.L = np.linalg.cholesky(self.Sigma)
        self.P = np.linalg.inv(self.Sigma)
        self.m = np.zeros(d)
        self.m[0] = h / 2
        w = np.linalg.solve(self.L, self.m)
        self.a = float(np.linalg.norm(w))
        u = w / self.a
        e = np.zeros(d)
        e[0] = 1.0
        v = u - e
        R = np.eye(d) - 2 * np.outer(v, v) / (v @ v) if np.linalg.norm(v) > 1e-12 else np.eye(d)
        self.R = R
        self.M = self.L @ R
        self.logdet_M = float(np.log(np.abs(np.diag(self.L))).sum())

    def logp(self, x, B):
        P = self.c('P', self.P, B, x)
        m = self.c('m', self.m, B, x)
        q1 = ((x - m) @ P * (x - m)).sum(-1)
        q2 = ((x + m) @ P * (x + m)).sum(-1)
        return _logaddexp(-0.5 * q1, -0.5 * q2, B)

    def features(self, x, B):
        x1 = x[..., 0]
        return B.stack([x1, x1 ** 2, B.where(x1 > 0, 1.0 + 0 * x1, 0 * x1), x.mean(-1)], axis=-1)

    @property
    def test_functions(self):
        mu = self.analytic_truth()['mean']
        return [(n, lambda x, j=j: self.F(x)[..., j], mu[j]) for j, n in enumerate(self.names)]

    def sample(self, n, generator=None):
        z = torch.randn(n, self.dim, generator=generator)
        s = torch.where(torch.rand(n, generator=generator) < 0.5, 1.0, -1.0)
        L = torch.as_tensor(self.L, dtype=z.dtype, device=z.device)
        m = torch.as_tensor(self.m, dtype=z.dtype, device=z.device)
        return z @ L.T + s[:, None] * m

    def analytic_truth(self):
        mu, s2 = (self.h / 2, self.Sigma[0, 0])
        m2 = s2 + mu ** 2
        m4 = mu ** 4 + 6 * mu ** 2 * s2 + 3 * s2 ** 2
        one = np.ones(self.dim)
        var_mean = one @ self.Sigma @ one / self.dim ** 2 + (mu / self.dim) ** 2
        return {'mean': [0.0, m2, 0.5, 0.0], 'var': [m2, m4 - m2 ** 2, 0.25, var_mean]}

    def ground_truth(self):
        t = self.analytic_truth()
        return {'mean': t['mean'], 'mean_ref': t['mean'], 'var': t['var'], 'mcse': [0.0] * 4, 'p_tail': 0.0}

    def init_point(self):
        return self.m.copy()

    def exact_transport(self, E=1):
        return ExactMixtureTransport(self, E)

    def scale_transport(self, s):
        return ExactMixtureTransport(self, 1, scale=s)

class ExactMixtureTransport(Transport):

    def __init__(self, target, E=1, scale=1.0):
        super().__init__()
        self._target = [target]
        self.E, self.dim, self.a = (E, target.dim, float(target.a))
        self.s = float(scale)
        dt = torch.get_default_dtype()
        self.register_buffer('M', torch.as_tensor(target.M, dtype=dt))
        self.register_buffer('Minv', torch.as_tensor(np.linalg.inv(target.M), dtype=dt))
        self.logdet_M = target.logdet_M

    def _log_G(self, v):
        a = self.a
        l1 = torch.special.log_ndtr(v - a)
        l2 = torch.special.log_ndtr(v + a)
        return _lae(l1, l2) - math.log(2.0)

    def _log_g(self, v):
        a = self.a
        c = -0.5 * math.log(2 * math.pi) - math.log(2.0)
        return _lae(-0.5 * (v - a) ** 2, -0.5 * (v + a) ** 2) + c

    def _inv(self, z1):
        with torch.no_grad():
            t = torch.special.log_ndtr(z1)
            lo = z1 - self.a - 10.0
            hi = torch.minimum(z1 + self.a + 10.0, torch.zeros_like(z1) + 0.0)
            hi = torch.maximum(hi, lo + 1e-09)
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                below = self._log_G(mid) < t
                lo = torch.where(below, mid, lo)
                hi = torch.where(below, hi, mid)
            v0 = 0.5 * (lo + hi)
        lG, lg = (self._log_G(v0), self._log_g(v0))
        return v0 + (torch.special.log_ndtr(z1) - lG) * torch.exp(lG - lg)

    def forward(self, z):
        if self.s != 1.0:
            x, ld = self._forward(self.s * z)
            return (x, ld + self.dim * math.log(self.s))
        return self._forward(z)

    def inverse(self, x):
        z, ld = self._inverse(x)
        if self.s != 1.0:
            return (z / self.s, ld - self.dim * math.log(self.s))
        return (z, ld)

    def _forward(self, z):
        shape = z.shape
        zf = z.reshape(-1, self.dim)
        z1 = zf[:, 0]
        neg = z1 <= 0
        v1 = torch.where(neg, self._inv(torch.where(neg, z1, -z1)), -self._inv(torch.where(neg, z1, -z1)))
        v = torch.cat([v1[:, None], zf[:, 1:]], 1)
        x = v @ self.M.T
        ld = self.logdet_M - 0.5 * z1 ** 2 - 0.5 * math.log(2 * math.pi) - self._log_g(v1)
        return (x.reshape(shape), ld.reshape(shape[:-1]))

    def _inverse(self, x):
        shape = x.shape
        v = x.reshape(-1, self.dim) @ self.Minv.T
        v1 = v[:, 0]
        z1 = torch.special.ndtri(torch.exp(self._log_G(v1)).clamp(1e-300, 1 - 1e-16))
        z = torch.cat([z1[:, None], v[:, 1:]], 1)
        ld = -(self.logdet_M - 0.5 * z1 ** 2 - 0.5 * math.log(2 * math.pi) - self._log_g(v1))
        return (z.reshape(shape), ld.reshape(shape[:-1]))

    def log_prob(self, x):
        z, ld = self.inverse(x)
        return -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi) + ld

    @torch.no_grad()
    def sample_latent(self, n, generator=None):
        x = self._target[0].sample(n, generator=generator)
        return self.inverse(x.to(self.M.dtype))[0]

class TMixture(GaussMixture):

    def __init__(self, d=10, h=6.0, nu=5.0, seed=0):
        super().__init__(d, h, seed)
        self.nu = float(nu)
        self.name = f'tmix_d{d}_h{int(h)}'
        self.S = self.Sigma * (self.nu - 2.0) / self.nu
        self.LS = np.linalg.cholesky(self.S)
        self.PS = np.linalg.inv(self.S)

    def logp(self, x, B):
        P = self.c('PS', self.PS, B, x)
        m = self.c('m', self.m, B, x)
        k = -0.5 * (self.nu + self.dim)
        q1 = ((x - m) @ P * (x - m)).sum(-1)
        q2 = ((x + m) @ P * (x + m)).sum(-1)
        return _logaddexp(k * B.log1p(q1 / self.nu), k * B.log1p(q2 / self.nu), B)

    def sample(self, n, generator=None):
        z = torch.randn(n, self.dim, generator=generator)
        g = torch.distributions.Gamma(self.nu / 2, self.nu / 2)
        k = int(self.nu)
        if abs(k - self.nu) < 1e-12:
            c = (torch.randn(n, k, generator=generator) ** 2).sum(-1) / self.nu
        else:
            c = g.sample((n,)).to(z)
        s = torch.where(torch.rand(n, generator=generator) < 0.5, 1.0, -1.0)
        L = torch.as_tensor(self.LS, dtype=z.dtype, device=z.device)
        m = torch.as_tensor(self.m, dtype=z.dtype, device=z.device)
        return z @ L.T / torch.sqrt(c)[:, None] + s[:, None] * m

    def analytic_truth(self):
        mu, s2, nu = (self.h / 2, self.Sigma[0, 0], self.nu)
        m2 = s2 + mu ** 2
        m4 = mu ** 4 + 6 * mu ** 2 * s2 + 3 * s2 ** 2 * (nu - 2) / (nu - 4)
        one = np.ones(self.dim)
        var_mean = one @ self.Sigma @ one / self.dim ** 2 + (mu / self.dim) ** 2
        return {'mean': [0.0, m2, 0.5, 0.0], 'var': [m2, m4 - m2 ** 2, 0.25, var_mean]}

    def exact_transport(self, E=1):
        raise NotImplementedError('no exact transport for the t mixture')
