import math
import torch
from .base import TargetDistribution

class NealFunnel(TargetDistribution):

    def __init__(self, dim=10, sigma_v=2.0):
        super().__init__(dim)
        self.sigma_v = sigma_v

    def log_prob(self, x):
        v, rest = (x[..., 0], x[..., 1:])
        return -v ** 2 / (2 * self.sigma_v ** 2) - 0.5 * (self.dim - 1) * v - 0.5 * torch.exp(-v) * (rest ** 2).sum(-1)

    def sample(self, n, generator=None):
        v = self.sigma_v * torch.randn(n, 1, generator=generator)
        rest = torch.exp(v / 2) * torch.randn(n, self.dim - 1, generator=generator)
        return torch.cat([v, rest], dim=-1)

    def gaussian_init(self):
        m = torch.zeros(self.dim)
        s = torch.full((self.dim,), math.exp(self.sigma_v ** 2 / 4))
        s[0] = self.sigma_v
        return (m, s)

    @property
    def log_normalizer(self):
        return 0.5 * math.log(2 * math.pi * self.sigma_v ** 2) + 0.5 * (self.dim - 1) * math.log(2 * math.pi)

    def tail_region(self, x):
        return x[..., 0] < -2 * self.sigma_v

    def mean_log1p(self):
        if getattr(self, '_mean_log1p', None) is None:
            import numpy as np
            from numpy.polynomial.hermite_e import hermegauss
            from scipy import integrate
            import warnings
            warnings.filterwarnings('ignore', module='scipy.integrate')

            def inner(v):
                a = math.exp(v)
                f = lambda z: math.log1p(a * z * z) * math.exp(-0.5 * z * z)
                val, _ = integrate.quad(f, 0, 40, epsabs=1e-14, epsrel=1e-13, limit=500)
                return 2 * val / math.sqrt(2 * math.pi)
            t, w = hermegauss(80)
            w = w / w.sum()
            self._mean_log1p = float(sum((wi * inner(self.sigma_v * ti) for ti, wi in zip(t, w))))
        return self._mean_log1p

    def scale_transport(self, rho):
        from transports.analytic import FunnelScale
        return FunnelScale(self.dim, self.sigma_v, rho)

    @property
    def test_functions(self):
        return [('v', lambda x: x[..., 0], 0.0), ('mean log(1+x^2)', lambda x: torch.log1p(x[..., 1:] ** 2).mean(-1), self.mean_log1p())]
