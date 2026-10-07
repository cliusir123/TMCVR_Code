import math
import torch
from .base import Transport

class BananaShear(Transport):

    def __init__(self, dim, b, p, c, pairs='all'):
        super().__init__()
        self.dim, self.b, self.p, self.c = (dim, b, p, float(c))
        self.bp = b * p
        self.n_pairs = dim // 2 if pairs == 'all' else int(pairs)

    def forward(self, z):
        k = 2 * self.n_pairs
        z1, z2 = (z[..., 0:k:2], z[..., 1:k:2])
        x1 = math.sqrt(self.p) * z1
        x2 = z2 + self.bp - (self.bp - self.c) * z1 ** 2
        x = torch.cat([torch.stack([x1, x2], dim=-1).reshape(*z.shape[:-1], k), z[..., k:]], dim=-1)
        ld = 0.5 * self.n_pairs * math.log(self.p)
        return (x, torch.full(z.shape[:-1], ld, dtype=z.dtype, device=z.device))

    def sample_latent(self, n, generator=None):
        u = torch.randn(n, self.dim, generator=generator)
        k = 2 * self.n_pairs
        u[:, 1:k:2] = u[:, 1:k:2] - self.c * u[:, 0:k:2] ** 2
        return u

    @property
    def fisher_exact(self):
        return 3.5 * 2 * self.n_pairs * self.c ** 2

    def extra_repr(self):
        return f'c={self.c:g}, pairs={self.n_pairs}'

class FunnelScale(Transport):

    def __init__(self, dim, sigma_v, rho):
        super().__init__()
        self.dim, self.sigma_v, self.rho = (dim, sigma_v, float(rho))
        self.lam = 1.0 - self.rho / sigma_v

    def forward(self, z):
        v = self.sigma_v * z[..., :1]
        rest = torch.exp(self.lam * v / 2) * z[..., 1:]
        ld = math.log(self.sigma_v) + (self.dim - 1) * self.lam * v[..., 0] / 2
        return (torch.cat([v, rest], dim=-1), ld)

    def sample_latent(self, n, generator=None):
        u = torch.randn(n, self.dim, generator=generator)
        z1 = u[:, :1]
        return torch.cat([z1, torch.exp(self.rho * z1 / 2) * u[:, 1:]], dim=-1)

    def extra_repr(self):
        return f'rho={self.rho:g}'

class LogNormalScale(Transport):

    def __init__(self, dim, sigma, s, E=1):
        super().__init__()
        self.dim, self.sigma, self.s, self.E = (dim, float(sigma), float(s), E)
        self.a = self.sigma / self.s

    def forward(self, z):
        u = self.a * z
        ld = u.sum(-1) + self.dim * math.log(self.a)
        return (torch.exp(u), ld)

    def ess_is(self):
        s2 = self.s ** 2
        return (s2 * (2.0 - s2)) ** (self.dim / 2)

    def extra_repr(self):
        return f's={self.s:g}'
