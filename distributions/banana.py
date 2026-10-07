import math
import torch
from .base import TargetDistribution

class ProductBanana(TargetDistribution):

    def __init__(self, dim=8, b=0.02, p=25.0):
        if dim % 2:
            raise ValueError('ProductBanana needs an even dimension')
        super().__init__(dim)
        self.b, self.p = (b, p)

    def log_prob(self, x):
        x1, x2 = (x[..., 0::2], x[..., 1::2])
        r = x2 + self.b * x1 ** 2 - self.b * self.p
        return (-x1 ** 2 / (2 * self.p) - 0.5 * r ** 2).sum(-1)

    def sample(self, n, generator=None):
        z = torch.randn(n, self.dim, generator=generator)
        x1 = math.sqrt(self.p) * z[:, 0::2]
        x2 = z[:, 1::2] - self.b * x1 ** 2 + self.b * self.p
        return torch.stack([x1, x2], dim=-1).reshape(n, self.dim)

    def gaussian_init(self):
        m = torch.zeros(self.dim)
        m[1::2] = self.b * self.p
        s = torch.ones(self.dim)
        s[0::2] = math.sqrt(self.p)
        return (m, s)

    @property
    def log_normalizer(self):
        return 0.25 * self.dim * (math.log(2 * math.pi * self.p) + math.log(2 * math.pi))

    def tail_region(self, x):
        return x[..., 1] < self.b * self.p * (1 - 2.576 ** 2) - 2.576

    def shear_transport(self, c):
        from transports.analytic import BananaShear
        return BananaShear(self.dim, self.b, self.p, c, pairs='all')

    @property
    def test_functions(self):
        return [('x2', lambda x: x[..., 1], 0.0), ('mean x_even', lambda x: x[..., 1::2].mean(-1), 0.0)]
