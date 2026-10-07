import math
import torch
from .base import TargetDistribution

class HaarioBanana(TargetDistribution):

    def __init__(self, dim=8, p=10.0, b=0.1):
        if dim < 2:
            raise ValueError('HaarioBanana needs dim >= 2')
        super().__init__(dim)
        self.p, self.b = (p, b)

    def log_prob(self, x):
        x1, x2 = (x[..., 0], x[..., 1])
        r = x2 + self.b * x1 ** 2 - self.b * self.p
        return -x1 ** 2 / (2 * self.p) - 0.5 * r ** 2 - 0.5 * (x[..., 2:] ** 2).sum(-1)

    def sample(self, n, generator=None):
        x = torch.randn(n, self.dim, generator=generator)
        x[:, 0] = math.sqrt(self.p) * x[:, 0]
        x[:, 1] = x[:, 1] - self.b * x[:, 0] ** 2 + self.b * self.p
        return x

    def gaussian_init(self):
        m = torch.zeros(self.dim)
        m[1] = self.b * self.p
        s = torch.ones(self.dim)
        s[0] = math.sqrt(self.p)
        return (m, s)

    @property
    def log_normalizer(self):
        return 0.5 * math.log(2 * math.pi * self.p) + 0.5 * (self.dim - 1) * math.log(2 * math.pi)

    def tail_region(self, x):
        return x[..., 1] < -5 * self.b * self.p

    def shear_transport(self, c):
        from transports.analytic import BananaShear
        return BananaShear(self.dim, self.b, self.p, c, pairs=1)

    @property
    def test_functions(self):
        return [('x2', lambda x: x[..., 1], 0.0), ('x2^2', lambda x: x[..., 1] ** 2, 1.0 + 2.0 * (self.b * self.p) ** 2)]
