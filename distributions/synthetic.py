import math
import torch
from .haario import HaarioBanana
from .funnel import NealFunnel

class BananaBench(HaarioBanana):
    names = ['x2', 'x2^2', 'x1^2', 'mean x2k', 'mean x']
    tex = ['$x_2$', '$x_2^2$', '$x_1^2$', '$\\frac{2}{d}\\sum_k x_{2k}$', '$\\frac{1}{d}\\sum_i x_i$']
    tail_tex = '$P(x_2<-5)$'

    def __init__(self, dim=8, b=0.1, p=10.0):
        super().__init__(dim, p=p, b=b)

    def logp_and_grad(self, x):
        x1, x2 = (x[..., 0], x[..., 1])
        r = x2 + self.b * x1 ** 2 - self.b * self.p
        lp = -x1 ** 2 / (2 * self.p) - 0.5 * r ** 2 - 0.5 * (x[..., 2:] ** 2).sum(-1)
        g = -x.clone()
        g[..., 0] = -x1 / self.p - 2 * self.b * x1 * r
        g[..., 1] = -r
        return (lp, g)

    def grad_log_prob(self, x):
        return self.logp_and_grad(x)[1]

    def F(self, x):
        return torch.stack([x[..., 1], x[..., 1] ** 2, x[..., 0] ** 2, x[..., 1::2].mean(-1), x.mean(-1)], dim=-1)

    @property
    def test_functions(self):
        return [(n, lambda x, j=j: self.F(x)[..., j], None) for j, n in enumerate(self.names)]

    def laplace(self):
        return self.gaussian_init()

    @torch.no_grad()
    def ground_truth(self, n=2000000, chunk=500000, seed=12345):
        gen = torch.Generator(device=torch.get_default_device())
        gen.manual_seed(seed)
        s1 = s2 = 0.0
        tail = 0.0
        for s in range(0, n, chunk):
            x = self.sample(min(chunk, n - s), generator=gen)
            F = self.F(x)
            s1 = s1 + F.sum(0)
            s2 = s2 + (F * F).sum(0)
            tail += self.tail_region(x).sum().item()
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        return {'mean': self.analytic_truth()['mean'], 'mean_mc': mean.tolist(), 'var': var.tolist(), 'p_tail': tail / n}

    def analytic_truth(self):
        v2 = 1 + 2 * self.b ** 2 * self.p ** 2
        return {'mean': [0.0, v2, self.p, 0.0, 0.0], 'var_x2': v2}

class FunnelBench(NealFunnel):
    names = ['v', 'mean log(1+x^2)', 'v^2', 'x1']
    tex = ['$v$', '$\\frac{1}{d-1}\\sum_i\\log(1+x_i^2)$', '$v^2$', '$x_1$']
    tail_tex = '$P(v<-4)$'

    def __init__(self, dim=10, sigma_v=2.0):
        super().__init__(dim, sigma_v)

    def logp_and_grad(self, x):
        v, rest = (x[..., 0], x[..., 1:])
        ev = torch.exp(-v)
        r2 = (rest ** 2).sum(-1)
        lp = -v ** 2 / (2 * self.sigma_v ** 2) - 0.5 * (self.dim - 1) * v - 0.5 * ev * r2
        g = torch.empty_like(x)
        g[..., 0] = -v / self.sigma_v ** 2 - 0.5 * (self.dim - 1) + 0.5 * ev * r2
        g[..., 1:] = -ev[..., None] * rest
        return (lp, g)

    def grad_log_prob(self, x):
        return self.logp_and_grad(x)[1]

    def F(self, x):
        v = x[..., 0]
        return torch.stack([v, torch.log1p(x[..., 1:] ** 2).mean(-1), v * v, x[..., 1]], dim=-1)

    @property
    def test_functions(self):
        return [(n, lambda x, j=j: self.F(x)[..., j], None) for j, n in enumerate(self.names)]

    def laplace(self):
        return self.gaussian_init()

    @torch.no_grad()
    def ground_truth(self, n=2000000, chunk=500000, seed=12345):
        gen = torch.Generator(device=torch.get_default_device())
        gen.manual_seed(seed)
        s1 = s2 = 0.0
        tail = 0.0
        for s in range(0, n, chunk):
            x = self.sample(min(chunk, n - s), generator=gen)
            F = self.F(x)
            s1 = s1 + F.sum(0)
            s2 = s2 + (F * F).sum(0)
            tail += self.tail_region(x).sum().item()
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        return {'mean': self.analytic_truth()['mean'], 'mean_mc': mean.tolist(), 'var': var.tolist(), 'p_tail': tail / n}

    def analytic_truth(self):
        s2 = self.sigma_v ** 2
        return {'mean': [0.0, self.mean_log1p(), s2, 0.0], 'var_v': s2, 'var_x1': math.exp(s2 / 2)}
