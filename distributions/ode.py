import math
import numpy as np
from .posterior import Posterior
YEARS = np.arange(1900, 1921)
LYNX = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6])
HARE = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])
N_SUB = 20

def _rk4_year(u, v, a, b, g, d, h: float, n: int):
    for _ in range(n):
        k1u, k1v = ((a - b * v) * u, (-g + d * u) * v)
        u2, v2 = (u + 0.5 * h * k1u, v + 0.5 * h * k1v)
        k2u, k2v = ((a - b * v2) * u2, (-g + d * u2) * v2)
        u3, v3 = (u + 0.5 * h * k2u, v + 0.5 * h * k2v)
        k3u, k3v = ((a - b * v3) * u3, (-g + d * u3) * v3)
        u4, v4 = (u + h * k3u, v + h * k3v)
        k4u, k4v = ((a - b * v4) * u4, (-g + d * u4) * v4)
        u = u + h / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return (u, v)
_RK4_COMPILED = None

class LotkaVolterra(Posterior):
    name = 'lotka_volterra'
    names = ['alpha', 'beta', 'gamma', 'delta']

    def __init__(self, n_sub=N_SUB, compiled=True):
        super().__init__(8)
        self.n_sub = n_sub
        self.compiled = compiled
        self.logy = np.log(np.stack([HARE, LYNX], axis=-1))

    def _trajectory_compiled(self, x):
        import torch
        global _RK4_COMPILED
        if _RK4_COMPILED is None:
            _RK4_COMPILED = torch.compile(_rk4_year, dynamic=True)
        shape = x.shape[:-1]
        xf = x.reshape(-1, x.shape[-1])
        a, b, g, d = torch.exp(xf[:, :4]).unbind(-1)
        u, v = (torch.exp(xf[:, 4]), torch.exp(xf[:, 5]))
        out = []
        for _ in range(20):
            u, v = _RK4_COMPILED(u, v, a, b, g, d, 1.0 / self.n_sub, self.n_sub)
            out.append(torch.stack([u, v], -1))
        z = torch.stack(out, -2).reshape(*shape, 20, 2)
        return torch.log(torch.clamp(z, 1e-10, 10000000000.0))

    def log_prob_twice_diff(self, x):
        from .backend import TORCH
        compiled, self.compiled = (self.compiled, False)
        try:
            return self.logp(x, TORCH)
        finally:
            self.compiled = compiled

    def trajectory(self, x, B):
        if B.name == 'torch' and self.compiled and (x[..., 0].numel() >= 64):
            return self._trajectory_compiled(x)
        th = B.exp(x[..., :4])
        a, b, g, d = (th[..., k:k + 1] for k in range(4))

        def f(z):
            u, v = (z[..., :1], z[..., 1:])
            return B.concat([(a - b * v) * u, (-g + d * u) * v], axis=-1)
        z0 = B.exp(x[..., 4:6])
        z = B.rk4(f, z0, 1.0 / self.n_sub, self.n_sub, 20)
        return B.log(B.clip(z, 1e-10, 10000000000.0))

    def logp(self, x, B):
        th = B.exp(x[..., :4])
        lp = -0.5 * ((th[..., 0] - 1.0) / 0.5) ** 2 - 0.5 * ((th[..., 2] - 1.0) / 0.5) ** 2 - 0.5 * ((th[..., 1] - 0.05) / 0.05) ** 2 - 0.5 * ((th[..., 3] - 0.05) / 0.05) ** 2 + x[..., :4].sum(-1)
        lz0, lsig = (x[..., 4:6], x[..., 6:8])
        lp = lp - 0.5 * ((lz0 - math.log(10.0)) ** 2).sum(-1)
        lp = lp - 0.5 * ((lsig + 1.0) ** 2).sum(-1)
        logy = self.c('logy', self.logy, B, x)
        lz = self.trajectory(x, B)
        res = B.concat([(logy[0] - lz0)[..., None, :], logy[1:] - lz], axis=-2)
        sig = B.exp(lsig)[..., None, :]
        return lp + (-lsig[..., None, :] - 0.5 * (res / sig) ** 2).sum((-1, -2))

    def features(self, x, B):
        return B.exp(x[..., :4])

    def init_point(self):
        return np.log([0.55, 0.028, 0.8, 0.024, 33.0, 6.0, 0.25, 0.25])
