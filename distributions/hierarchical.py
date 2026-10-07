import math
import numpy as np
from .posterior import Posterior
SCHOOLS_Y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SCHOOLS_SIGMA = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

class EightSchools(Posterior):
    name = 'eight_schools'
    names = ['mu', 'log tau', 'theta1']
    reference_method = 'nuts_ref'

    def __init__(self):
        super().__init__(10)

    def _prior_mu_tau(self, mu, logtau, B):
        tau = B.exp(logtau)
        return -mu ** 2 / 50.0 - B.log1p((tau / 5.0) ** 2) + logtau

    def logp(self, x, B):
        mu, logtau, theta = (x[..., 0], x[..., 1], x[..., 2:])
        y = self.c('y', SCHOOLS_Y, B, x)
        sig = self.c('s', SCHOOLS_SIGMA, B, x)
        return self._prior_mu_tau(mu, logtau, B) - 8 * logtau - 0.5 * (((theta - mu[..., None]) / B.exp(logtau)[..., None]) ** 2).sum(-1) - 0.5 * (((y - theta) / sig) ** 2).sum(-1)

    def features(self, x, B):
        return B.stack([x[..., 0], x[..., 1], x[..., 2]], axis=-1)

    def reference_logp(self, x, B):
        mu, logtau, eta = (x[..., 0], x[..., 1], x[..., 2:])
        y = self.c('y', SCHOOLS_Y, B, x)
        sig = self.c('s', SCHOOLS_SIGMA, B, x)
        theta = mu[..., None] + B.exp(logtau)[..., None] * eta
        return self._prior_mu_tau(mu, logtau, B) - 0.5 * (eta ** 2).sum(-1) - 0.5 * (((y - theta) / sig) ** 2).sum(-1)

    def reference_to_x(self, xr):
        x = np.array(xr, copy=True)
        x[..., 2:] = xr[..., :1] + np.exp(xr[..., 1:2]) * xr[..., 2:]
        return x

    def reference_init(self):
        return np.concatenate([[5.0, 1.0], np.zeros(8)])

    def init_point(self):
        return np.concatenate([[5.0, 1.0], np.full(8, 5.0)])

    def init_scale(self):
        return np.concatenate([[4.0, 1.0], np.full(8, 6.0)])
SV_TRUE = dict(m=-1.0, phi=0.95, s2=0.05)
SV_ADT = dict(m=-0.85, phi=0.98, s2=0.15 ** 2)

def simulate_sv(T, m, phi, s2, seed=0):
    rng = np.random.default_rng(seed)
    s = math.sqrt(s2)
    h = np.empty(T)
    h[0] = m + s / math.sqrt(1 - phi ** 2) * rng.standard_normal()
    for t in range(1, T):
        h[t] = m + phi * (h[t - 1] - m) + s * rng.standard_normal()
    return (np.exp(h / 2) * rng.standard_normal(T), h)

class StochasticVolatility(Posterior):
    names = ['m', 'phi', 's2', 'mean h']
    reference_method = 'nuts_ref'

    def __init__(self, T=50, returns=None, seed=0, name=None, params=None):
        if returns is None:
            returns, self.h_true = simulate_sv(T, seed=seed, **params or SV_TRUE)
        self.r = np.asarray(returns, dtype=float)
        self.T = len(self.r)
        super().__init__(self.T + 3)
        self.name = name or f'stochvol_T{self.T}'

    @staticmethod
    def _prior(m, u, v, B):
        return -m ** 2 / 20.0 - 19.0 * B.softplus(-2 * u) - 2.5 * B.softplus(2 * u) + 2 * u + 0.5 * v - 0.5 * B.exp(v)

    def logp(self, x, B):
        m, u, v, h = (x[..., 0], x[..., 1], x[..., 2], x[..., 3:])
        r2 = self.c('r2', self.r ** 2, B, x)
        phi = B.tanh(u)
        s2 = B.exp(v)
        log1mphi2 = math.log(4.0) + 2 * u - 2 * B.softplus(2 * u)
        hc = h - m[..., None]
        first = 0.5 * (log1mphi2 - v) - 0.5 * hc[..., 0] ** 2 * B.exp(log1mphi2 - v)
        innov = hc[..., 1:] - phi[..., None] * hc[..., :-1]
        trans = -0.5 * (self.T - 1) * v - 0.5 * (innov ** 2).sum(-1) / s2
        obs = (-0.5 * h - 0.5 * r2 * B.exp(-h)).sum(-1)
        return self._prior(m, u, v, B) + first + trans + obs

    def features(self, x, B):
        return B.stack([x[..., 0], B.tanh(x[..., 1]), B.exp(x[..., 2]), x[..., 3:].mean(-1)], axis=-1)

    def reference_logp(self, x, B):
        m, u, v, w = (x[..., 0], x[..., 1], x[..., 2], x[..., 3:])
        r2 = self.c('r2', self.r ** 2, B, x)
        h = self._h_from_w(m, u, v, w, B)
        obs = (-0.5 * h - 0.5 * r2 * B.exp(-h)).sum(-1)
        return self._prior(m, u, v, B) - 0.5 * (w ** 2).sum(-1) + obs

    def _h_from_w(self, m, u, v, w, B):
        phi = B.tanh(u)[..., None]
        s = B.exp(0.5 * v)[..., None]
        sd1 = s / B.sqrt(1 - phi ** 2)
        hs = [sd1[..., 0] * w[..., 0]]
        for t in range(1, self.T):
            hs.append(phi[..., 0] * hs[-1] + s[..., 0] * w[..., t])
        return B.stack(hs, axis=-1) + m[..., None]

    def reference_to_x(self, xr):
        m, u, v, w = (xr[..., 0], xr[..., 1], xr[..., 2], xr[..., 3:])
        phi, s = (np.tanh(u), np.exp(0.5 * v))
        h = np.empty_like(w)
        h[..., 0] = s / np.sqrt(1 - phi ** 2) * w[..., 0]
        for t in range(1, self.T):
            h[..., t] = phi * h[..., t - 1] + s * w[..., t]
        x = np.array(xr, copy=True)
        x[..., 3:] = h + m[..., None]
        return x

    def reference_init(self):
        return np.concatenate([[math.log(np.mean(self.r ** 2)), 1.5, math.log(0.1)], np.zeros(self.T)])

    def init_point(self):
        lv = math.log(np.mean(self.r ** 2))
        return np.concatenate([[lv, math.atanh(0.9), math.log(0.1)], np.full(self.T, lv)])

    def init_scale(self):
        return np.concatenate([[0.5, 0.5, 0.7], np.full(self.T, 1.0)])
