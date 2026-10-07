import json
import math
import numpy as np
from .posterior import DATA_DIR, Posterior
PDB = DATA_DIR / 'posteriordb'

def _data(name):
    return json.loads((PDB / f'{name}.json').read_text())

def _log_jac_sigmoid(u, B):
    return B.logsigmoid(u) + B.logsigmoid(-u)

class Dugongs(Posterior):
    name = 'pdb_dugongs'
    names = ['alpha', 'beta', 'lambda', 'sigma']

    def __init__(self):
        super().__init__(4)
        d = _data('dugongs_data')
        self.x, self.Y = (np.asarray(d['x'], float), np.asarray(d['Y'], float))

    def _unpack(self, x, B):
        a, b, u, v = (x[..., 0], x[..., 1], x[..., 2], x[..., 3])
        lam = 0.5 + 0.5 * B.sigmoid(u)
        return (a, b, lam, u, v)

    def logp(self, x, B):
        a, b, lam, u, v = self._unpack(x, B)
        xs = self.c('x', self.x, B, x)
        Y = self.c('Y', self.Y, B, x)
        m = a[..., None] - b[..., None] * B.exp(B.log(lam)[..., None] * xs)
        tau = B.exp(v)
        n = self.x.size
        lik = 0.5 * n * v - 0.5 * tau * ((Y - m) ** 2).sum(-1)
        prior = -a ** 2 / 2000000.0 - b ** 2 / 2000000.0 + _log_jac_sigmoid(u, B) + 0.0001 * v - 0.0001 * tau
        return lik + prior

    def features(self, x, B):
        a, b, lam, u, v = self._unpack(x, B)
        return B.stack([a, b, lam, B.exp(-0.5 * v)], axis=-1)

    def init_point(self):
        return np.array([2.6, 1.0, 0.0, math.log(100.0)])

def _garch_loglik(y, mu, a0, a1, b1, s1sq: float):
    import torch
    s2 = s1sq + 0.0 * mu
    lp = -0.5 * (y[0] - mu) ** 2 / s2 - 0.5 * torch.log(s2)
    for t in range(1, y.shape[0]):
        s2 = a0 + a1 * (y[t - 1] - mu) ** 2 + b1 * s2
        lp = lp - 0.5 * (y[t] - mu) ** 2 / s2 - 0.5 * torch.log(s2)
    return lp
_GARCH_COMPILED = None

class Garch11(Posterior):
    name = 'pdb_garch'
    names = ['mu', 'alpha0', 'alpha1', 'beta1']

    def __init__(self, compiled=True):
        super().__init__(4)
        d = _data('garch')
        self.y, self.sigma1 = (np.asarray(d['y'], float), float(d['sigma1']))
        self.compiled = compiled

    def _unpack(self, x, B):
        mu, u0, u1, u2 = (x[..., 0], x[..., 1], x[..., 2], x[..., 3])
        a0 = B.exp(u0)
        a1 = B.sigmoid(u1)
        b1 = (1.0 - a1) * B.sigmoid(u2)
        return (mu, a0, a1, b1, u0, u1, u2)

    def logp(self, x, B):
        mu, a0, a1, b1, u0, u1, u2 = self._unpack(x, B)
        y = self.c('y', self.y, B, x)
        if B.name == 'torch' and self.compiled and (mu.numel() >= 2):
            import torch
            global _GARCH_COMPILED
            if _GARCH_COMPILED is None:
                _GARCH_COMPILED = torch.compile(_garch_loglik, dynamic=True)
            shape = mu.shape
            lp = _GARCH_COMPILED(y, mu.reshape(-1), a0.reshape(-1), a1.reshape(-1), b1.reshape(-1), self.sigma1 ** 2).reshape(shape)
        else:
            s2 = self.sigma1 ** 2 + 0.0 * mu
            lp = -0.5 * (y[0] - mu) ** 2 / s2 - 0.5 * B.log(s2)
            for t in range(1, self.y.size):
                s2 = a0 + a1 * (y[t - 1] - mu) ** 2 + b1 * s2
                lp = lp - 0.5 * (y[t] - mu) ** 2 / s2 - 0.5 * B.log(s2)
        jac = u0 + _log_jac_sigmoid(u1, B) + B.logsigmoid(-u1) + _log_jac_sigmoid(u2, B)
        return lp + jac

    def log_prob_twice_diff(self, x):
        from .backend import TORCH
        compiled, self.compiled = (self.compiled, False)
        try:
            return self.logp(x, TORCH)
        finally:
            self.compiled = compiled

    def features(self, x, B):
        mu, a0, a1, b1, *_ = self._unpack(x, B)
        return B.stack([mu, a0, a1, b1], axis=-1)

    def init_point(self):
        return np.array([5.0, math.log(1.5), 0.3, -0.5])
N_SUB = 25

def _rk4_pk(y, t, ka, Km, Vm, D, V, h: float, n: int):
    import torch
    for _ in range(n):
        f = lambda yy, tt: torch.exp(-ka * tt) * D * ka / V - Vm / V * yy / (Km + yy)
        k1 = f(y, t)
        k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(y + h * k3, t + h)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return y
_PK_COMPILED = None

class OneCompMM(Posterior):
    name = 'pdb_one_comp'
    names = ['k_a', 'log K_m', 'log V_m', 'sigma']

    def __init__(self, n_sub=N_SUB, compiled=True):
        super().__init__(4)
        d = _data('one_comp_mm_elim_abs')
        self.D, self.V = (float(d['D']), float(d['V']))
        self.times = np.asarray(d['times'], float)
        self.logC = np.log(np.asarray(d['C_hat'], float))
        self.n_sub, self.compiled = (n_sub, compiled)
        assert np.allclose(np.diff(self.times), 0.5) and self.times[0] == 0.5

    def _traj_compiled(self, th):
        import torch
        global _PK_COMPILED
        if _PK_COMPILED is None:
            _PK_COMPILED = torch.compile(_rk4_pk, dynamic=True)
        shape = th.shape[:-1]
        tf = th.reshape(-1, 3)
        ka, Km, Vm = tf.unbind(-1)
        y = torch.zeros_like(ka)
        t = torch.zeros_like(ka)
        h = 0.5 / self.n_sub
        out = []
        for _ in range(self.times.size):
            y = _PK_COMPILED(y, t, ka, Km, Vm, self.D, self.V, h, self.n_sub)
            t = t + 0.5
            out.append(y)
        return torch.stack(out, -1).reshape(*shape, -1)

    def trajectory(self, th, B):
        if B.name == 'torch' and self.compiled and (th[..., 0].numel() >= 64):
            return self._traj_compiled(th)
        ka, Km, Vm = (th[..., 0:1], th[..., 1:2], th[..., 2:3])
        h = 0.5 / self.n_sub

        def f(z):
            y, t = (z[..., 0:1], z[..., 1:2])
            dy = B.exp(-ka * t) * self.D * ka / self.V - Vm / self.V * y / (Km + y)
            return B.concat([dy, 1.0 + 0.0 * t], axis=-1)
        z0 = 0.0 * B.concat([ka, ka], axis=-1)
        z = B.rk4(f, z0, h, self.n_sub, self.times.size)
        return z[..., 0]

    def logp(self, x, B):
        th = B.exp(x[..., :3])
        sig = B.exp(x[..., 3])
        C = B.clip(self.trajectory(th, B), 1e-12, 1000000000000.0)
        logC = self.c('logC', self.logC, B, x)
        lik = (-x[..., 3][..., None] - 0.5 * ((logC - B.log(C)) / sig[..., None]) ** 2).sum(-1)
        prior = (-B.log1p(B.exp(2 * x)) + x).sum(-1)
        return lik + prior

    def log_prob_twice_diff(self, x):
        from .backend import TORCH
        compiled, self.compiled = (self.compiled, False)
        try:
            return self.logp(x, TORCH)
        finally:
            self.compiled = compiled

    def features(self, x, B):
        return B.stack([B.exp(x[..., 0]), x[..., 1], x[..., 2], B.exp(x[..., 3])], axis=-1)

    def init_point(self):
        return np.log([0.76, 1.0, 0.9, 0.13])

class GPPoisson(Posterior):
    name = 'pdb_gp_pois'
    names = ['rho', 'alpha', 'f1', 'f6']

    def __init__(self):
        d = _data('gp_pois_regr')
        self.x, self.k = (np.asarray(d['x'], float), np.asarray(d['k'], float))
        self.N = self.x.size
        super().__init__(2 + self.N)
        self.D2 = (self.x[:, None] - self.x[None, :]) ** 2

    def _f(self, x, B):
        lrho, lalpha, ft = (x[..., 0], x[..., 1], x[..., 2:])
        D2 = self.c('D2', self.D2, B, x)
        I = self.c('I', np.eye(self.N), B, x)
        cov = B.exp(2 * lalpha)[..., None, None] * B.exp(-D2 / (2 * B.exp(2 * lrho))[..., None, None]) + 1e-10 * I
        L = B.cholesky(cov)
        return (L @ ft[..., None])[..., 0]

    def logp(self, x, B):
        lrho, lalpha, ft = (x[..., 0], x[..., 1], x[..., 2:])
        f = self._f(x, B)
        k = self.c('k', self.k, B, x)
        lik = (k * f - B.exp(f)).sum(-1)
        prior = 25.0 * lrho - 4.0 * B.exp(lrho) - B.exp(2 * lalpha) / 8.0 + lalpha - 0.5 * (ft ** 2).sum(-1)
        return lik + prior

    def features(self, x, B):
        f = self._f(x, B)
        return B.stack([B.exp(x[..., 0]), B.exp(x[..., 1]), f[..., 0], f[..., 5]], axis=-1)

    def init_point(self):
        return np.concatenate([[math.log(5.7), math.log(2.9)], np.zeros(self.N)])

class Surgical(Posterior):
    name = 'pdb_surgical'
    names = ['mu', 'log sigma2', 'b1', 'pop_mean']

    def __init__(self):
        d = _data('surgical_data')
        self.n, self.r = (np.asarray(d['n'], float), np.asarray(d['r'], float))
        self.N = self.n.size
        super().__init__(2 + self.N)

    def logp(self, x, B):
        mu, v, b = (x[..., 0], x[..., 1], x[..., 2:])
        n = self.c('n', self.n, B, x)
        r = self.c('r', self.r, B, x)
        lik = (r * b - n * B.softplus(b)).sum(-1)
        hier = -0.5 * self.N * v - 0.5 * ((b - mu[..., None]) ** 2).sum(-1) * B.exp(-v)
        prior = -mu ** 2 / 2000000.0 - 0.001 * v - 0.001 * B.exp(-v)
        return lik + hier + prior

    def features(self, x, B):
        mu = x[..., 0]
        return B.stack([mu, x[..., 1], x[..., 2], B.sigmoid(mu)], axis=-1)
    reference_method = 'nuts_ref'

    def reference_logp(self, x, B):
        mu, v, eta = (x[..., 0], x[..., 1], x[..., 2:])
        b = mu[..., None] + B.exp(0.5 * v)[..., None] * eta
        n = self.c('n', self.n, B, x)
        r = self.c('r', self.r, B, x)
        lik = (r * b - n * B.softplus(b)).sum(-1)
        prior = -mu ** 2 / 2000000.0 - 0.001 * v - 0.001 * B.exp(-v)
        return lik - 0.5 * (eta ** 2).sum(-1) + prior

    def reference_to_x(self, xr):
        x = np.array(xr, copy=True)
        x[..., 2:] = xr[..., :1] + np.exp(0.5 * xr[..., 1:2]) * xr[..., 2:]
        return x

    def reference_init(self):
        return np.concatenate([[-2.5, math.log(0.1)], np.zeros(self.N)])

    def init_point(self):
        return np.concatenate([[-2.5, math.log(0.1)], np.full(self.N, -2.5)])

    def init_scale(self):
        return np.concatenate([[0.3, 1.0], np.full(self.N, 0.4)])

class Pilots(Posterior):
    name = 'pdb_pilots'
    names = ['mu_a', 'mu_b', 'log sigma_a', 'log sigma_b', 'sigma_y']

    def __init__(self):
        d = _data('pilots')
        self.y = np.asarray(d['y'], float)
        self.g = np.asarray(d['group_id'], int) - 1
        self.s = np.asarray(d['scenario_id'], int) - 1
        self.G, self.S = (int(d['n_groups']), int(d['n_scenarios']))
        super().__init__(self.G + self.S + 5)

    def _unpack(self, x, B):
        G, S = (self.G, self.S)
        a, b = (x[..., :G], x[..., G:G + S])
        mu_a, mu_b = (x[..., G + S], x[..., G + S + 1])
        u = x[..., G + S + 2:]
        sig = 100.0 * B.sigmoid(u)
        return (a, b, mu_a, mu_b, sig, u)

    def logp(self, x, B):
        a, b, mu_a, mu_b, sig, u = self._unpack(x, B)
        sa, sb, sy = (sig[..., 0], sig[..., 1], sig[..., 2])
        y = self.c('y', self.y, B, x)
        yh = a[..., list(self.g)] + b[..., list(self.s)]
        n = self.y.size
        lik = -n * B.log(sy) - 0.5 * ((y - yh) ** 2).sum(-1) / sy ** 2
        ha = -self.G * B.log(sa) - 0.5 * ((a - 10 * mu_a[..., None]) ** 2).sum(-1) / sa ** 2
        hb = -self.S * B.log(sb) - 0.5 * ((b - 10 * mu_b[..., None]) ** 2).sum(-1) / sb ** 2
        prior = -0.5 * mu_a ** 2 - 0.5 * mu_b ** 2 + _log_jac_sigmoid(u, B).sum(-1)
        return lik + ha + hb + prior

    def features(self, x, B):
        a, b, mu_a, mu_b, sig, u = self._unpack(x, B)
        return B.stack([mu_a, mu_b, B.log(sig[..., 0]), B.log(sig[..., 1]), sig[..., 2]], axis=-1)
    reference_method = 'nuts_ref'

    def reference_logp(self, x, B):
        ea, eb, mu_a, mu_b, sig, u = self._unpack(x, B)
        sa, sb, sy = (sig[..., 0], sig[..., 1], sig[..., 2])
        a = 10 * mu_a[..., None] + sa[..., None] * ea
        b = 10 * mu_b[..., None] + sb[..., None] * eb
        y = self.c('y', self.y, B, x)
        yh = a[..., list(self.g)] + b[..., list(self.s)]
        n = self.y.size
        lik = -n * B.log(sy) - 0.5 * ((y - yh) ** 2).sum(-1) / sy ** 2
        prior = -0.5 * (ea ** 2).sum(-1) - 0.5 * (eb ** 2).sum(-1) - 0.5 * mu_a ** 2 - 0.5 * mu_b ** 2 + _log_jac_sigmoid(u, B).sum(-1)
        return lik + prior

    def reference_to_x(self, xr):
        G, S = (self.G, self.S)
        x = np.array(xr, copy=True)
        sig = 100.0 / (1.0 + np.exp(-xr[..., G + S + 2:]))
        x[..., :G] = 10 * xr[..., G + S:G + S + 1] + sig[..., 0:1] * xr[..., :G]
        x[..., G:G + S] = 10 * xr[..., G + S + 1:G + S + 2] + sig[..., 1:2] * xr[..., G:G + S]
        return x

    def reference_init(self):
        p = self.init_point()
        p[:self.G + self.S] = 0.0
        return p

    def init_point(self):
        lg = lambda s: math.log(s / (100.0 - s))
        return np.concatenate([np.full(self.G, 0.44), np.zeros(self.S), [0.044, 0.0, lg(0.2), lg(0.3), lg(0.3)]])

    def init_scale(self):
        return np.concatenate([np.full(self.G, 0.3), np.full(self.S, 0.3), [0.3, 0.3, 1.0, 1.0, 0.5]])
PDB_POSTERIORS = {'pdb_dugongs': Dugongs, 'pdb_garch': Garch11, 'pdb_one_comp': OneCompMM, 'pdb_gp_pois': GPPoisson, 'pdb_surgical': Surgical, 'pdb_pilots': Pilots}
