import math
import numpy as np
from .posterior import Posterior
SENSOR_TRUE = np.array([[0.57, 0.91], [0.1, 0.37], [0.26, 0.14], [0.85, 0.04], [0.5, 0.3], [0.3, 0.7]])
SENSOR_R, SENSOR_SIGMA = (0.3, 0.02)

def simulate_sensor(locs=SENSOR_TRUE, R=SENSOR_R, sigma=SENSOR_SIGMA, seed=3):
    rng = np.random.default_rng(seed)
    n = len(locs)
    W = np.zeros((n, n))
    Y = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(locs[i] - locs[j])
            if rng.uniform() < math.exp(-dist ** 2 / (2 * R ** 2)):
                W[i, j] = W[j, i] = 1
                Y[i, j] = Y[j, i] = dist + sigma * rng.standard_normal()
    return (W, Y)

class SensorNetwork(Posterior):
    name = 'sensor'
    reference_method = 'pt'

    def __init__(self, seed=3):
        super().__init__(8)
        self.known = SENSOR_TRUE[4:]
        self.W, self.Y = simulate_sensor(seed=seed)
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6) if not (i >= 4 and j >= 4)]
        self.pi_ = np.array([p[0] for p in pairs])
        self.pj_ = np.array([p[1] for p in pairs])
        self.w = np.array([self.W[i, j] for i, j in pairs])
        self.yobs = np.array([self.Y[i, j] for i, j in pairs])
        self.names = [f"x{k // 2 + 1}{'ab'[k % 2]}" for k in range(8)]

    def _locs(self, x, B):
        known = self.c('known', self.known, B, x)
        unk = [x[..., 2 * k:2 * k + 2] for k in range(4)]
        kn = [known[k] + 0 * x[..., :2] for k in range(2)]
        return B.stack(unk + kn, axis=-2)

    def log_prior(self, x, B):
        return -0.5 * (x ** 2).sum(-1) / 100.0

    def log_lik(self, x, B):
        L = self._locs(x, B)
        a = L[..., list(self.pi_), :]
        b = L[..., list(self.pj_), :]
        d2 = ((a - b) ** 2).sum(-1)
        w = self.c('w', self.w, B, x)
        y = self.c('y', self.yobs, B, x)
        q = d2 / (2 * SENSOR_R ** 2)
        d = B.sqrt(d2 + 1e-300)
        obs = -q - 0.5 * ((y - d) / SENSOR_SIGMA) ** 2
        miss = B.log(-B.expm1(-B.clip(q, 1e-12, 1e+300)))
        return (w * obs + (1 - w) * miss).sum(-1)

    def logp(self, x, B):
        return self.log_prior(x, B) + self.log_lik(x, B)

    def features(self, x, B):
        return x

    def init_point(self):
        return np.full(8, 0.5)

    def pt_init(self, n, seed=0):
        return np.random.default_rng(seed).uniform(0, 1, (n, 8))

class Phi4(Posterior):
    reference_method = 'nuts_sym'
    names = ['1{M>0}', 'M', 'M^2', 'U/N']

    def __init__(self, N=16, a=0.1, beta=20.0):
        super().__init__(N)
        self.a, self.beta = (a, beta)
        self.delta = 1.0 / (N + 1)
        self.name = f'phi4_N{N}'

    def U(self, x, B):
        zero = 0 * x[..., :1]
        full = B.concat([zero, x, zero], axis=-1)
        grad = ((full[..., 1:] - full[..., :-1]) ** 2).sum(-1)
        pot = ((1 - x ** 2) ** 2).sum(-1)
        return self.a / (2 * self.delta) * grad + self.delta / (4 * self.a) * pot

    def logp(self, x, B):
        return -self.beta * self.U(x, B)

    def features(self, x, B):
        M = x.mean(-1)
        return B.stack([B.where(M > 0, 1.0 + 0 * M, 0 * M), M, M ** 2, self.U(x, B) / self.dim], axis=-1)

    def analytic_truth(self):
        return {'mean': [0.5, 0.0, None, None]}

    def init_point(self):
        t = np.arange(1, self.dim + 1) * self.delta
        return np.tanh(np.minimum(t, 1 - t) / (math.sqrt(2) * self.a))
