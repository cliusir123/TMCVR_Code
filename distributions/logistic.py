import numpy as np
from .posterior import DATA_DIR, Posterior

def load_mat(dataset):
    import scipy.io as sio
    X = np.asarray(sio.loadmat(DATA_DIR / f'{dataset}.mat')['X'], dtype=float)
    Z, y = (X[:, :-1], X[:, -1])
    y = (y == y.max()).astype(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    Z = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    return (Z, y)

class LogisticRegression(Posterior):

    def __init__(self, dataset='german'):
        self.Z, self.y = load_mat(dataset)
        super().__init__(self.Z.shape[1])
        self.name = f'logistic_{dataset}'
        self.names = ['beta0'] + [f'beta{j}' for j in range(1, self.dim)]

    def logp(self, x, B):
        Z = self.c('Z', self.Z.T, B, x)
        y = self.c('y', self.y, B, x)
        eta = x @ Z
        return (y * eta - B.softplus(eta)).sum(-1)

    def features(self, x, B):
        return x

class SparseLogistic(Posterior):

    def __init__(self, dataset='german'):
        self.Z, self.y = load_mat(dataset)
        self.p = self.Z.shape[1]
        super().__init__(2 * self.p + 1)
        self.name = f'sparse_logistic_{dataset}'
        self.names = ['log tau'] + [f'coef{j}' for j in range(self.p)]

    def split(self, x):
        p = self.p
        return (x[..., :p], x[..., p:2 * p], x[..., 2 * p])

    def logp(self, x, B):
        beta, loglam, logtau = self.split(x)
        Z = self.c('Z', self.Z.T, B, x)
        y = self.c('y', self.y, B, x)
        w = B.exp(logtau)[..., None] * beta * B.exp(loglam)
        eta = w @ Z
        lik = (y * eta - B.softplus(eta)).sum(-1)
        gam = lambda u: 0.5 * u - 0.5 * B.exp(u)
        return lik - 0.5 * (beta ** 2).sum(-1) + gam(loglam).sum(-1) + gam(logtau)

    def features(self, x, B):
        beta, loglam, logtau = self.split(x)
        coef = B.exp(logtau)[..., None] * beta * B.exp(loglam)
        return B.concat([logtau[..., None], coef], axis=-1)

    def init_point(self):
        return np.concatenate([np.zeros(self.p), np.zeros(self.p), [0.0]])
