import torch
from transports.base import Transport

class Affine(Transport):

    def __init__(self, m, L, E=1):
        super().__init__()
        self.E, self.dim = (E, m.numel())
        self.register_buffer('m', m)
        self.register_buffer('L', L)
        self.register_buffer('Li', torch.linalg.inv(L))
        self.register_buffer('ld', torch.log(torch.diagonal(L).abs()).sum())

    def forward(self, z):
        return (self.m + z @ self.L.T, self.ld.expand(z.shape[:-1]))

    def inverse(self, x):
        return ((x - self.m) @ self.Li.T, (-self.ld).expand(x.shape[:-1]))

def gaussian_approx(target, rtol=0.01):
    m, s = target.laplace()
    m = m.detach().clone()
    try:
        x = m.clone().requires_grad_(True)
        H = torch.autograd.functional.hessian(lambda v: target.log_prob(v[None])[0], x)
        S = torch.linalg.inv(-H)
        S = 0.5 * (S + S.T)
        L = torch.linalg.cholesky(S)
        if torch.allclose(torch.diagonal(S).sqrt(), s, rtol=rtol):
            return (m, L)
    except (RuntimeError, torch.linalg.LinAlgError):
        pass
    return (m, torch.diag(s))

def affine_transport(target, E=1, identity=False):
    d = target.dim
    if identity:
        return Affine(torch.zeros(d), torch.eye(d), E)
    m, L = gaussian_approx(target)
    return Affine(m, L, E)
