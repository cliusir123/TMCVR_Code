import torch
from .base import Transport

class Precomposed(Transport):

    def __init__(self, transport, S):
        super().__init__()
        self.inner = transport
        evals, evecs = torch.linalg.eigh(S)
        root = evecs @ torch.diag(evals.clamp_min(1e-12).sqrt()) @ evecs.T
        self.register_buffer('root', root)
        self.register_buffer('logdet_root', 0.5 * torch.log(evals.clamp_min(1e-12)).sum())

    def forward(self, z):
        x, ld = self.inner(z @ self.root.T)
        return (x, ld + self.logdet_root)

class FixedLinear(Transport):

    def __init__(self, flow, L):
        super().__init__()
        self.flow, self.E, self.dim = (flow, flow.E, flow.dim)
        self.register_buffer('L', L)
        self.register_buffer('Li', torch.linalg.inv(L))
        ld = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs()).sum(-1)
        self.register_buffer('ld', ld)

    def _mul(self, M, x):
        return apply_per_replicate(M, x, self.E)

    def _add_ld(self, ld, sign):
        return add_per_replicate(ld, sign * self.ld, self.E)

    def forward(self, z):
        y, ld = self.flow(z)
        return (self._mul(self.L, y), self._add_ld(ld, 1.0))

    def inverse(self, x):
        z, ld = self.flow.inverse(self._mul(self.Li, x))
        return (z, self._add_ld(ld, -1.0))

    def log_prob(self, x):
        import math
        z, ld = self.inverse(x)
        return -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi) + ld

def apply_per_replicate(M, x, E):
    if M.dim() == 2:
        return x @ M.T
    shape = x.shape
    xe = x.reshape(E, -1, shape[-1])
    return torch.einsum('eij,enj->eni', M, xe).reshape(shape)

def add_per_replicate(v, c, E):
    if c.dim() == 0:
        return v + c
    shape = v.shape
    return (v.reshape(E, -1) + c[:, None]).reshape(shape)
