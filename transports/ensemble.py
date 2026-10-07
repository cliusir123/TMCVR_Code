import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .base import Transport
SQRT2 = math.sqrt(2.0)
SQRT_PI_2 = math.sqrt(math.pi / 2)

def log_erfc(x):
    return torch.log(torch.special.erfcx(x)) - x * x

def _init_linear(E, n_in, n_out, zero=False):
    W = torch.zeros(E, n_in, n_out)
    b = torch.zeros(E, 1, n_out)
    if not zero:
        bound = 1.0 / math.sqrt(n_in)
        W.uniform_(-bound, bound)
        b.uniform_(-bound, bound)
    return (nn.Parameter(W), nn.Parameter(b))

class EnsembleCoupling(nn.Module):

    def __init__(self, E, mask, hidden, scale_bound=1.5):
        super().__init__()
        d = mask.numel()
        self.register_buffer('mask', mask)
        self.W1, self.b1 = _init_linear(E, d, hidden)
        self.W2, self.b2 = _init_linear(E, hidden, hidden)
        self.W3, self.b3 = _init_linear(E, hidden, 2 * d, zero=True)
        self.scale_bound = scale_bound

    def _s_t(self, h):
        a = torch.tanh(torch.baddbmm(self.b1, h * self.mask, self.W1))
        a = torch.tanh(torch.baddbmm(self.b2, a, self.W2))
        s, t = torch.baddbmm(self.b3, a, self.W3).chunk(2, dim=-1)
        free = 1.0 - self.mask
        return (self.scale_bound * torch.tanh(s) * free, t * free)

    def forward(self, h):
        s, t = self._s_t(h)
        return (h * torch.exp(s) + t, s.sum(-1))

    def inverse(self, y):
        s, t = self._s_t(y)
        return ((y - t) * torch.exp(-s), -s.sum(-1))

class EnsembleFlow(Transport):

    def __init__(self, E, dim, m, s, n_layers=8, hidden=64, ttf=False, ttf_init=0.05, scale_bound=1.5):
        super().__init__()
        self.E, self.dim = (E, dim)
        dtype = torch.get_default_dtype()
        masks = [torch.tensor([(j + k) % 2 for j in range(dim)], dtype=dtype) for k in range(n_layers)]
        self.layers = nn.ModuleList([EnsembleCoupling(E, mk, hidden, scale_bound) for mk in masks])
        self.m = nn.Parameter(m.expand(E, 1, dim).clone())
        self.log_s = nn.Parameter(torch.log(s).expand(E, 1, dim).clone())
        self.ttf = ttf
        if ttf:
            raw = math.log(math.expm1(ttf_init))
            self.lam_raw = nn.Parameter(torch.full((E, 2, dim), raw))

    def _split(self, z):
        if z.dim() == 3:
            return (z, None)
        B, d = z.shape
        return (z.reshape(self.E, B // self.E, d), z.shape)

    @staticmethod
    def _merge(y, ld, shape):
        if shape is None:
            return (y, ld)
        return (y.reshape(shape), ld.reshape(shape[:-1]))

    def _lam(self, u):
        lam = Fn.softplus(self.lam_raw)
        return torch.where(u > 0, lam[:, :1], lam[:, 1:])

    def _ttf(self, u):
        lam = self._lam(u)
        L = log_erfc(u.abs() / SQRT2)
        v = SQRT_PI_2 * torch.sign(u) * torch.expm1(-lam * L) / lam
        return (v, (-0.5 * u * u - (lam + 1) * L).sum(-1))

    def _ttf_inv(self, v):
        lam = self._lam(v)
        q = torch.exp(-torch.log1p(lam * v.abs() / SQRT_PI_2) / lam)
        u = torch.sign(v) * -torch.special.ndtri(q / 2)
        L = log_erfc(u.abs() / SQRT2)
        return (u, -(-0.5 * u * u - (lam + 1) * L).sum(-1))

    def forward(self, z):
        h, shape = self._split(z)
        ld = torch.zeros(h.shape[:-1], dtype=h.dtype, device=h.device)
        for layer in self.layers:
            h, s = layer(h)
            ld = ld + s
        if self.ttf:
            h, s = self._ttf(h)
            ld = ld + s
        x = self.m + torch.exp(self.log_s) * h
        return self._merge(x, ld + self.log_s.sum(-1), shape)

    def inverse(self, x):
        v, shape = self._split(x)
        h = (v - self.m) * torch.exp(-self.log_s)
        ld = -self.log_s.sum(-1).expand(h.shape[:-1]).clone()
        if self.ttf:
            h, s = self._ttf_inv(h)
            ld = ld + s
        for layer in reversed(self.layers):
            h, s = layer.inverse(h)
            ld = ld + s
        return self._merge(h, ld, shape)

    @torch.no_grad()
    def push(self, z, chunk=None):
        return self.forward(z)[0]

def replicate(flow, E):
    import copy
    from .linear import FixedLinear
    if isinstance(flow, FixedLinear):
        inner = replicate(flow.flow, E)
        L = flow.L if flow.L.dim() == 2 else flow.L[:1].expand(E, -1, -1)
        return FixedLinear(inner, L.clone())
    assert flow.E == 1, 'replicate a single flow'
    out = copy.deepcopy(flow)
    with torch.no_grad():
        for p in out.parameters():
            p.data = p.data.expand(E, *p.shape[1:]).clone()
    out.E = E
    return out.requires_grad_(False)

class SharedFlow(Transport):

    def __init__(self, flow, E):
        super().__init__()
        self.flow, self.E, self.dim = (flow, E, flow.dim)

    def forward(self, z):
        shape = z.shape
        y, ld = self.flow(z.reshape(-1, shape[-1]))
        return (y.reshape(shape), ld.reshape(shape[:-1]))
