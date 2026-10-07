import torch
import torch.nn as nn
from .base import Transport

class Coupling(nn.Module):

    def __init__(self, mask, hidden, scale_bound=1.5):
        super().__init__()
        self.register_buffer('mask', mask)
        d = mask.numel()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2 * d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_bound = scale_bound

    def _s_t(self, h):
        s, t = self.net(h * self.mask).chunk(2, dim=-1)
        free = 1.0 - self.mask
        return (self.scale_bound * torch.tanh(s) * free, t * free)

    def forward(self, h):
        s, t = self._s_t(h)
        return (h * torch.exp(s) + t, s.sum(-1))

    def inverse(self, y):
        s, t = self._s_t(y)
        return ((y - t) * torch.exp(-s), -s.sum(-1))

class RealNVP(Transport):

    def __init__(self, dim, n_layers=8, hidden=64):
        super().__init__()
        dtype = torch.get_default_dtype()
        masks = [torch.tensor([(j + k) % 2 for j in range(dim)], dtype=dtype) for k in range(n_layers)]
        self.layers = nn.ModuleList([Coupling(m, hidden) for m in masks])
        self.dim = dim

    def forward(self, z):
        h = z
        logdet = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        for layer in self.layers:
            h, s = layer(h)
            logdet = logdet + s
        return (h, logdet)

    def inverse(self, x):
        h = x
        logdet = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for layer in reversed(self.layers):
            h, s = layer.inverse(h)
            logdet = logdet + s
        return (h, logdet)
