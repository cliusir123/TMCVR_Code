import abc
import math
import torch
import torch.nn as nn

class Transport(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, z):
        pass

    def inverse(self, x):
        raise NotImplementedError(f'{type(self).__name__} has no inverse')

    def log_prob(self, x):
        z, ld = self.inverse(x)
        d = x.shape[-1]
        return -0.5 * (z ** 2).sum(-1) - 0.5 * d * math.log(2 * math.pi) + ld

    def sample_latent(self, n, generator=None):
        raise NotImplementedError(f'{type(self).__name__} cannot sample pi_z exactly')

    @torch.no_grad()
    def push(self, z, chunk=None):
        d = z.shape[-1]
        chunk = chunk or max(1, 16000000 // max(d, 1))
        flat = z.reshape(-1, d)
        outs = [self.forward(flat[s:s + chunk])[0] for s in range(0, flat.shape[0], chunk)]
        return torch.cat(outs).reshape(z.shape)
