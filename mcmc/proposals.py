import torch

def _col(gamma):
    return gamma.unsqueeze(-1) if gamma.dim() else gamma

class Proposal:
    name = None
    order = 0

    def moments(self, s, gamma):
        raise NotImplementedError

    def cost(self, dim):
        raise NotImplementedError

    @staticmethod
    def log_q(y, m, v):
        return -((y - m) ** 2 / (2 * v)).sum(-1) - 0.5 * torch.log(v).expand_as(y).sum(-1)

class GTIRWM(Proposal):
    name, order = ('rwm', 0)

    def moments(self, s, gamma):
        beta = 1.0 - _col(gamma)
        return (beta * s.z, 1.0 - beta ** 2, beta ** 2)

    def cost(self, dim):
        return 1.0

class GTIMALA(Proposal):
    name, order = ('mala', 1)

    def moments(self, s, gamma):
        g = _col(gamma)
        beta = 1.0 - g
        return (s.z + g * s.grad, 1.0 - beta ** 2, beta ** 2)

    def cost(self, dim):
        return 2.0

class GTIpMALA(Proposal):
    name, order = ('pmala', 2)

    def moments(self, s, gamma):
        g = _col(gamma)
        v = (1.0 - (1.0 - g) ** 2) * s.a
        return (s.z + g * s.a * s.grad, v, 1.0 - v)

    def cost(self, dim):
        return 2.0 + 2.0 * dim

class MALA(Proposal):
    name, order = ('mala_plain', 1)

    def moments(self, s, gamma):
        g = _col(gamma)
        return (s.z + g * s.grad, 2.0 * g, None)

    def cost(self, dim):
        return 2.0
PROPOSALS = {'rwm': GTIRWM, 'mala': GTIMALA, 'pmala': GTIpMALA, 'mala_plain': MALA}

def make_proposal(name):
    try:
        return PROPOSALS[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown sampler '{name}' (choose from {sorted(PROPOSALS)})") from None
