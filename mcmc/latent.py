import torch

class State:
    __slots__ = ('z', 'x', 'lp', 'grad', 'a')

    def __init__(self, z, x, lp, grad=None, a=None):
        self.z, self.x, self.lp, self.grad, self.a = (z, x, lp, grad, a)

    def where(self, accept, other):

        def pick(new, old):
            if old is None:
                return None
            mask = accept.view(accept.shape + (1,) * (old.dim() - accept.dim()))
            return torch.where(mask, new, old)
        return State(*(pick(getattr(other, k), getattr(self, k)) for k in self.__slots__))

    def detach(self):
        return State(*(None if getattr(self, k) is None else getattr(self, k).detach() for k in self.__slots__))

    def scatter(self, idx, sub):

        def put(old, new):
            if old is None:
                return None
            out = old.clone()
            out[idx] = new
            return out
        return State(*(put(getattr(self, k), getattr(sub, k)) for k in self.__slots__))

class LatentTarget:

    def __init__(self, target, transport, a_clip=(0.01, 2.0), precond='diag'):
        self.target = target
        self.transport = transport
        self.a_clip = a_clip
        self.precond = precond
        self.dim = target.dim

    def evaluate(self, z, order=1, differentiable=False):
        if not differentiable:
            z = z.detach()
            if order == 0:
                with torch.no_grad():
                    x, ld = self.transport(z)
                    return State(z, x, self.target.log_prob(x) + ld)
            z = z.requires_grad_(True)
        twice = order >= 2 or (differentiable and order >= 1)
        log_prob = getattr(self.target, 'log_prob_twice_diff', self.target.log_prob) if twice else self.target.log_prob
        with torch.enable_grad():
            x, ld = self.transport(z)
            lp = log_prob(x) + ld
            grad = a = None
            if order >= 1:
                grad, = torch.autograd.grad(lp.sum(), z, create_graph=differentiable or order >= 2)
            if order >= 2:
                a = self.local_variance(self.hess_diag(z, grad, create_graph=differentiable))
        s = State(z, x, lp, grad, a)
        return s if differentiable else s.detach()

    @staticmethod
    def hess_diag(z, grad, create_graph=False):
        cols = []
        for j in range(z.shape[-1]):
            h, = torch.autograd.grad(grad[..., j].sum(), z, retain_graph=True, create_graph=create_graph, allow_unused=True)
            cols.append(torch.zeros_like(z[..., j]) if h is None else h[..., j])
        return torch.stack(cols, dim=-1)

    def local_variance(self, hdiag):
        lo, hi = self.a_clip
        k = -hdiag
        if self.precond == 'scalar':
            k = k.mean(-1, keepdim=True).expand_as(k)
        a = torch.where(k > 1.0 / hi, 1.0 / k.clamp_min(1.0 / hi), torch.full_like(k, hi))
        return a.clamp(lo, hi)

    def grad_log_w(self, z):
        return self.evaluate(z, order=1).grad + z
