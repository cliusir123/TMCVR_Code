import json
from pathlib import Path
import numpy as np
import torch
from .backend import TORCH, jax_backend
from .base import TargetDistribution
REFERENCE_DIR = Path(__file__).resolve().parent.parent / 'results' / 'references'
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

class Posterior(TargetDistribution):
    name = 'posterior'
    names = []
    tex = None
    reference_method = 'nuts'

    def __init__(self, dim):
        super().__init__(dim)
        self._laplace = None
        self._consts = {}

    def logp(self, x, B):
        raise NotImplementedError

    def features(self, x, B):
        raise NotImplementedError

    def c(self, key, arr, B, like):
        if B is TORCH:
            k = (key, like.device, like.dtype)
            if k not in self._consts:
                self._consts[k] = B.const(arr, like)
            return self._consts[k]
        return B.const(arr)

    def log_prob(self, x):
        return self.logp(x, TORCH)

    def log_prob_twice_diff(self, x):
        return self.log_prob(x)

    def logp_and_grad(self, x):
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            lp = self.log_prob(xx)
            g, = torch.autograd.grad(lp.sum(), xx)
        return (lp.detach(), g)

    def grad_log_prob(self, x):
        return self.logp_and_grad(x)[1]

    def F(self, x):
        return self.features(x, TORCH)

    @property
    def test_functions(self):
        truth = self._reference_means()
        return [(n, lambda x, j=j: self.F(x)[..., j], None if truth is None else truth[j]) for j, n in enumerate(self.names)]

    def tail_region(self, x):
        return torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)

    def jax_logp(self, x):
        return self.logp(x, jax_backend())

    def reference_logp(self, x, B):
        return self.logp(x, B)

    def reference_to_x(self, xr):
        return xr

    def reference_init(self):
        return self.laplace_np()[0]

    def init_point(self):
        return np.zeros(self.dim)

    def init_scale(self):
        return None

    def laplace(self, max_iter=2000):
        if self._laplace is None:
            dev = torch.get_default_device()
            dt = torch.get_default_dtype()
            s0 = self.init_scale()
            x0 = torch.as_tensor(self.init_point(), dtype=dt, device=dev)
            if s0 is not None:
                self._laplace = (x0, torch.as_tensor(s0, dtype=dt, device=dev))
                return self._laplace
            x = x0.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([x], lr=1.0, max_iter=max_iter, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                loss = -self.log_prob(x)
                loss.backward()
                return loss
            opt.step(closure)
            m = x.detach()
            H = torch.autograd.functional.hessian(self.log_prob_twice_diff, m)
            try:
                cov = torch.linalg.inv(-H)
                s = torch.sqrt(torch.diagonal(cov))
                ok = bool(torch.isfinite(s).all()) and bool((torch.linalg.eigvalsh(-H) > 0).all())
            except RuntimeError:
                ok = False
            if not ok:
                raise RuntimeError(f'{self.name}: Laplace approximation failed (non-negative-definite Hessian); override init_scale')
            self._laplace = (m, s)
        return self._laplace

    def gaussian_init(self):
        return self.laplace()

    def laplace_np(self):
        m, s = self.laplace()
        return (m.cpu().numpy(), s.cpu().numpy())

    @property
    def reference_path(self):
        return REFERENCE_DIR / f'reference_{self.name}.json'

    def _reference_means(self):
        p = self.reference_path
        if not p.exists():
            return None
        return json.loads(p.read_text())['mean']

    def analytic_truth(self):
        return {'mean': [None] * len(self.names)}

    def ground_truth(self):
        p = self.reference_path
        if not p.exists():
            raise FileNotFoundError(f'no reference values for {self.name}: run python workflow/reference.py --target {self.name}')
        ref = json.loads(p.read_text())
        mean = list(ref['mean'])
        for j, v in enumerate(self.analytic_truth()['mean']):
            if v is not None:
                mean[j] = v
        return {'mean': mean, 'mean_ref': ref['mean'], 'var': ref['var'], 'mcse': ref['mcse'], 'p_tail': 0.0}
