import math
import torch
J_LAMBDA = 1.0
J_ROBUST = 0.1

def j_objective(gamma, mean_min0_logr, acceptance, V0=0.0, mean_log_a=0.0, lam=J_LAMBDA):
    if torch.is_tensor(gamma):
        lg = torch.log(2 * gamma - gamma ** 2)
        return lg + mean_min0_logr + mean_log_a - lam * torch.log1p(V0 * (1 - acceptance))
    return math.log(2 * gamma - gamma ** 2) + mean_min0_logr + mean_log_a - lam * math.log1p(V0 * (1 - acceptance))

class JAdapter:

    def __init__(self, V0=0.0, lr=0.05, bounds=(0.001, 1.0), log_r_floor=-50.0, max_step=1.0, lam=J_LAMBDA, robust=None):
        self.V0, self.lr, self.bounds = (float(V0), lr, bounds)
        self.lam = lam
        self.robust = robust
        self.n_chains = None
        self.log_r_floor = log_r_floor
        self.max_step = max_step
        self.history = []

    def clip(self, gamma):
        lo, hi = self.bounds
        return float(min(max(gamma, lo), hi)) if math.isfinite(gamma) else lo

    def _robust_dg(self, gamma, ratio_fn, mask=None):
        g = torch.full((self.n_chains,), float(gamma), requires_grad=True)
        log_r = ratio_fn(g)
        log_r = torch.where(torch.isfinite(log_r), log_r, torch.full_like(log_r, self.log_r_floor))
        lr0 = log_r.clamp(max=0.0)
        a = torch.exp(lr0)
        du, = torch.autograd.grad(lr0.sum(), g)
        da = a.detach() * du
        q = self.robust
        a = a.detach()
        if mask is not None:
            du, da, a = (du[mask], da[mask], a[mask])

        def wmean(v):
            v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            lo, hi = torch.quantile(v, torch.tensor([q, 1 - q], dtype=v.dtype, device=v.device))
            return float(v.clamp(lo, hi).mean())
        abar = float(a.mean())
        return (2 - 2 * gamma) / (2 * gamma - gamma ** 2) + wmean(du) + self.lam * self.V0 * wmean(da) / (1 + self.V0 * (1 - abar))

    def update(self, gamma, ratio_fn, mask=None):
        if mask is not None and (not bool(mask.any())):
            self.history.append(math.log(gamma))
            return gamma
        if self.robust is not None and self.n_chains:
            step = self.lr * gamma * self._robust_dg(gamma, ratio_fn, mask)
            step = max(-self.max_step, min(self.max_step, step)) if math.isfinite(step) else 0.0
            new = self.clip(math.exp(math.log(gamma) + step))
            self.history.append(math.log(new))
            return new
        g = torch.tensor(gamma, requires_grad=True)
        log_r = ratio_fn(g)
        log_r = torch.where(torch.isfinite(log_r), log_r, torch.full_like(log_r, self.log_r_floor))
        lr0 = log_r.clamp(max=0.0)
        if mask is not None:
            lr0 = lr0[mask]
        J = j_objective(g, lr0.mean(), torch.exp(lr0).mean(), self.V0, lam=self.lam)
        dg, = torch.autograd.grad(J, g)
        step = self.lr * gamma * float(dg)
        step = max(-self.max_step, min(self.max_step, step)) if math.isfinite(step) else 0.0
        new = self.clip(math.exp(math.log(gamma) + step))
        self.history.append(math.log(new))
        return new

    def final(self, gamma):
        if not self.history:
            return gamma
        tail = self.history[len(self.history) // 2:]
        return self.clip(math.exp(sum(tail) / len(tail)))

class GroupJAdapter:

    def __init__(self, groups, V0, lr=0.05, bounds=(0.001, 1.0), log_r_floor=-50.0, max_step=1.0, lam=J_LAMBDA, robust=None):
        self.groups = groups
        self.lam = lam
        self.robust = robust
        self.V0 = torch.as_tensor(V0, dtype=torch.get_default_dtype(), device=groups.device)
        self.G = int(self.V0.numel())
        self.lr, self.bounds, self.log_r_floor = (lr, bounds, log_r_floor)
        self.max_step = max_step
        self.count = torch.bincount(groups, minlength=self.G).to(self.V0.dtype)
        self.history = []

    def init(self, gvec):
        g = torch.zeros(self.G, dtype=gvec.dtype, device=gvec.device)
        return g.index_add_(0, self.groups, gvec) / self.count

    def expand(self, g):
        if g.shape[0] == self.groups.shape[0] and self.G != g.shape[0]:
            return g
        return g[self.groups]

    def _group_mean(self, v, mask=None):
        out = torch.zeros(self.G, dtype=v.dtype, device=v.device)
        if mask is None:
            return out.index_add(0, self.groups, v) / self.count
        w = mask.to(v.dtype)
        cnt = torch.zeros_like(out).index_add(0, self.groups, w)
        return out.index_add(0, self.groups, v * w) / cnt.clamp_min(1)

    def _robust_dg(self, gamma, ratio_fn, mask=None):
        gc = gamma.detach()[self.groups].clone().requires_grad_(True)
        log_r = ratio_fn(gc)
        log_r = torch.where(torch.isfinite(log_r), log_r, torch.full_like(log_r, self.log_r_floor))
        lr0 = log_r.clamp(max=0.0)
        a = torch.exp(lr0)
        du, = torch.autograd.grad(lr0.sum(), gc)
        da = a.detach() * du
        du = torch.nan_to_num(du, nan=0.0, posinf=0.0, neginf=0.0)
        da = torch.nan_to_num(da, nan=0.0, posinf=0.0, neginf=0.0)
        q = self.robust
        qs = torch.tensor([q, 1 - q], dtype=du.dtype, device=du.device)
        dg = torch.zeros_like(gamma)
        for k in range(self.G):
            m = self.groups == k
            if mask is not None:
                m = m & mask
                if not bool(m.any()):
                    continue
            u, v = (du[m], da[m])
            lo, hi = torch.quantile(u, qs)
            mu = u.clamp(lo, hi).mean()
            lo, hi = torch.quantile(v, qs)
            ma = v.clamp(lo, hi).mean()
            ab = a.detach()[m].mean()
            gk = gamma[k]
            dg[k] = (2 - 2 * gk) / (2 * gk - gk ** 2) + mu + self.lam * self.V0[k] * ma / (1 + self.V0[k] * (1 - ab))
        return dg

    def update(self, gamma, ratio_fn, mask=None):
        if self.robust is not None:
            dg = self._robust_dg(gamma, ratio_fn, mask)
            step = (self.lr * gamma * dg).clamp(-self.max_step, self.max_step)
            lo, hi = self.bounds
            new = torch.exp(torch.log(gamma) + step).clamp(lo, hi)
            self.history.append(torch.log(new))
            return new
        g = gamma.detach().clone().requires_grad_(True)
        log_r = ratio_fn(g)
        log_r = torch.where(torch.isfinite(log_r), log_r, torch.full_like(log_r, self.log_r_floor))
        lr0 = log_r.clamp(max=0.0)
        J = j_objective(g, self._group_mean(lr0, mask), self._group_mean(torch.exp(lr0), mask), self.V0, lam=self.lam)
        dg, = torch.autograd.grad(J.sum(), g)
        step = (self.lr * gamma * torch.nan_to_num(dg.detach(), nan=0.0, posinf=0.0, neginf=0.0)).clamp(-self.max_step, self.max_step)
        lo, hi = self.bounds
        new = torch.exp(torch.log(gamma) + step).clamp(lo, hi)
        self.history.append(torch.log(new))
        return new

    def final(self, gamma):
        if not self.history:
            return gamma
        tail = torch.stack(self.history[len(self.history) // 2:])
        lo, hi = self.bounds
        return torch.exp(tail.mean(0)).clamp(lo, hi)

class GroupAccAdapter:
    differentiable = False

    def __init__(self, groups, target, rho=0.05, bounds=(0.0001, 1.99)):
        self.groups, self.target, self.rho, self.bounds = (groups, target, rho, bounds)
        self.G = int(groups.max().item()) + 1
        self.count = torch.bincount(groups, minlength=self.G).to(torch.get_default_dtype())
        self.history = []

    def init(self, gvec):
        g = torch.zeros(self.G, dtype=gvec.dtype, device=gvec.device)
        return g.index_add_(0, self.groups, gvec) / self.count

    def expand(self, g):
        return g[self.groups]

    def observe(self, gamma, alpha, mask=None):
        acc = torch.zeros(self.G, dtype=alpha.dtype, device=alpha.device)
        if mask is None:
            acc = acc.index_add_(0, self.groups, alpha) / self.count
            gap = acc - self.target
        else:
            w = mask.to(alpha.dtype)
            cnt = torch.zeros_like(acc).index_add_(0, self.groups, w)
            acc = acc.index_add_(0, self.groups, alpha * w) / cnt.clamp_min(1)
            gap = torch.where(cnt > 0, acc - self.target, torch.zeros_like(acc))
        lo, hi = self.bounds
        new = torch.exp(torch.log(gamma) + self.rho * gap).clamp(lo, hi)
        self.history.append(torch.log(new))
        return new

    def final(self, gamma):
        if not self.history:
            return gamma
        tail = torch.stack(self.history[len(self.history) // 2:])
        lo, hi = self.bounds
        return torch.exp(tail.mean(0)).clamp(lo, hi)
