import math
import time
import torch
from transports.ensemble import EnsembleFlow, log_erfc

def log_weights(target, flow, zeta):
    x, ld = flow(zeta)
    d = zeta.shape[-1]
    return target.log_prob(x) + ld + 0.5 * (zeta ** 2).sum(-1) + 0.5 * d * math.log(2 * math.pi)

@torch.no_grad()
def flow_diagnostics(target, flow, n=20000):
    lw = log_weights(target, flow, torch.randn(flow.E, n, flow.dim))
    lw = torch.where(torch.isfinite(lw), lw, torch.full_like(lw, -1e+300))
    ess = torch.exp(2 * torch.logsumexp(lw, 1) - torch.logsumexp(2 * lw, 1)) / n
    gap = lw.max(1).values - torch.quantile(lw, 0.999, dim=1)
    return {'ess_is': ess.tolist(), 'logw_max_minus_q999': gap.tolist()}

def clip_per_replicate(flow, clip):
    params = [p for p in flow.parameters() if p.grad is not None]
    E = flow.E
    bad = torch.zeros(E, dtype=torch.bool, device=params[0].device)
    sq = torch.zeros(E, dtype=params[0].dtype, device=params[0].device)
    for p in params:
        g = p.grad.reshape(E, -1)
        bad |= ~torch.isfinite(g).all(1)
        sq += torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).pow(2).sum(1)
    scale = torch.where(bad, torch.zeros_like(sq), torch.clamp(clip / (sq.sqrt() + 1e-12), max=1.0))
    for p in params:
        g = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        p.grad.copy_(g * scale.view(E, *[1] * (g.dim() - 1)))

def _cosine(opt, steps, lr, ratio):
    for g in opt.param_groups:
        g['lr'] = lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * ratio)

def tail_ramp(tail, it, steps):
    if tail is None:
        return 0.0
    s0, r = (tail.get('start', 0.0) * steps, tail.get('ramp', 0.0) * steps)
    if it < s0:
        return 0.0
    return 1.0 if r <= 0 else min(1.0, (it - s0 + 1) / r)

def tail_penalty(target, flow, lw_ref, tail, gen=None):
    E, d = (flow.E, flow.dim)
    if tail.get('band'):
        lo, hi = tail['band']
        u = torch.randn(E, tail['batch'], d, generator=gen)
        u = u / u.norm(dim=-1, keepdim=True)
        r = d ** 0.5 + lo + (hi - lo) * torch.rand(E, tail['batch'], 1, generator=gen)
        zeta = r * u
    else:
        zeta = tail['scale'] * torch.randn(E, tail['batch'], d, generator=gen)
    lw = log_weights(target, flow, zeta)
    lw = torch.where(torch.isfinite(lw), lw, lw_ref[:, None] + tail['margin'])
    over = torch.relu(lw - lw_ref.detach()[:, None] - tail['margin'])
    return tail['weight'] * (over * over).mean(1)

def train_reverse_kl(flow, target, steps=2000, batch=64, lr=0.001, lr_min_ratio=0.01, callback=None, clip=100.0, tail=None):
    flow.requires_grad_(True)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = _cosine(opt, steps, lr, lr_min_ratio)
    t0 = time.perf_counter()
    nt_used = 0
    for it in range(steps):
        lw = log_weights(target, flow, torch.randn(flow.E, batch, flow.dim))
        lw = torch.where(torch.isfinite(lw), lw, torch.zeros_like(lw))
        loss = -lw.mean(1)
        f = tail_ramp(tail, it, steps)
        if f > 0:
            loss = loss + f * tail_penalty(target, flow, lw.detach().mean(1), tail)
            nt_used += tail['batch']
        loss = loss.sum()
        opt.zero_grad()
        loss.backward()
        if clip:
            clip_per_replicate(flow, clip)
        opt.step()
        sched.step()
        if callback is not None:
            callback(it + 1, flow)
    flow.requires_grad_(False)
    return {'n_logp': steps * batch + nt_used, 'n_grad': steps * batch + nt_used, 'seconds': time.perf_counter() - t0}

class WhitenedTarget:

    def __init__(self, target, L):
        self.target, self.L, self.dim = (target, L, target.dim)
        self.E = L.shape[0] if L.dim() == 3 else None
        self.ld = torch.log(torch.diagonal(L, dim1=-2, dim2=-1).abs()).sum(-1)

    def log_prob(self, y):
        from transports.linear import add_per_replicate, apply_per_replicate
        x = apply_per_replicate(self.L, y, self.E)
        return add_per_replicate(self.target.log_prob(x), self.ld, self.E)

def train_combined(flow, target, steps=10000, batch=256, lr=0.001, lr_min_ratio=0.01, fkl_weight=1.0, rkl_weight=1.0, fkl_batch=256, pilot=None, chains=32, refresh=10, chain_steps=10, gamma=0.5, buffer_size=20000, fkl_start=None, seed=0, clip=100.0, callback=None, adaptive=False, pilot_steps=500, chain_kind='fixed', acc_target=0.5, adapt_lr=0.2, gamma_local=0.05, flush=False, min_fill=1000, anchor=0.0, sym_chain=False, rkl_after=None, tail=None):
    E, d = (flow.E, flow.dim)
    if fkl_start is None:
        fkl_start = 0.0 if pilot is not None else 0.1
    fixed_pilot = pilot is not None and (not adaptive)
    use_chain = pilot is None or adaptive
    t_switch = pilot_steps if pilot is not None and adaptive else int(fkl_start * steps)
    dev = torch.get_default_device()
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ar = torch.arange(E, device=dev)
    n_logp = n_grad = 0.0
    if use_chain:
        cbuf = torch.zeros(E, buffer_size, d)
        cfill = torch.zeros(E, dtype=torch.long)
        head = 0
        with torch.no_grad():
            if pilot is not None:
                pick = (torch.rand(E, chains, generator=gen) * pilot.shape[1]).long()
                xc = pilot[ar[:, None], pick].clone()
            else:
                xc, _ = flow(torch.randn(E, chains, d, generator=gen))
    gam = torch.full((E,), float(gamma))
    acc_last = torch.zeros(E)

    def log_w(z):
        x, ld = flow(z)
        return (target.log_prob(x) + ld + 0.5 * (z ** 2).sum(-1), x)
    flow.requires_grad_(True)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    sched = _cosine(opt, steps, lr, lr_min_ratio)
    t0 = time.perf_counter()
    for it in range(steps):
        if use_chain and it % refresh == 0:
            flow.requires_grad_(False)
            keep = not flush or it >= t_switch
            with torch.no_grad():
                z, _ = flow.inverse(xc)
                lw, x = log_w(z)
                lw = torch.where(torch.isfinite(lw), lw, torch.full_like(lw, -1e+300))
                acc = torch.zeros(E)
                for _ in range(chain_steps):
                    if chain_kind == 'mixed':
                        g = torch.where(torch.rand(E, chains, generator=gen) < 0.5, gamma_local, 1.0)
                    else:
                        g = gam[:, None].expand(E, chains)
                    beta = (1.0 - g)[..., None]
                    sig = torch.sqrt(1.0 - beta * beta)
                    zp = beta * z + sig * torch.randn(z.shape, generator=gen)
                    lwp, xp = log_w(zp)
                    ok = (torch.log(torch.rand(lw.shape, generator=gen)) < lwp - lw) & torch.isfinite(lwp)
                    acc += ok.double().mean(1)
                    z = torch.where(ok[..., None], zp, z)
                    x = torch.where(ok[..., None], xp, x)
                    lw = torch.where(ok, lwp, lw)
                    if keep:
                        for xx in (x, -x) if sym_chain else (x,):
                            idx = (head + torch.arange(chains)) % buffer_size
                            cbuf[:, idx] = xx
                            head = (head + chains) % buffer_size
                            cfill = torch.clamp(cfill + chains, max=buffer_size)
                xc = x
                acc_last = acc / chain_steps
                if chain_kind == 'gti':
                    gam = torch.clamp(gam * torch.exp(adapt_lr * (acc_last - acc_target)), 0.001, 1.0)
            n_logp += chains * (chain_steps + 1)
            flow.requires_grad_(True)
        rw = rkl_weight
        if rkl_after is not None and use_chain and (it >= t_switch):
            rw = rkl_after
        if rw > 0:
            lwr = log_weights(target, flow, torch.randn(E, batch, d, generator=gen))
            lwr = torch.where(torch.isfinite(lwr), lwr, torch.zeros_like(lwr))
            loss = -rw * lwr.mean(1)
            n_logp += batch
            n_grad += batch
            f = tail_ramp(tail, it, steps)
            if f > 0:
                loss = loss + f * tail_penalty(target, flow, lwr.detach().mean(1), tail, gen)
                n_logp += tail['batch']
                n_grad += tail['batch']
        else:
            loss = torch.zeros(E)
        src = None
        if fkl_weight > 0:
            if fixed_pilot:
                if it >= fkl_start * steps:
                    src, fill = (pilot, torch.full((E,), pilot.shape[1], dtype=torch.long))
            elif it >= t_switch and int(cfill.min()) >= min(min_fill, buffer_size):
                src, fill = (cbuf, cfill)
            elif pilot is not None:
                src, fill = (pilot, torch.full((E,), pilot.shape[1], dtype=torch.long))
            elif it >= t_switch and int(cfill.min()) > 0 and (not flush):
                src, fill = (cbuf, cfill)
        if src is not None:
            u = torch.rand(E, fkl_batch, generator=gen)
            xb = src[ar[:, None], (u * fill[:, None]).long()]
            if anchor > 0 and pilot is not None and (src is not pilot):
                na = int(round(anchor * fkl_batch))
                ua = torch.rand(E, na, generator=gen)
                xb = torch.cat([xb[:, :fkl_batch - na], pilot[ar[:, None], (ua * pilot.shape[1]).long()]], 1)
            lq = flow.log_prob(xb)
            lq = torch.where(torch.isfinite(lq), lq, torch.zeros_like(lq))
            loss = loss - fkl_weight * lq.mean(1)
        opt.zero_grad()
        loss.sum().backward()
        if clip:
            clip_per_replicate(flow, clip)
        opt.step()
        sched.step()
        if callback is not None:
            callback(it + 1, flow)
    flow.requires_grad_(False)
    return {'n_logp': n_logp, 'n_grad': n_grad, 'seconds': time.perf_counter() - t0, 'chain_gamma': gam.tolist(), 'chain_acceptance': acc_last.tolist()}
