import torch
LOG_R_FLOOR = -50.0

def log_t(z, df):
    return -0.5 * (df + z.shape[-1]) * torch.log1p((z * z).sum(-1) / df)

def make_generator(seed):
    gen = torch.Generator(device=torch.get_default_device())
    gen.manual_seed(int(seed))
    return gen

def run_chains(latent, proposal, z0, gamma, n, burn=0, adapter=None, recorder_factory=None, generator=None, thin=None, per_chain=False, indep=None):
    R, d = z0.shape
    dtype, device = (z0.dtype, z0.device)
    grid = torch.is_tensor(gamma) and gamma.dim() == 1 and (not per_chain)
    if grid and adapter is not None:
        raise ValueError('gamma adaptation needs a scalar gamma')
    G = gamma.numel() if grid else 1
    B = G * R
    order = proposal.order
    s = latent.evaluate(z0.repeat(G, 1), order)
    if per_chain:
        gvec = gamma.to(dtype).clone()
        gamma_now = adapter.init(gvec) if adapter is not None else None
    else:
        gamma_now = float(gamma) if not grid else None
        gvec = gamma.to(dtype).repeat_interleave(R) if grid else torch.full((B,), float(gamma), dtype=dtype, device=device)

    def set_gamma(g):
        if per_chain:
            gvec.copy_(adapter.expand(g))
        else:
            gvec.fill_(g)
    if adapter is not None and hasattr(adapter, 'n_chains'):
        adapter.n_chains = B
    stats = {k: torch.zeros(B, dtype=dtype, device=device) for k in ('acc', 'esjd', 'min0', 'log_a')}
    recorder, states = (None, [])
    zeros = lambda: torch.zeros(B, dtype=dtype, device=device)
    run, max_rej = (zeros(), zeros())
    delta = float(indep['delta']) if indep else 0.0
    use_ind = delta > 0
    if use_ind:
        df = int(indep['df'])
        ist = {k: zeros() for k in ('n_tgi', 'n_ind', 'acc_ind', 'n_ind_all', 'n_ind_acc_all', 'n_tgi_burn')}
    bm = None
    for t in range(burn + n):
        eps = torch.randn(R, d, generator=generator, dtype=dtype, device=device).repeat(G, 1)
        u = torch.rand(R, generator=generator, dtype=dtype, device=device).repeat(G)
        if use_ind:
            bm = (torch.rand(R, generator=generator, dtype=dtype, device=device) < delta).repeat(G)
            gz = torch.randn(R, d, generator=generator, dtype=dtype, device=device)
            chi2 = (torch.randn(R, df, generator=generator, dtype=dtype, device=device) ** 2).sum(-1)
            zi = (gz / torch.sqrt(chi2 / df).unsqueeze(-1)).repeat(G, 1)
        mkw = {} if bm is None else {'mask': ~bm}
        if t == burn:
            if adapter is not None and burn > 0:
                gamma_now = adapter.final(gamma_now)
                set_gamma(gamma_now)
            if recorder_factory is not None:
                recorder = recorder_factory(gvec.clone())
        diff = adapter is not None and getattr(adapter, 'differentiable', True)
        if t < burn and diff:
            cache = {}

            def ratio_fn(g, s=s, eps=eps):
                gv = adapter.expand(g) if per_chain else g
                m, v, tau = proposal.moments(s, gv)
                zp = m + v.sqrt() * eps
                sp = latent.evaluate(zp, order, differentiable=True)
                mp, vp, _ = proposal.moments(sp, gv)
                lr = sp.lp - s.lp + proposal.log_q(s.z, mp, vp) - proposal.log_q(zp, m, v)
                cache.update(zp=zp.detach(), sp=sp.detach(), log_r=lr.detach())
                return lr
            new_gamma = adapter.update(gamma_now, ratio_fn, **mkw)
            zp, sp, log_r = (cache['zp'], cache['sp'], cache['log_r'])
            m = tau = None
            if bm is not None and bool(bm.any()):
                si = latent.evaluate(zi, order)
                sp = sp.where(bm, si)
                zp = torch.where(bm.unsqueeze(-1), zi, zp)
                log_r = torch.where(bm, si.lp - s.lp + log_t(s.z, df) - log_t(zi, df), log_r)
        else:
            new_gamma = None
            m, v, tau = proposal.moments(s, gvec)
            zp = m + v.sqrt() * eps
            if bm is not None:
                zp = torch.where(bm.unsqueeze(-1), zi, zp)
            sp = latent.evaluate(zp, order)
            mp, vp, _ = proposal.moments(sp, gvec)
            log_r = sp.lp - s.lp + proposal.log_q(s.z, mp, vp) - proposal.log_q(zp, m, v)
            if bm is not None:
                log_r = torch.where(bm, sp.lp - s.lp + log_t(s.z, df) - log_t(zp, df), log_r)
        bad = ~torch.isfinite(log_r)
        log_r = torch.where(bad, torch.full_like(log_r, -float('inf')), log_r)
        alpha = torch.exp(log_r.clamp(max=0.0))
        accept = u < alpha
        if t < burn and adapter is not None and (not diff):
            new_gamma = adapter.observe(gamma_now, alpha, **mkw)
        if bm is not None:
            ist['n_ind_all'] += bm
            ist['n_ind_acc_all'] += bm & accept
            if t < burn:
                ist['n_tgi_burn'] += ~bm
        if t >= burn:
            w = None if bm is None else (~bm).to(dtype)
            if recorder is not None:
                recorder.record(s, sp, m, tau, alpha, **{} if w is None else {'mask': w})
            ww = 1.0 if w is None else w
            stats['acc'] += alpha * ww
            stats['esjd'] += alpha * ((zp - s.z) ** 2).sum(-1) * ww
            stats['min0'] += log_r.clamp(max=0.0).clamp_min(LOG_R_FLOOR) * ww
            if s.a is not None:
                stats['log_a'] += torch.log(s.a).mean(-1) * ww
            rej = ~accept if bm is None else ~accept & ~bm
            run = torch.where(accept, torch.zeros_like(run), run + rej.to(dtype))
            max_rej = torch.maximum(max_rej, run)
            if bm is not None:
                ist['n_tgi'] += w
                ist['n_ind'] += bm
                ist['acc_ind'] += alpha * bm
            if thin and (t - burn) % thin == 0:
                states.append(s.z.clone())
        s = s.where(accept, sp)
        if new_gamma is not None:
            gamma_now = new_gamma
            set_gamma(gamma_now)
    if n == 0 and adapter is not None and (burn > 0):
        gamma_now = adapter.final(gamma_now)
        set_gamma(gamma_now)
    if use_ind:
        out = {k: v / ist['n_tgi'].clamp_min(1) for k, v in stats.items()}
        out['indep'] = {'n_ind': ist['n_ind'], 'acc_ind': ist['acc_ind'] / ist['n_ind'].clamp_min(1), 'n_tgi': ist['n_tgi'], 'n_ind_all': ist['n_ind_all'], 'n_ind_acc_all': ist['n_ind_acc_all'], 'n_tgi_burn': ist['n_tgi_burn'], 'delta': delta, 'df': df}
    else:
        out = {k: v / max(n, 1) for k, v in stats.items()}
    out['max_rej'] = max_rej
    out.update(gamma=gvec, z=s.z, recorder=recorder, G=G, R=R)
    if thin:
        out['states'] = torch.stack(states) if states else None
    return out
