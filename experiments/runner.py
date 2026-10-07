import json
import time
from pathlib import Path
import torch
from hermite import CVRecorder, fit_dictionary
from mcmc import LatentTarget, make_generator, make_proposal, run_chains
from transports.diagnostics import delta_dispersion, ess_is, fisher, logw_gap

def completed(out_path, label):
    p = Path(out_path)
    if not p.exists():
        return False
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get('kind') == 'transport' and r['label'] == label:
            return True
    return False

def _lists(t):
    return t.detach().cpu().tolist()

class _TrajRecorder:

    def __init__(self, inner, keep):
        self.inner, self.keep = (inner, keep)
        self.z, self.zp = ([], [])

    def record(self, s, sp, m, tau, alpha, **kw):
        self.inner.record(s, sp, m, tau, alpha, **kw)
        self.z.append(s.z[:self.keep].float().cpu())
        self.zp.append(sp.z[:self.keep].float().cpu())

    def results(self):
        return self.inner.results()

    def arrays(self):
        return (torch.stack(self.z, 1).numpy(), torch.stack(self.zp, 1).numpy())

def run_grid(latent, sampler, gammas, z0, n, burn, fit, F_fns, arms, seed, thin=None, traj_keep=None, indep=None):
    holder = {}

    def factory(g):
        rec = CVRecorder(fit, F_fns, arms, g, n_steps=n)
        if traj_keep:
            rec = holder.setdefault('r', _TrajRecorder(rec, traj_keep))
        return rec
    out = run_chains(latent, make_proposal(sampler), z0, torch.tensor(gammas), n=n, burn=burn, generator=make_generator(seed), thin=thin, recorder_factory=factory, indep=indep)
    G, R, nF = (len(gammas), z0.shape[0], len(F_fns))
    res = out['recorder'].results()
    recs = []
    for gi, g in enumerate(gammas):
        rows = slice(gi * R, (gi + 1) * R)
        rec = {'sampler': sampler, 'gamma': float(g)}
        for k in ('acc', 'esjd', 'min0', 'log_a'):
            rec[k] = out[k][rows].mean().item()
        rec['acc_chain'] = _lists(out['acc'][rows])
        rec['max_rej_chain'] = _lists(out['max_rej'][rows])
        if 'indep' in out:
            rec['indep'] = {k: _lists(out['indep'][k][rows]) for k in ('n_ind', 'acc_ind', 'n_tgi')}
        rec['chain'] = {}
        for a, arm in enumerate(arms):
            cols = slice(a * nF, (a + 1) * nF)
            rec['chain'][arm] = {k: _lists(v[rows, cols]) for k, v in res.items()}
        recs.append(rec)
    if traj_keep:
        return (recs, out.get('states'), holder['r'].arrays())
    return (recs, out.get('states'))

def run_transport(target, transport, exact_latent, dictionary, arms, out_path, E, var_pi, means, a_clip, samplers=None, extra=None, verbose=True, gammas=None, traj_keep=None, x0_fn=None, indep=None):
    say = print if verbose else lambda *a, **k: None
    extra = dict(extra or {})
    label = extra['label']
    if completed(out_path, label):
        say(f'      (done)')
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tfs = target.test_functions
    F_fns = [f for _, f, _ in tfs]
    samplers = samplers or E.SAMPLERS
    t0 = time.time()
    fit = fit_dictionary(dictionary, transport, F_fns, arms, E.N_SELECT, E.N_MC, E.N_RESID, names=[n for n, _, _ in tfs])
    latent = LatentTarget(target, transport, a_clip)
    gen = make_generator(E.SEED + 1000)
    if x0_fn is not None:
        with torch.no_grad():
            z0 = transport.inverse(x0_fn(E.CHAINS, gen))[0]
    else:
        z0 = torch.randn(E.CHAINS, target.dim, generator=gen)
    burn = E.FLOW_BURN
    key = {'target': extra.get('target'), 'label': label, 'd': target.dim}
    states = None
    with out_path.open('a') as f:
        for s in samplers:
            keep = s == 'mala' or len(samplers) == 1
            thin = E.FLOW_STATE_THIN if not exact_latent and keep else None
            gs = (gammas or {}).get(s, E.GAMMAS)
            got = run_grid(latent, s, gs, z0, E.N_STEPS, burn, fit, F_fns, list(arms), E.SEED + 2000, thin=thin, traj_keep=traj_keep, indep=indep)
            recs, st = (got[0], got[1])
            if traj_keep:
                import numpy as np
                zz, zp = got[2]
                np.savez_compressed(out_path.parent / f'{out_path.stem}_{label}_{s}_traj.npz', z=zz, zp=zp, gamma=np.array(gs), burn=burn)
            for rec in recs:
                f.write(json.dumps({'kind': 'run', **key, **rec}) + '\n')
            if st is not None:
                g = max(range(len(recs)), key=lambda i: recs[i]['esjd'])
                states = st[:, g * E.CHAINS:(g + 1) * E.CHAINS].reshape(-1, target.dim)
            say(f'      {s:<6} acceptance ' + ' '.join((f"{r['acc']:.2f}" for r in recs)) + f'   [{time.time() - t0:.0f}s]')
        z_diag = transport.sample_latent(E.N_DIAG, generator=gen) if exact_latent else states[-E.N_DIAG:]
        meta = {'kind': 'transport', **key, **extra, 'names': fit.names, 'truths': [t if t is not None else float(means[j]) for j, (_, _, t) in enumerate(tfs)], 'var_pi': _lists(var_pi), 'mean_pi': _lists(means), 'var_rho': _lists(fit.var_rho), 'arms': {a: {'V0': _lists(fit.V0(a)), 'kept': _lists(fit.n_kept(a))} for a in arms}, 'candidates': len(dictionary), 'cost': {s: make_proposal(s).cost(target.dim) for s in samplers}, 'n_steps': E.N_STEPS, 'chains': E.CHAINS, 'burn': burn, 'gammas': {k: list(v) for k, v in gammas.items()} if gammas else list(E.GAMMAS), 'a_clip': list(a_clip), 'n_select': E.N_SELECT, 'n_mc': E.N_MC, 'F': fisher(latent, z_diag), 'ess_is': ess_is(target, transport, E.N_DIAG), 'dispersion': delta_dispersion(latent, z_diag), 'logw_gap': logw_gap(target, transport, E.N_DIAG), 'seconds': time.time() - t0, 'indep': dict(indep) if indep else None}
        f.write(json.dumps(meta) + '\n')
    say(f"      F = {meta['F']:.4g}   ESS_IS/N = {meta['ess_is']:.4f}   dispersion = {meta['dispersion']:.3f}   logw gap = {meta['logw_gap']:.2f}   [{meta['seconds']:.0f}s]")
    return meta
