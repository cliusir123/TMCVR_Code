import math
import time
import torch
from hermite import HermiteDictionary, count_indices
from hermite.coefficients import batch_size
from mcmc import GroupAccAdapter, GroupJAdapter, LatentTarget, make_proposal, run_chains

@torch.no_grad()
def _moments(flow, D, F_fn, n, cols=None, budget=20000000):
    E, d = (flow.E, flow.dim)
    sub = D if cols is None else D.restrict(cols)
    K = len(sub)
    m = max(1, min(n, budget // max(E * max(K, d), 1)))
    acc = None
    done = 0
    while done < n:
        k = min(m, n - done)
        z = torch.randn(E, k, d)
        F = F_fn(flow(z)[0])
        H = sub.columns(sub.he_table(z))
        parts = {'sH': H.sum(1), 'sF': F.sum(1), 'sFH': torch.einsum('ekj,ekf->ejf', H, F)}
        H.mul_(H)
        parts['sH2'] = H.sum(1)
        parts['sFH2'] = torch.einsum('ekj,ekf->ejf', H, F)
        parts['sF2H2'] = torch.einsum('ekj,ekf->ejf', H, F * F)
        acc = parts if acc is None else {q: acc[q] + v for q, v in parts.items()}
        done += k
    Fb = (acc['sF'] / n)[:, None, :]
    c = (acc['sFH'] - Fb * acc['sH'][..., None]) / n
    second = (acc['sF2H2'] - 2 * Fb * acc['sFH2'] + Fb ** 2 * acc['sH2'][..., None]) / n
    var = (second - c ** 2).clamp_min(0.0) * (n / max(n - 1, 1))
    return (c, (var / n).sqrt())

@torch.no_grad()
def _residuals(flow, sub, coef, F_fn, n, budget=20000000):
    E, d = (flow.E, flow.dim)
    K = max(len(sub), 1)
    m = max(1, min(n, budget // max(E * max(K, d), 1)))
    sF = sF2 = sR = sR2 = 0.0
    done = 0
    while done < n:
        k = min(m, n - done)
        z = torch.randn(E, k, d)
        F = F_fn(flow(z)[0])
        G = torch.bmm(sub.columns(sub.he_table(z)), coef) if len(sub) else torch.zeros_like(F)
        r = F - G
        sF, sF2 = (sF + F.sum(1), sF2 + (F * F).sum(1))
        sR, sR2 = (sR + r.sum(1), sR2 + (r * r).sum(1))
        done += k
    var = lambda s1, s2: (s2 - s1 ** 2 / n) / (n - 1)
    return (var(sF, sF2), var(sR, sR2))

def fit_dictionary(flow, target, K=4, p=2, lam='2log', n_select=10000, n_mc=100000, n_resid=100000):
    t0 = time.perf_counter()
    D = HermiteDictionary(target.dim, K, max_support=p)
    lv = 2 * math.log(count_indices(target.dim, K, p)) if lam == '2log' else float(lam)
    c0, se0 = _moments(flow, D, target.F, n_select)
    keep = c0 ** 2 > lv * se0 ** 2
    union = keep.any(0).any(-1)
    sub = D.restrict(union)
    c, _ = _moments(flow, D, target.F, n_mc, cols=union)
    coef = c * keep[:, union, :]
    var_rho, R = _residuals(flow, sub, coef, target.F, n_resid)
    return {'sub': sub, 'coef': coef, 'kept': keep.sum(1), 'V0': var_rho / R.clamp_min(1e-300), 'var_rho': var_rho, 'R': R, 'n_full': len(D), 'seconds': time.perf_counter() - t0}

class EnsembleRecorder:

    def __init__(self, target, sub, coef, gamma, E, C, n, n_batches=None):
        self.target, self.sub, self.coef = (target, sub, coef)
        self.E, self.C, self.n = (E, C, n)
        self.a = n_batches or max(int(math.isqrt(n)), 1)
        self.b = max(n // self.a, 1)
        B, nF = (E * C, coef.shape[-1])
        deg = sub.degrees.long()
        self.deg_idx = [(k, (deg == k).nonzero().flatten()) for k in range(1, sub.max_degree + 1) if bool((deg == k).any())]
        beta = (1.0 - gamma).reshape(E, C, 1)
        ks = torch.arange(1, sub.max_degree + 1, dtype=gamma.dtype)
        self.W = 1.0 / (1.0 - beta ** ks)
        z = lambda *s: torch.zeros(*s)
        self.batch = {k: z(self.a, B, nF) for k in ('F', '1', '2')}
        self.S = {k: z(B, nF) for k in ('F', '1', '2', 'FF', '11', '22', 'F1', 'F2', '12')}
        self.t = 0
        self.cnt = self.bcnt = None

    def _G(self, tab):
        out = torch.zeros(self.E, self.C, self.coef.shape[-1])
        if len(self.sub):
            phi = self.sub.columns(tab).reshape(self.E, self.C, -1)
            for k, idx in self.deg_idx:
                out += self.W[..., k - 1:k] * torch.bmm(phi[..., idx], self.coef[:, idx])
        return out.reshape(self.E * self.C, -1)

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        D = self.sub
        Gz, Gp = (self._G(D.he_table(s.z)), self._G(D.he_table(sp.z)))
        EG = self._G(D.cond_table(m, tau))
        F = self.target.F(s.x)
        H1 = alpha[:, None] * (Gp - Gz)
        H2 = Gp - EG
        if mask is not None:
            w = mask[:, None]
            F, H1, H2 = (F * w, H1 * w, H2 * w)
            if self.cnt is None:
                self.cnt = torch.zeros_like(mask)
                self.bcnt = torch.zeros(self.a, mask.shape[0], dtype=mask.dtype)
            self.cnt += mask
        i = min(self.t // self.b, self.a - 1)
        if self.t < self.a * self.b:
            self.batch['F'][i] += F
            self.batch['1'][i] += H1
            self.batch['2'][i] += H2
            if self.bcnt is not None:
                self.bcnt[i] += mask
        S = self.S
        for k, v in (('F', F), ('1', H1), ('2', H2), ('FF', F * F), ('11', H1 * H1), ('22', H2 * H2), ('F1', F * H1), ('F2', F * H2), ('12', H1 * H2)):
            S[k] += v
        self.t += 1

    def results(self):
        E, C, n = (self.E, self.C, self.t)
        tot = C * n if self.cnt is None else self.cnt.reshape(E, C).sum(1).clamp_min(1)[:, None]
        pool = {k: v.reshape(E, C, -1).sum(1) / tot for k, v in self.S.items()}
        mF, m1, m2 = (pool['F'], pool['1'], pool['2'])
        k11 = pool['11'] - m1 * m1
        k22 = pool['22'] - m2 * m2
        k12 = pool['12'] - m1 * m2
        q1, q2 = (pool['F1'] - mF * m1, pool['F2'] - mF * m2)
        ridge = 1e-12 * (k11 + k22) + 1e-300
        k11, k22 = (k11 + ridge, k22 + ridge)
        det = k11 * k22 - k12 ** 2
        ok = det > 0
        safe = torch.where(ok, det, torch.ones_like(det))
        th1 = torch.where(ok, -(k22 * q1 - k12 * q2) / safe, torch.zeros_like(det))
        th2 = torch.where(ok, -(k11 * q2 - k12 * q1) / safe, torch.zeros_like(det))
        mu = mF + th1 * m1 + th2 * m2
        mu_fixed = mF + m1 - m2
        bat = self.batch
        if self.bcnt is not None:
            sc = self.b / self.bcnt.clamp_min(1)[..., None]
            bat = {k: v * sc for k, v in bat.items()}
        bF = bat['F'].reshape(self.a, E, C, -1)
        b1 = bat['1'].reshape(self.a, E, C, -1)
        b2 = bat['2'].reshape(self.a, E, C, -1)
        bS = bF + th1[None, :, None] * b1 + th2[None, :, None] * b2
        bX = bF + b1 - b2
        a_eff = min(self.a, n // self.b)
        nc = n if self.cnt is None else self.cnt.clamp_min(1)[:, None]
        var = lambda x: self.b * (x[:a_eff] / self.b).var(0, unbiased=True) if a_eff >= 2 else torch.full(x.shape[1:], float('nan'))
        return {'theta1': th1, 'theta2': th2, 'plain': mF, 'cv': mu, 'cv_fixed': mu_fixed, 'bm_plain': var(bF), 'bm_cv': var(bS), 'bm_fixed': var(bX), 'n': n, 'chain_plain': (self.S['F'] / nc).reshape(E, C, -1), 'chain_fixed': ((self.S['F'] + self.S['1'] - self.S['2']) / nc).reshape(E, C, -1)}

class LeanRecorder(EnsembleRecorder):

    def __init__(self, target, sub, coef, gamma, E, C, n, n_batches=None):
        super().__init__(target, sub, coef, gamma, E, C, n, n_batches=1)
        self.batch = {}
        self.S = {k: v for k, v in self.S.items() if k in ('F', '1', '2')}

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        D = self.sub
        Gz, Gp = (self._G(D.he_table(s.z)), self._G(D.he_table(sp.z)))
        EG = self._G(D.cond_table(m, tau))
        F = self.target.F(s.x)
        H1 = alpha[:, None] * (Gp - Gz)
        H2 = Gp - EG
        if mask is not None:
            w = mask[:, None]
            F, H1, H2 = (F * w, H1 * w, H2 * w)
            if self.cnt is None:
                self.cnt = torch.zeros_like(mask)
            self.cnt += mask
        self.S['F'] += F
        self.S['1'] += H1
        self.S['2'] += H2
        self.t += 1

    def results(self):
        E, C, n = (self.E, self.C, self.t)
        tot = C * n if self.cnt is None else self.cnt.reshape(E, C).sum(1).clamp_min(1)[:, None]
        pool = {k: v.reshape(E, C, -1).sum(1) / tot for k, v in self.S.items()}
        fixed = pool['F'] + pool['1'] - pool['2']
        nc = n if self.cnt is None else self.cnt.clamp_min(1)[:, None]
        zero = torch.zeros_like(fixed)
        return {'theta1': zero, 'theta2': zero, 'plain': pool['F'], 'cv': fixed, 'cv_fixed': fixed, 'n': n, 'chain_plain': (self.S['F'] / nc).reshape(E, C, -1), 'chain_fixed': ((self.S['F'] + self.S['1'] - self.S['2']) / nc).reshape(E, C, -1)}

class RegRecorder(LeanRecorder):

    def __init__(self, target, sub, coef, gamma, E, C, n, n_batches=None, ridge=1e-10):
        super().__init__(target, sub, coef, gamma, E, C, n, n_batches)
        K, nF = (len(sub), coef.shape[-1])
        p = 2 * K
        self.p, self.ridge = (p, ridge)
        self.R_X, self.R_XX = (torch.zeros(E, p), torch.zeros(E, p, p))
        self.R_XF, self.R_F = (torch.zeros(E, p, nF), torch.zeros(E, nF))
        self.c_X = torch.zeros(E * C, p)

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        assert mask is None, 'RegRecorder: no independence steps'
        D = self.sub
        tz, tp, tq = (D.he_table(s.z), D.he_table(sp.z), D.cond_table(m, tau))
        Gz, Gp, EG = (self._G(tz), self._G(tp), self._G(tq))
        F = self.target.F(s.x)
        self.S['F'] += F
        self.S['1'] += alpha[:, None] * (Gp - Gz)
        self.S['2'] += Gp - EG
        self.t += 1
        Pz, Pp, Pq = (D.columns(tz), D.columns(tp), D.columns(tq))
        X = torch.cat([alpha[:, None] * (Pp - Pz), Pp - Pq], dim=-1)
        E, C = (self.E, self.C)
        Xr, Fr = (X.view(E, C, self.p), F.view(E, C, -1))
        self.R_X += Xr.sum(1)
        self.R_F += Fr.sum(1)
        self.R_XX += torch.bmm(Xr.transpose(1, 2), Xr)
        self.R_XF += torch.bmm(Xr.transpose(1, 2), Fr)
        self.c_X += X

    def results(self):
        out = super().results()
        E, C, n = (self.E, self.C, self.t)
        tot = C * n
        mX, mF = (self.R_X / tot, self.R_F / tot)
        cXX = self.R_XX / tot - mX[:, :, None] * mX[:, None, :]
        cXF = self.R_XF / tot - mX[:, :, None] * mF[:, None, :]
        sc = cXX.diagonal(dim1=1, dim2=2).mean(-1).clamp_min(1e-300)
        beta = torch.linalg.lstsq(cXX + self.ridge * sc[:, None, None] * torch.eye(self.p), cXF).solution
        out['cv_reg'] = mF - (mX[:, :, None] * beta).sum(1)
        cF = self.S['F'].view(E, C, -1)
        cX = self.c_X.view(E, C, self.p)
        out['chain_reg'] = (cF - torch.bmm(cX, beta)) / n
        return out

class SnapshotRecorder:

    def __init__(self, inner, steps):
        self.inner, self.steps, self.snaps = (inner, set(steps), {})

    def record(self, *args, **kw):
        self.inner.record(*args, **kw)
        if self.inner.t in self.steps:
            self.snaps[self.inner.t] = self.inner.results()

    def results(self):
        return self.inner.results()

class PlainRecorder:

    def __init__(self, target, E, C):
        self.target, self.E, self.C, self.S, self.t = (target, E, C, 0.0, 0)
        self.cnt = None

    @torch.no_grad()
    def record(self, s, sp, m, tau, alpha, mask=None):
        F = self.target.F(s.x)
        if mask is not None:
            F = F * mask[:, None]
            self.cnt = mask.clone() if self.cnt is None else self.cnt + mask
        self.S = self.S + F
        self.t += 1

    def results(self):
        E, C, n = (self.E, self.C, self.t)
        nc = n if self.cnt is None else self.cnt.clamp_min(1)[:, None]
        if self.cnt is not None:
            tot = self.cnt.reshape(E, C).sum(1).clamp_min(1)[:, None]
            return {'plain': self.S.reshape(E, C, -1).sum(1) / tot, 'chain_plain': (self.S / nc).reshape(E, C, -1), 'n': n}
        chain = (self.S / n).reshape(E, C, -1)
        return {'plain': chain.mean(1), 'chain_plain': chain, 'n': n}

def run_gti(target, flow, dictfit, sampler, budget_gti, C, var_pi=None, gamma0=0.3, j_lr=0.05, burn_frac=0.2, burn_max=1000, headline=0, a_clip=(0.01, 2.0), generator=None, j_robust=None, adapt='J', gamma_bounds=None, precond='diag', snapshots=None, indep=None, lean=False, j_lam=None, cv_reg=False):
    E, d = (flow.E, flow.dim)
    prop = make_proposal(sampler)
    use_j = adapt == 'J'
    ce = prop.cost(d)
    cb = ce + (1.0 if use_j else 0.0)
    per_chain = budget_gti / C
    n_burn = int(min(burn_max, math.floor(burn_frac * per_chain / cb)))
    n_eval = int((per_chain - n_burn * cb) // ce)
    if n_eval < 4:
        return None
    B = E * C
    groups = torch.arange(B) // C
    latent = LatentTarget(target, flow, a_clip, precond)
    adapter = None
    if n_burn > 0 and use_j:
        adapter = GroupJAdapter(groups, dictfit['V0'][:, headline], j_lr, robust=j_robust, **{'bounds': gamma_bounds} if gamma_bounds else {}, **{'lam': j_lam} if j_lam is not None else {})
    elif n_burn > 0 and adapt != 'fixed':
        adapter = GroupAccAdapter(groups, adapt[1], **{'bounds': gamma_bounds} if gamma_bounds else {})
    z0 = torch.randn(B, d, generator=generator)
    base = (lambda g: PlainRecorder(target, E, C)) if dictfit is None else lambda g: (RegRecorder if cv_reg else LeanRecorder if lean else EnsembleRecorder)(target, dictfit['sub'], dictfit['coef'], g, E, C, n_eval)
    snap_steps = {}
    for N in snapshots or ():
        k = int((N / C - n_burn * cb) // ce)
        if 4 <= k <= n_eval:
            snap_steps[k] = N
    factory = (lambda g: SnapshotRecorder(base(g), snap_steps)) if snap_steps else base
    t0 = time.perf_counter()
    g0 = torch.as_tensor(gamma0, dtype=torch.get_default_dtype())
    gvec0 = g0.expand(B).clone() if g0.numel() == 1 else g0.reshape(-1)[groups]
    out = run_chains(latent, prop, z0, gvec0, n=n_eval, burn=n_burn, adapter=adapter, per_chain=True, generator=generator, recorder_factory=factory, indep=indep)
    secs = time.perf_counter() - t0
    res = out['recorder'].results()
    rec = {'plain': res['plain'], 'chain_plain': res['chain_plain'], 'gamma': out['gamma'].reshape(E, C)[:, 0], 'acceptance': out['acc'].reshape(E, C).mean(1), 'n_burn': n_burn, 'n_eval': n_eval, 'C': C, 'cost': C * (n_burn * cb + n_eval * ce), 'seconds': secs, 'cost_eff': C * n_eval * ce}
    if 'indep' in out:
        st = out['indep']
        extra = 0.0 if prop.order == 0 else ce
        n_tgi = n_burn + n_eval - st['n_ind_all']
        per = n_tgi * ce + st['n_tgi_burn'] * (cb - ce) + st['n_ind_all'] + st['n_ind_acc_all'] * extra
        rec['cost'] = per.reshape(E, C).sum(1)
        rec['cost_eff'] = (st['n_tgi'] * ce + st['n_ind']).reshape(E, C).sum(1)
        rec['indep'] = {'n_ind': st['n_ind'].reshape(E, C).sum(1), 'acc_ind': st['acc_ind'].reshape(E, C).mean(1)}
    rec['max_rej'] = out['max_rej'].reshape(E, C).max(1).values
    rec['acc_chain'] = out['acc'].reshape(E, C)
    rec['max_rej_chain'] = out['max_rej'].reshape(E, C)
    if dictfit is not None:
        rec.update({'estimate': res['cv'], 'estimate_fixed': res['cv_fixed'], 'theta1': res['theta1'], 'theta2': res['theta2'], 'chain_fixed': res['chain_fixed']})
        if 'cv_reg' in res:
            rec.update({'estimate_reg': res['cv_reg'], 'chain_reg': res['chain_reg']})
    if snap_steps:
        rec['snapshots'] = {}
        for k, N in snap_steps.items():
            r = out['recorder'].snaps[k]
            rec['snapshots'][N] = {'plain': r['plain'], 'cost': C * (n_burn * cb + k * ce), 'cost_eff': C * k * ce, **({'estimate': r['cv'], 'estimate_fixed': r['cv_fixed']} if dictfit is not None else {})}
    return rec
