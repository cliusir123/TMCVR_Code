import torch
DTYPE = torch.float64
SMALL = dict(n_layers=8, hidden=64)
FINE = dict(n_layers=12, hidden=128, scale_bound=3.0)
TRAIN = dict(steps=10000, batch=256, lr=0.001, lr_min_ratio=0.01, clip=100.0, seed=2026)
ADAPTIVE = dict(chains=32, refresh=10, chain_steps=10, buffer=20000, fkl_start=0.1, acc_target=0.5)
MULTISTART = dict(chains=16, anchor=0.5, group_threshold=5.0)
TAIL = dict(band=(1.0, 5.0), scale=2.0, batch=128, margin=1.0, weight=1.0, start=0.1, ramp=0.1)
CHAINS, BURN, N_STEPS = (500, 500, 5000)
SAMPLERS = ['rwm', 'mala']
GAMMA_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075] + [round(0.1 + 0.05 * i, 2) for i in range(19)]
PRUNE_GRID = [0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
PRUNE_KS = [1, 2, 3, 4, 5, 6]
DICT = dict(K=4, p=2, lam='2log', n_select=100000, n_mc=100000)
J_ADAPT_RUNS = 5
REPLICATES = 50
BENCH_METHODS = ['tgi_rwm', 'gi_mala', 'gi_pmala_lgm', 'nuts']
BENCH_PLOT = ['TGI-RWM+CV', 'TGI-RWM', 'GI-MALA+CV', 'GI-MALA', 'NUTS+ZV2', 'NUTS']
COST_CAP = 15000000.0
FLOW_STEPS = [0, 100, 1000, 10000]
COST_REF = 10000000.0
CONV = dict(chains=5, steps=10000, burn=500, nuts_budget=2000000.0)
_SYN_CKPTS = [0, 100, 250, 500, 1000, 2000, 5000, 10000]
_REAL_CKPTS = [500, 1000, 2000, 5000, 10000]
TARGETS = {'banana': dict(scheme='A', arch=FINE, ckpts=_SYN_CKPTS, headline=4, tex_names=['x_2', 'x_2^2', 'x_1^2', '\\tfrac14\\sum_k x_{2k}', '\\bar x'], bench_ckpts=[100, 500, 2000, 10000], gti_budgets=[10000.0, 100000.0, 1000000.0, 10000000.0], chain_len=20000.0, nuts_budgets=[30000.0, 100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0], kwargs=dict(dim=8, p=10.0, b=0.1), analytic={'accurate': ('shear', 0.01), 'laplace_c1': ('shear', 1.0), 'coarse': ('shear', 0.3)}), 'funnel': dict(scheme='A', arch=FINE, ckpts=_SYN_CKPTS, headline=2, tex_names=['v', '\\tfrac1{d-1}\\sum_i\\log(1+x_i^2)', 'v^2', 'x_1'], table_note=' ZV-CV is exact for $v$: its second-order Stein features represent it.', bench_ckpts=[100, 500, 2000, 10000], gti_budgets=[10000.0, 100000.0, 1000000.0, 10000000.0], chain_len=20000.0, nuts_budgets=[30000.0, 100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0], kwargs=dict(dim=10, sigma_v=2.0), analytic={'accurate': ('scale', 0.01), 'coarse': ('scale', 0.3)}), 'gmix_d10_h6': dict(scheme='Aa', arch=FINE, pilot=400000.0, ms_cost_guess=100.0, tex_names=['x_1', 'x_1^2', '\\mathbb 1\\{x_1>0\\}', '\\bar x'], table_omit=['TGI-MALA'], ckpts=[0, 100, 250, 1000, 3000, 10000], headline=0, bench_ckpts=[100, 1000, 3000, 10000], gti_budgets=[10000.0, 100000.0, 1000000.0, 10000000.0], chain_len=20000.0, nuts_budgets=[30000.0, 100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0], analytic={'accurate': ('scale', 0.99501), 'coarse': ('scale', 0.85431)}), 'tmix_d10_h6': dict(scheme='Aa', arch=FINE, pilot=400000.0, ms_cost_guess=100.0, tex_names=['x_1', 'x_1^2', '\\mathbb 1\\{x_1>0\\}', '\\bar x'], ckpts=[0, 100, 1000, 10000], headline=0, chain_len=20000.0, bench_ckpts=[100, 1000, 10000], gti_budgets=[10000.0, 100000.0, 1000000.0, 10000000.0], nuts_budgets=[30000.0, 100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0]), 'sparse_logistic_german': dict(scheme='B', arch=FINE, headline=0, ckpts=[1000, 2000, 5000, 10000], gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=100000.0), 'eight_schools': dict(scheme='A', arch=SMALL, ckpts=_REAL_CKPTS, headline=1, gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=100000.0), 'stochvol_T100': dict(scheme='B', arch=FINE, headline=0, ckpts=[1000, 2000, 5000, 10000], gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=100000.0), 'lotka_volterra': dict(scheme='A', arch=SMALL, ckpts=_REAL_CKPTS, headline=0, skip_pmala=True, gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=20000.0), 'sensor': dict(scheme='Aa', arch=FINE, pilot=1000000.0, ms_cost_guess=30.0, ckpts=[2000, 5000, 10000], headline=0, gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=100000.0), 'phi4_N32': dict(scheme='Aa', arch=FINE, pilot=1000000.0, ms_cost_guess=30.0, ckpts=[1000, 3000, 10000], headline=1, gti_budgets=[1000000.0, 10000000.0], chain_len=100000.0), **{t: dict(scheme=s, arch=SMALL, ckpts=_REAL_CKPTS, headline=h, gti_budgets=[100000.0, 1000000.0, 10000000.0], chain_len=20000.0) for t, s, h in (('pdb_dugongs', 'B', 2), ('pdb_garch', 'A', 2), ('pdb_one_comp', 'A', 0), ('pdb_gp_pois', 'A', 0), ('pdb_surgical', 'A', 1), ('pdb_pilots', 'B', 2))}}
INDEP = dict(delta=0.01, df=10)

def indep(t):
    return TARGETS[t].get('indep', INDEP)
TARGETS['pdb_one_comp']['skip_pmala'] = True
TARGETS['banana']['indep'] = dict(INDEP, delta=0.0)
TARGETS['stochvol_T100'].update(indep=dict(INDEP, delta=0.0))
TARGETS['pdb_one_comp'].update(indep=dict(INDEP, delta=0.0), gti_budgets=[1000000.0, 10000000.0], bench_ckpts=[10000], nuts_budgets=[1000000.0, 10000000.0], tex_names=['k_a', '\\log K_m', '\\log V_m', '\\sigma'])
TARGETS['pdb_garch']['tex_names'] = ['\\mu', '\\alpha_0', '\\alpha_1', '\\beta_1']
TARGETS['pdb_dugongs']['tex_names'] = ['\\alpha', '\\beta', '\\lambda', '\\sigma']
TARGETS['pdb_gp_pois']['tex_names'] = ['\\ell', '\\alpha', 'f_1', 'f_6']
TARGETS['stochvol_T100']['tex_names'] = ['\\mu', '\\phi', '\\sigma^2', '\\bar h']
PREFIX_N = [1000000.0, 10000000.0]
PREFIX_CHAINS = 50
for _t in ('phi4_N32', 'sparse_logistic_german', 'stochvol_T100', 'pdb_pilots'):
    TARGETS[_t]['prefix'] = PREFIX_N
for _t in ('sparse_logistic_german', 'stochvol_T100', 'pdb_pilots'):
    TARGETS[_t]['tail'] = False
TARGETS['pdb_pilots']['nuts_vectorized'] = True
TARGETS['phi4_N32'].update(tail=False, pilot=4000000.0, multistart=dict(chains=64), indep=dict(INDEP, delta=0.0), tex_names=['\\mathbf 1\\{M>0\\}', 'M', 'M^2', 'U/N'])
TARGETS['eight_schools'].update(arch=FINE, tail=False, train=dict(steps=30000), ckpts=[3000, 10000, 30000], bench_ckpts=[30000], gti_budgets=[1000000.0, 10000000.0], indep=dict(INDEP, delta=0.0), tex_names=['\\mu', '\\log\\tau', '\\theta_1'])
TARGETS['pdb_surgical'].update(arch=FINE, tail=False, bench_ckpts=[10000], gti_budgets=[1000000.0, 10000000.0], indep=dict(INDEP, delta=0.0))
TARGETS['sparse_logistic_german'].update(indep=dict(INDEP, delta=0.0))
TARGETS['pdb_gp_pois'].update(arch=FINE, tail=False, bench_ckpts=[10000], gti_budgets=[1000000.0, 10000000.0], indep=dict(INDEP, delta=0.0), table_note=' At $N=10^7$ the errors on $\\ell$ and $\\alpha$ are limited by the precision of the reference values (Monte Carlo standard error $6.6\\times10^{-4}$ and $3.3\\times10^{-4}$ posterior standard deviations).')
TARGETS['sparse_logistic_german'].update(arch=dict(n_layers=12, hidden=256, scale_bound=3.0), train=dict(steps=300000, lr=0.0003, batch=1024), ckpts=[30000, 100000, 300000], bench_ckpts=[300000], gti_budgets=[1000000.0, 10000000.0], chain_len=20000.0, prefix_methods=['gi_mala'])
TARGETS['sparse_logistic_german'].update(nuts_extrapolate=True, fig_fns=[0, 1, 19, 20], table_range=dict(keep=[0], range=list(range(1, 26)), name='$w_1,\\dots,w_{25}$ (range)'), tex_names=['\\log\\tau'] + [f'w_{{{j + 1}}}' for j in range(25)])
TARGETS['lotka_volterra']['tex_names'] = ['\\alpha', '\\beta', '\\gamma', '\\delta']
TARGETS['pdb_surgical']['tex_names'] = ['\\mu', '\\log\\sigma^2', 'b_1', '\\mathrm{logit}^{-1}\\mu']
TARGETS['phi4_N32'].update(nuts_extrapolate=True, nuts_extrapolate_skip=[0, 1])
TARGETS['phi4_N32'].update(chain_len=20000.0, gti_budgets=[1000000.0, 10000000.0], bench_ckpts=[10000])
TARGETS['phi4_N32']['prefix_methods'] = ['gi_mala']
for _t, _c in TARGETS.items():
    _c.setdefault('nuts_budgets', [100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0])
    _c.setdefault('kwargs', {})
    _c.setdefault('analytic', {})
    _c.setdefault('skip_pmala', _t not in ('banana', 'funnel', 'gmix_d10_h6'))
    _c.setdefault('bench_ckpts', _c['ckpts'] if len(_c['ckpts']) <= 3 else [_c['ckpts'][1], _c['ckpts'][-2], _c['ckpts'][-1]])
for _t, _j in {'pdb_garch': 3, 'banana': 0, 'pdb_one_comp': 1, 'lotka_volterra': 0, 'pdb_dugongs': 1, 'gmix_d10_h6': 0, 'funnel': 0, 'pdb_gp_pois': 1, 'sparse_logistic_german': 24, 'phi4_N32': 0, 'eight_schools': 2, 'tmix_d10_h6': 0}.items():
    TARGETS[_t]['main_fn'] = _j
