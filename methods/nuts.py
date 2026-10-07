import math
import time
import numpy as np
COST_GUESS = 30.0

def _potential(target_name, target):
    import jax.numpy as jnp
    if hasattr(target, 'jax_logp'):
        return lambda x: -target.jax_logp(x)
    if target_name == 'banana':
        p, b = (float(target.p), float(target.b))

        def logp(x):
            x1, x2 = (x[0], x[1])
            r = x2 + b * x1 ** 2 - b * p
            return -x1 ** 2 / (2 * p) - 0.5 * r ** 2 - 0.5 * jnp.sum(x[2:] ** 2)
    else:
        sv, d = (float(target.sigma_v), target.dim)

        def logp(x):
            v, rest = (x[0], x[1:])
            return -v ** 2 / (2 * sv ** 2) - 0.5 * (d - 1) * v - 0.5 * jnp.exp(-v) * jnp.sum(rest ** 2)
    return lambda x: -logp(x)

def run_nuts(target_name, target, E, budget, seed=0, block=2000, max_tree_depth=10, init=None, chain_method='parallel', cost_guess=COST_GUESS):
    import numpyro
    numpyro.set_host_device_count(E)
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS
    jax.config.update('jax_enable_x64', True)
    t0 = time.perf_counter()
    pot = _potential(target_name, target)
    m, _ = target.gaussian_init()
    m = np.asarray(m.cpu(), dtype=np.float64)
    rng = np.random.default_rng(seed)
    if init is None:
        init = m[None, :] + 0.1 * rng.standard_normal((E, m.size))
    else:
        init = np.asarray(init, dtype=np.float64)
        assert init.shape == (E, m.size)
    n_warm = int(max(20, min(1000, 0.2 * budget / cost_guess)))
    kernel = NUTS(potential_fn=pot, target_accept_prob=0.8, max_tree_depth=max_tree_depth, adapt_mass_matrix=True, dense_mass=False)
    mcmc = MCMC(kernel, num_warmup=n_warm, num_samples=block, num_chains=E, chain_method=chain_method, progress_bar=False)
    key = jax.random.PRNGKey(seed)
    mcmc.warmup(key, init_params=jnp.asarray(init), extra_fields=('num_steps',), collect_warmup=True)
    warm_steps = np.asarray(mcmc.get_extra_fields(group_by_chain=True)['num_steps']).sum(1).astype(float)
    spent = 2.0 * warm_steps
    w = np.asarray(mcmc.get_extra_fields(group_by_chain=True)['num_steps'], dtype=float)
    per_iter = max(1.0, w[:, w.shape[1] // 2:].mean())
    need = max(budget - spent.min(), 2.0 * per_iter)
    mcmc.num_samples = int(1.5 * need / (2.0 * per_iter)) + 10
    xs, gs, steps = ([], [], [])
    while True:
        mcmc.post_warmup_state = mcmc.last_state
        mcmc.run(mcmc.post_warmup_state.rng_key, extra_fields=('num_steps', 'z_grad'))
        x = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=np.float32)
        ex = mcmc.get_extra_fields(group_by_chain=True)
        xs.append(x)
        gs.append(-np.asarray(ex['z_grad'], dtype=np.float32))
        st = np.asarray(ex['num_steps']).astype(float)
        steps.append(st)
        spent = spent + 2.0 * st.sum(1)
        if spent.min() >= budget:
            break
        rate = max(1.0, st.mean())
        mcmc.num_samples = int(2.0 * (budget - spent.min()) / (2.0 * rate)) + 10
    X, Gr, S = (np.concatenate(a, 1) for a in (xs, gs, steps))
    out = []
    for e in range(E):
        c = 2.0 * warm_steps[e] + np.cumsum(2.0 * S[e])
        n = int(np.searchsorted(c, budget, side='right'))
        out.append({'x': X[e, :n].astype(np.float64), 'grad': Gr[e, :n].astype(np.float64), 'cost_warm': 2.0 * warm_steps[e], 'cum_cost': c[:n], 'cost_sampling': float(2.0 * S[e, :n].sum()), 'warm_grad': warm_steps[e], 'grad_evals': S[e, :n].sum(), 'mean_steps': float(S[e, :n].mean()) if n else math.nan, 'n_warm': n_warm})
    return (out, time.perf_counter() - t0)
