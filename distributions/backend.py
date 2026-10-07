import numpy as np
import torch

class _Torch:
    name = 'torch'
    exp, log, log1p, expm1 = (torch.exp, torch.log, torch.log1p, torch.expm1)
    tanh, sqrt, abs, square = (torch.tanh, torch.sqrt, torch.abs, torch.square)
    where = staticmethod(torch.where)
    softplus = staticmethod(torch.nn.functional.softplus)
    logsigmoid = staticmethod(torch.nn.functional.logsigmoid)
    sigmoid = staticmethod(torch.sigmoid)

    @staticmethod
    def cholesky(A):
        L, info = torch.linalg.cholesky_ex(A)
        bad = (info != 0)[..., None, None]
        return torch.where(bad, torch.full_like(L, float('nan')), L)

    @staticmethod
    def stack(xs, axis=-1):
        return torch.stack(xs, dim=axis)

    @staticmethod
    def concat(xs, axis=-1):
        return torch.cat(xs, dim=axis)

    @staticmethod
    def const(a, like):
        return torch.as_tensor(a, dtype=like.dtype, device=like.device)

    @staticmethod
    def clip(x, lo, hi):
        return torch.clamp(x, lo, hi)

    @staticmethod
    def rk4(f, y0, h, n_sub, n_out):
        y, out = (y0, [])
        for _ in range(n_out):
            for _ in range(n_sub):
                k1 = f(y)
                k2 = f(y + 0.5 * h * k1)
                k3 = f(y + 0.5 * h * k2)
                k4 = f(y + h * k3)
                y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(y)
        return torch.stack(out, dim=-2)

class _Jax:
    name = 'jax'

    def __init__(self):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp = (jax, jnp)
        self.exp, self.log, self.log1p = (jnp.exp, jnp.log, jnp.log1p)
        self.expm1, self.tanh, self.sqrt = (jnp.expm1, jnp.tanh, jnp.sqrt)
        self.abs, self.square, self.where = (jnp.abs, jnp.square, jnp.where)
        self.softplus = jax.nn.softplus
        self.logsigmoid = jax.nn.log_sigmoid
        self.sigmoid = jax.nn.sigmoid
        self.cholesky = jnp.linalg.cholesky

    def stack(self, xs, axis=-1):
        return self.jnp.stack(xs, axis=axis)

    def concat(self, xs, axis=-1):
        return self.jnp.concatenate(xs, axis=axis)

    def const(self, a, like=None):
        return self.jnp.asarray(np.asarray(a))

    def clip(self, x, lo, hi):
        return self.jnp.clip(x, lo, hi)

    def rk4(self, f, y0, h, n_sub, n_out):
        jax, jnp = (self.jax, self.jnp)

        def step(y, _):
            k1 = f(y)
            k2 = f(y + 0.5 * h * k1)
            k3 = f(y + 0.5 * h * k2)
            k4 = f(y + h * k3)
            return (y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None)

        def block(y, _):
            y, _ = jax.lax.scan(step, y, None, length=n_sub)
            return (y, y)
        _, ys = jax.lax.scan(block, y0, None, length=n_out)
        return jnp.moveaxis(ys, 0, -2)
TORCH = _Torch()
_JAX = None

def jax_backend():
    global _JAX
    if _JAX is None:
        _JAX = _Jax()
    return _JAX
