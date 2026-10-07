import itertools
import math
from collections import Counter
import torch

def count_indices(dim, max_degree, max_support=None):
    p = max_degree if max_support is None else min(max_support, max_degree)
    return sum((math.comb(dim, s) * math.comb(deg - 1, s - 1) for deg in range(1, max_degree + 1) for s in range(1, min(p, deg) + 1)))

def _compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield ((first,) + rest)

def enumerate_indices(dim, max_degree, max_support=None, max_indices=4000000):
    n = count_indices(dim, max_degree, max_support)
    if n > max_indices:
        raise ValueError(f'dictionary would hold {n:,} indices (dim={dim}, max_degree={max_degree}, max_support={max_support}); cap the support (max_support=1 gives {count_indices(dim, max_degree, 1):,}) or raise max_indices if you really mean it')
    if max_support is None or max_support >= max_degree:
        return [comb for deg in range(1, max_degree + 1) for comb in itertools.combinations_with_replacement(range(dim), deg)]
    out = []
    for deg in range(1, max_degree + 1):
        for s in range(1, min(max_support, deg) + 1):
            for coords in itertools.combinations(range(dim), s):
                for parts in _compositions(deg, s):
                    out.append(tuple((c for c, k in zip(coords, parts) for _ in range(k))))
    return out

class HermiteDictionary:

    def __init__(self, dim, max_degree=3, max_support=None, multi_idx=None, max_indices=4000000):
        self.dim, self.max_degree = (dim, max_degree)
        self.max_support = max_support
        if multi_idx is None:
            multi_idx = enumerate_indices(dim, max_degree, max_support, max_indices)
        self.multi_idx = list(multi_idx)
        self.idx_powers = [tuple(sorted(Counter(comb).items())) for comb in self.multi_idx]
        self.degrees = torch.tensor([float(len(comb)) for comb in self.multi_idx])
        self.supports = torch.tensor([float(len(set(comb))) for comb in self.multi_idx])
        dev = self.degrees.device
        groups = {}
        for i, powers in enumerate(self.idx_powers):
            key = tuple((p for _, p in powers))
            pos, coords = groups.setdefault(key, ([], []))
            pos.append(i)
            coords.append([c for c, _ in powers])
        self._groups = [(key, torch.tensor(pos, dtype=torch.long, device=dev), torch.tensor(coords, dtype=torch.long, device=dev)) for key, (pos, coords) in sorted(groups.items())]

    def __len__(self):
        return len(self.multi_idx)

    def restrict(self, mask):
        keep = [i for i, m in enumerate(mask.tolist()) if m]
        return HermiteDictionary(self.dim, self.max_degree, max_support=self.max_support, multi_idx=[self.multi_idx[i] for i in keep])

    def he_table(self, z):
        tab = [torch.ones_like(z)]
        if self.max_degree >= 1:
            tab.append(z)
        for n in range(2, self.max_degree + 1):
            tab.append((z * tab[n - 1] - math.sqrt(n - 1) * tab[n - 2]) / math.sqrt(n))
        return tab

    def cond_table(self, m, tau):
        tau = torch.as_tensor(tau, dtype=m.dtype, device=m.device)
        tab = [torch.ones_like(m)]
        if self.max_degree >= 1:
            tab.append(m.clone())
        for n in range(2, self.max_degree + 1):
            tab.append((m * tab[n - 1] - math.sqrt(n - 1) * tau * tab[n - 2]) / math.sqrt(n))
        return tab

    def columns(self, tab, out=None):
        shape = tab[0].shape[:-1] + (len(self),)
        if out is None:
            out = torch.empty(shape, dtype=tab[0].dtype, device=tab[0].device)
        for powers, pos, coords in self._groups:
            v = tab[powers[0]][..., coords[:, 0]]
            for k in range(1, len(powers)):
                v = v * tab[powers[k]][..., coords[:, k]]
            out[..., pos] = v
        return out
