import torch
RIDGE_GRID = torch.logspace(-8, 3, 23).tolist()

def features(x, g, m, s, degree):
    y = (x - m) / s
    gs = g / s
    cols = [gs]
    if degree >= 2:
        d = x.shape[-1]
        j, k = torch.triu_indices(d, d, device=x.device)
        diag = (j == k).to(x.dtype) * 2.0 / s[j] ** 2
        cols.append(diag + y[:, k] * gs[:, j] + y[:, j] * gs[:, k])
    return torch.cat(cols, dim=1)

def _gram_parts(H, F):
    return (H.T @ H, H.T @ F, H.sum(0), F.sum(0), H.shape[0])

def _decompose(parts):
    HH, HF, sH, sF, n = parts
    hb, fb = (sH / n, sF / n)
    A = HH - n * torch.outer(hb, hb)
    b = HF - n * torch.outer(hb, fb)
    w, V = torch.linalg.eigh(A)
    return (w, V, V.T @ b, hb, fb)

def _beta(dec, lam=None, rel_tol=1e-10):
    w, V, Vb, _, _ = dec
    if lam is None:
        keep = w > rel_tol * w.max().clamp_min(1e-300)
        inv = torch.where(keep, 1.0 / w.clamp_min(1e-300), torch.zeros_like(w))
    else:
        inv = 1.0 / (w.clamp_min(0.0) + lam)
    return V @ (inv[:, None] * Vb)

def fit_estimate(H, F, folds=None, ridge=False):
    if not ridge:
        dec = _decompose(_gram_parts(H, F))
        beta = _beta(dec)
        return (dec[4] - dec[3] @ beta, beta, None)
    sd = H.std(0).clamp_min(1e-12)
    Hs = H / sd
    n = H.shape[0]
    parts = [_gram_parts(Hs[folds == f], F[folds == f]) for f in range(5)]
    total = [sum((p[i] for p in parts)) for i in range(4)] + [n]
    decs = []
    for f in range(5):
        tr = [total[i] - parts[f][i] for i in range(4)] + [n - parts[f][4]]
        decs.append(_decompose(tr))
    best, best_err = (None, None)
    for rel in RIDGE_GRID:
        lam = rel * n
        err = torch.zeros(F.shape[1], dtype=F.dtype, device=F.device)
        for f in range(5):
            beta = _beta(decs[f], lam)
            hb, fb = (decs[f][3], decs[f][4])
            Hf, Ff = (Hs[folds == f], F[folds == f])
            err += ((Ff - fb - (Hf - hb) @ beta) ** 2).sum(0)
        tot = err.sum()
        if best_err is None or tot < best_err:
            best, best_err = (lam, tot)
    dec = _decompose(total)
    beta = _beta(dec, best)
    return (dec[4] - dec[3] @ beta, beta / sd[:, None], best)

def fit_estimate_chunked(x, g, F, m, s, degree, chunk=20000):
    parts = None
    for a in range(0, x.shape[0], chunk):
        H = features(x[a:a + chunk], g[a:a + chunk], m, s, degree)
        p = _gram_parts(H, F[a:a + chunk])
        parts = p if parts is None else tuple((u + v for u, v in zip(parts, p)))
    dec = _decompose(parts)
    beta = _beta(dec)
    return (dec[4] - dec[3] @ beta, beta)
