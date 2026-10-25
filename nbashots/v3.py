"""v3: a choice model over the feasible grid (README, "v3").

P(d | x) ∝ exp(a·EP(d; x) - b_mid·[mid] - b_long·[long] - b_three·[three] - c·t)
with a = 1/T, b = β/T, c = α/T. Utility is linear in w = (a, b_mid, b_long,
b_three, c), so the log-likelihood is concave and is maximised by Newton's
method with the exact gradient and information matrix.
"""

import numpy as np
import torch

from .v2 import S_GRID, T_GRID


PARAMS = ["a", "b_mid", "b_long", "b_three", "c"]
THETA = ["T", "beta_mid", "beta_long", "beta_three", "alpha"]
BAND_EDGES = (4.0, 14.0, 22.0)


def band_features():
    s = S_GRID.double()
    return torch.stack(
        [((s > 4) & (s <= 14)).double(), ((s > 14) & (s <= 22)).double(), (s > 22).double()], 1
    )  # (141, 3)


def features(epg):
    """Per-cell feature tensor f with u = f·w: (n, 141, 241, 5), float64."""
    n, ns, nt = epg.shape
    bands = band_features()[None, :, None, :].expand(n, ns, nt, 3)
    t = T_GRID.double()[None, None, :, None].expand(n, ns, nt, 1)
    return torch.cat([epg.double()[..., None], -bands, -t], -1)


def feasible(k):
    return T_GRID[None, None, :] <= k[:, None, None] + 1e-6  # (n, 1, 241)


def cell_index(d, k):
    """Nearest grid cell for observed decisions, t capped at the last feasible cell."""
    si = torch.clamp(torch.round(d[:, 0] / 0.25).long(), 0, len(S_GRID) - 1)
    tmax = torch.floor(k / 0.1 + 1e-6).long()
    ti = torch.minimum(torch.clamp(torch.round(d[:, 1] / 0.1).long(), 0, len(T_GRID) - 1), tmax)
    return si, ti


class ChoiceData:
    """Holds features, feasibility and observed cells for one set of shots, in chunks."""

    def __init__(self, epg, k, d=None, chunk=24):
        self.chunks = []
        for i in range(0, len(epg), chunk):
            e, kk = epg[i : i + chunk], k[i : i + chunk]
            self.chunks.append((e, kk, None if d is None else cell_index(d[i : i + chunk], kk)))

    def stats(self, w, need_info=True):
        """Log-likelihood, gradient, information, and per-shot probabilities' moments."""
        w = torch.as_tensor(w, dtype=torch.float64)
        ll, grad, info = (
            0.0,
            torch.zeros(5, dtype=torch.float64),
            torch.zeros(5, 5, dtype=torch.float64),
        )
        per_shot = []
        for e, kk, obs in self.chunks:
            f = features(e)
            u = (f * w).sum(-1)
            u = u.masked_fill(~feasible(kk).expand_as(u), -np.inf)
            logz = torch.logsumexp(u.reshape(len(u), -1), 1)
            p = torch.exp(u - logz[:, None, None])
            ef = (p[..., None] * f).sum((1, 2))
            if obs is not None:
                si, ti = obs
                idx = torch.arange(len(u))
                lli = u[idx, si, ti] - logz
                ll += float(lli.sum())
                grad += (f[idx, si, ti] - ef).sum(0)
                per_shot.append(lli)
            if need_info:
                eff = torch.einsum("nij,nijk,nijl->nkl", p, f, f)
                info += (eff - ef[:, :, None] * ef[:, None, :]).sum(0)
        return ll, grad, info, (torch.cat(per_shot) if per_shot else None)

    def probs(self, w):
        """Yield per-chunk probability grids (n, 141, 241)."""
        w = torch.as_tensor(w, dtype=torch.float64)
        for e, kk, _ in self.chunks:
            u = (features(e) * w).sum(-1)
            u = u.masked_fill(~feasible(kk).expand_as(u), -np.inf)
            yield torch.softmax(u.reshape(len(u), -1), 1).reshape(u.shape)


def fit(data, free=None, w0=None, iters=50, tol=1e-8):
    """Newton's method on the concave log-likelihood. `free` selects parameters to fit;
    the others stay at w0 (default 0, except a = 5)."""
    free = list(range(5)) if free is None else list(free)
    w = torch.tensor(w0 if w0 is not None else [5.0, 0, 0, 0, 0], dtype=torch.float64)
    ll_old = -np.inf
    for _ in range(iters):
        ll, g, H, _ = data.stats(w)
        g, H = g[free], H[np.ix_(free, free)]
        step = torch.linalg.pinv(H) @ g  # pseudo-inverse: a band with no shots makes H singular
        t = 1.0
        while True:  # backtracking line search
            w_new = w.clone()
            w_new[free] = w[free] + t * step
            ll_new = data.stats(w_new, need_info=False)[0]
            if ll_new >= ll - 1e-9 or t < 1e-4:
                break
            t /= 2
        w = w_new
        if abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
    ll, g, H, per = data.stats(w)
    return w, ll, H, per


def theta(w):
    a = w[0]
    return np.array([1 / a, w[1] / a, w[2] / a, w[3] / a, w[4] / a], dtype=float)


def theta_cov(w, H):
    """Delta-method covariance of theta from the inverse information of w."""
    a = float(w[0])
    J = np.zeros((5, 5))
    J[0, 0] = -1 / a**2
    for j in range(1, 5):
        J[j, 0] = -float(w[j]) / a**2
        J[j, j] = 1 / a
    cov_w = np.linalg.pinv(H.numpy())
    return J @ cov_w @ J.T, cov_w


def identifiability(w, H):
    Hn = H.numpy()
    eig = np.linalg.eigvalsh(Hn)
    cov = np.linalg.pinv(Hn)
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    off = np.abs(corr - np.eye(5)).max()
    wn = np.abs(w.numpy())
    se_ok = [bool(se[j] <= 0.25 * wn[j]) if wn[j] >= 1 else bool(se[j] <= 0.25) for j in range(5)]
    return {
        "min_eigenvalue": float(eig.min()),
        "condition_number": float(eig.max() / eig.min()),
        "max_abs_correlation": round(float(off), 4),
        "correlation": np.round(corr, 3).tolist(),
        "estimate": np.round(w.numpy(), 4).tolist(),
        "se": np.round(se, 4).tolist(),
        "se_ok": se_ok,
        "passed": bool(eig.min() > 0 and off <= 0.95 and all(se_ok)),
    }


def band_shares(data, w):
    """Mean predicted share of each distance band (rim, mid, long, three) over the shots."""
    bands = torch.cat([1 - band_features().sum(1, keepdim=True), band_features()], 1)  # (141, 4)
    tot, n = torch.zeros(4, dtype=torch.float64), 0
    for p in data.probs(w):
        tot += torch.einsum("nij,ik->k", p, bands)
        n += len(p)
    return (tot / n).numpy()


def expected_s(data, w):
    out = []
    for p in data.probs(w):
        out.append((p.sum(2) * S_GRID.double()).sum(1))
    return torch.cat(out).numpy()


def sample(data, w, rng):
    """Draw one cell per shot from the choice distribution; returns (s, t)."""
    out = []
    for p in data.probs(w):
        flat = p.reshape(len(p), -1).numpy()
        for row in flat:
            j = rng.choice(len(row), p=row / row.sum())
            out.append((float(S_GRID[j // len(T_GRID)]), float(T_GRID[j % len(T_GRID)])))
    return torch.tensor(out, dtype=torch.float32)
