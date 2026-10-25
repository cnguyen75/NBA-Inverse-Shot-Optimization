"""v2: inverse optimization in which the optimality conditions depend on the cost.

Forward problem (README, "v2"): a player chooses d = (s, t), shot distance in
feet and touch time in seconds, to maximise EP(s, t; x) - C(s, t) subject to
0 <= s <= 35 and 0 <= t <= k, where k is the shot clock at the catch.

EP is a frozen expected-points model; C(d) = b.d + 0.5 d'Qd with Q = LL'.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


S_MAX = 35.0
S_GRID = torch.arange(0, S_MAX + 1e-6, 0.25)  # 141 points
T_GRID = torch.arange(0, 24 + 1e-6, 0.1)  # 241 points
PLAYERS = ["lebron james", "stephen curry", "james harden"]


# ------------------------------------------------------------------ data
def load(path):
    """All shots passing the pre-registered filters, with decision and context columns."""
    df = pd.read_csv(path)
    n0 = len(df)
    df["k"] = df["SHOT_CLOCK"] + df["TOUCH_TIME"]
    keep = (
        (df["TOUCH_TIME"] > 0)
        & df["SHOT_CLOCK"].notna()
        & (df["k"] <= 24)
        & (df["SHOT_DIST"] <= S_MAX)
    )
    df = df[keep].reset_index(drop=True)
    df["s"] = df["SHOT_DIST"].astype(float)
    df["t"] = df["TOUCH_TIME"].astype(float)
    df["def"] = df["CLOSE_DEF_DIST"].astype(float)
    df["late"] = (df["PERIOD"] >= 4).astype(float)
    df["three"] = (df["PTS_TYPE"] == 3).astype(float)
    return df, n0


def split_player(df, player, seed):
    """60/20/20 by shot for one player (pre-registered)."""
    idx = np.flatnonzero(df["player_name"].to_numpy() == player)
    idx = np.random.default_rng(seed).permutation(idx)
    a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
    return idx[:a], idx[a:b], idx[b:]


# ------------------------------------------------------------------ expected points
class EPModel(nn.Module):
    """P(make | s, t, def, k, late, player) by a smooth MLP; P(three | s) logistic.

    EP = P(make) * (2 + P(three | s)). All inputs in real units; standardised inside.
    """

    def __init__(self, mean, std, three_a, three_c, hidden=32):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.three_a, self.three_c = float(three_a), float(three_c)
        self.net = nn.Sequential(
            nn.Linear(8, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def logit(self, s, t, ctx):
        """s, t: (...,) tensors; ctx: (..., 6) = def, k, late, three player flags."""
        num = torch.stack([s, t, ctx[..., 0], ctx[..., 1]], -1)
        num = (num - self.mean) / self.std
        return self.net(torch.cat([num, ctx[..., 2:]], -1)).squeeze(-1)

    def p_three(self, s):
        return torch.sigmoid(self.three_a * (s - self.three_c))

    def ep(self, s, t, ctx):
        return torch.sigmoid(self.logit(s, t, ctx)) * (2 + self.p_three(s))


def context(df):
    flags = np.stack([(df["player_name"] == p).to_numpy(float) for p in PLAYERS], 1)
    return np.concatenate([df[["def", "k", "late"]].to_numpy(float), flags], 1).astype(np.float32)


def fit_ep(train, val, seed=0, hidden=32, epochs=60, lr=3e-3, batch=2048):
    from sklearn.linear_model import LogisticRegression

    lr3 = LogisticRegression().fit(train[["s"]], train["three"])
    a = float(lr3.coef_[0, 0])
    c = float(-lr3.intercept_[0] / a)
    num = train[["s", "t", "def", "k"]].to_numpy(float)
    torch.manual_seed(seed)
    m = EPModel(num.mean(0), num.std(0), a, c, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tens = lambda d: (
        torch.tensor(d["s"].to_numpy(), dtype=torch.float32),  # noqa: E731
        torch.tensor(d["t"].to_numpy(), dtype=torch.float32),
        torch.from_numpy(context(d)),
        torch.tensor(d["FGM"].to_numpy(), dtype=torch.float32),
    )
    S, T, X, Y = tens(train)
    Sv, Tv, Xv, Yv = tens(val)
    bce = nn.BCEWithLogitsLoss()
    best, state, gen = 1e9, None, torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(S), generator=gen)
        for i in range(0, len(S), batch):
            j = perm[i : i + batch]
            loss = bce(m.logit(S[j], T[j], X[j]), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = bce(m.logit(Sv, Tv, Xv), Yv).item()
        if v < best:
            best, state = v, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(state)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m, best


# ------------------------------------------------------------------ cost and optimality
class QuadCost(nn.Module):
    """C(d) = b.d + 0.5 d'Qd, Q = LL' (positive semi-definite). d = (s ft, t s)."""

    def __init__(self, b=(0.0, 0.0), L=((0.05, 0.0), (0.0, 0.05))):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.L = nn.Parameter(torch.tensor(L, dtype=torch.float32))

    def Q(self):
        L = torch.tril(self.L)
        return L @ L.T

    def forward(self, d):
        return d @ self.b + 0.5 * ((d @ self.Q()) * d).sum(-1)

    def theta(self):
        Q = self.Q().detach()
        return np.array(
            [self.b[0].item(), self.b[1].item(), Q[0, 0].item(), Q[0, 1].item(), Q[1, 1].item()]
        )


def constraints(d, k):
    """g_j(d) <= 0 in feet and seconds: -s, s - 35, -t, t - k."""
    s, t = d[:, 0], d[:, 1]
    return torch.stack([-s, s - S_MAX, -t, t - k], 1)


def kkt_loss(ep_model, cost, d, ctx, k, lam_raw, rho, stationarity=True):
    """Mean squared stationarity residual + rho * complementary slackness, multipliers >= 0."""
    d = d.clone().requires_grad_(True)
    lam = nn.functional.softplus(lam_raw)  # dual feasibility
    f = ep_model.ep(d[:, 0], d[:, 1], ctx) - cost(d)  # objective (maximised)
    g = constraints(d, k)
    grad_f = torch.autograd.grad(f.sum(), d, create_graph=True)[0]
    grad_g = torch.autograd.grad((lam * g).sum(), d, create_graph=True)[0]
    r = grad_f - grad_g  # grad f = sum lam grad g
    stat = (r**2).sum(1).mean()
    cs = (lam * (-g)).sum(1).mean()  # >= 0, zero iff slack
    loss = (stat if stationarity else 0.0) + rho * cs
    return loss, stat.detach().item(), cs.detach().item(), r.detach()


def fit_cost(ep_model, d, ctx, k, rho, stationarity=True, steps=3000, lr=0.01, seed=0):
    torch.manual_seed(seed)
    cost = QuadCost()
    lam_raw = nn.Parameter(torch.full((len(d), 4), -4.0))
    opt = torch.optim.Adam(list(cost.parameters()) + [lam_raw], lr=lr)
    for _ in range(steps):
        loss, stat, cs, _ = kkt_loss(ep_model, cost, d, ctx, k, lam_raw, rho, stationarity)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _, stat, cs, r = kkt_loss(ep_model, cost, d, ctx, k, lam_raw, rho, True)
    return cost, {"stationarity": stat, "comp_slackness": cs}


# ------------------------------------------------------------------ forward solver
def ep_grid(ep_model, ctx, chunk=16):
    """EP on the (s, t) grid for each context row: (n, 141, 241)."""
    ns, nt = len(S_GRID), len(T_GRID)
    ss, tt = torch.meshgrid(S_GRID, T_GRID, indexing="ij")
    out = []
    with torch.no_grad():
        for i in range(0, len(ctx), chunk):
            c = ctx[i : i + chunk]
            n = len(c)
            x = c[:, None, None, :].expand(n, ns, nt, c.shape[-1])
            out.append(ep_model.ep(ss.expand(n, ns, nt), tt.expand(n, ns, nt), x))
    return torch.cat(out)


def solve(epg, k, cost):
    """argmax of EP - C over the grid with t <= k. Returns (n, 2) decisions."""
    ss, tt = torch.meshgrid(S_GRID, T_GRID, indexing="ij")
    with torch.no_grad():
        c = (
            torch.zeros_like(ss)
            if cost is None
            else cost(torch.stack([ss.reshape(-1), tt.reshape(-1)], 1)).reshape(ss.shape)
        )
        obj = epg - c
        obj = obj.masked_fill(tt[None] > k[:, None, None] + 1e-6, -1e9)
        flat = obj.reshape(len(obj), -1).argmax(1)
    nt = len(T_GRID)
    return torch.stack([S_GRID[flat // nt], T_GRID[flat % nt]], 1)
