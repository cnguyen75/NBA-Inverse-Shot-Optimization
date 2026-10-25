"""v3 criterion (a) and (a'): synthetic recovery and identifiability.

    python scripts/v3_synthetic.py      # writes results/v3_synthetic.json

Synthetic parameters, noise, thresholds and checks are those pre-registered in
the README (v3 section). Decisions are drawn from the true choice distribution
on one real player's training contexts, then Gaussian noise (sd 1.0 ft,
0.25 s) is added, clipped to the feasible set and rounded to the grid.
Band shares for the total-variation check use that player's validation
contexts. Profile likelihoods: each parameter of the linear form fixed at 9
values across ±4 standard errors, the others refitted.
"""

import json
import os
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v2_synthetic import SEED, load_ep  # noqa: E402

from nbashots.data import SHOT_LOGS  # noqa: E402
from nbashots.v2 import PLAYERS, S_MAX, context, ep_grid, load, split_player  # noqa: E402
from nbashots.v3 import (  # noqa: E402
    PARAMS,
    THETA,
    ChoiceData,
    band_shares,
    fit,
    identifiability,
    sample,
    theta,
    theta_cov,
)


NOISE = (1.0, 0.25)
SYNTHETIC = {  # (T, beta_mid, beta_long, beta_three, alpha), as pre-registered
    "rim": (0.15, 0.15, 0.25, 0.30, 0.05),
    "three": (0.15, 0.05, 0.05, -0.20, 0.05),
    "midrange_pullup": (0.15, -0.15, -0.15, 0.10, 0.02),
}


def to_w(th):
    T = th[0]
    return [1 / T, th[1] / T, th[2] / T, th[3] / T, th[4] / T]


def profile(data, w, H, j, n=9):
    se = float(np.sqrt(abs(np.linalg.pinv(H.numpy())[j, j])))
    ll_max = data.stats(w, need_info=False)[0]
    vals = np.linspace(float(w[j]) - 4 * se, float(w[j]) + 4 * se, n)
    free = [i for i in range(5) if i != j]
    drops = []
    for v in vals:
        w0 = w.clone().numpy()
        w0[j] = v
        _, ll, _, _ = fit(data, free=free, w0=w0, iters=20)
        drops.append(round(ll_max - ll, 3))
    inside = [v for v, dr in zip(vals, drops) if dr <= 1.92]
    return {
        "values": np.round(vals, 4).tolist(),
        "loglik_drop": drops,
        "interval_95": [round(min(inside), 4), round(max(inside), 4)] if inside else None,
        "bounded": bool(drops[0] > 1.92 and drops[-1] > 1.92),
    }


def main():
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    rng = np.random.default_rng(SEED)
    out = {}
    for i, (name, th_true) in enumerate(SYNTHETIC.items()):
        tr, va, _ = split_player(df, PLAYERS[i], SEED + i)
        ctx_tr, ctx_va = (torch.from_numpy(context(df.iloc[x])) for x in (tr, va))
        k_tr, k_va = (
            torch.tensor(df.iloc[x]["k"].to_numpy(), dtype=torch.float32) for x in (tr, va)
        )
        g_tr, g_va = ep_grid(ep, ctx_tr), ep_grid(ep, ctx_va)
        w_true = torch.tensor(to_w(th_true), dtype=torch.float64)
        d = sample(ChoiceData(g_tr, k_tr), w_true, rng)
        s = np.clip(d[:, 0].numpy() + rng.normal(0, NOISE[0], len(d)), 0, S_MAX)
        t = np.clip(d[:, 1].numpy() + rng.normal(0, NOISE[1], len(d)), 0, k_tr.numpy())
        d_obs = torch.tensor(np.stack([s, t], 1), dtype=torch.float32)
        data = ChoiceData(g_tr, k_tr, d_obs)
        w, ll, H, _ = fit(data)
        th_hat = theta(w)
        cov_th, _ = theta_cov(w, H)
        rel = float(np.linalg.norm(th_hat - np.array(th_true)) / np.linalg.norm(th_true))
        va_data = ChoiceData(g_va, k_va)
        sh_hat, sh_true = band_shares(va_data, w), band_shares(va_data, w_true)
        tv = float(0.5 * np.abs(sh_hat - sh_true).sum())
        ident = identifiability(w, H)
        prof = {PARAMS[j]: profile(data, w, H, j) for j in range(5)}
        out[name] = {
            "contexts_from": PLAYERS[i],
            "n_train": len(tr),
            "theta_true": dict(zip(THETA, th_true)),
            "theta_hat": dict(zip(THETA, np.round(th_hat, 4).tolist())),
            "theta_se": dict(zip(THETA, np.round(np.sqrt(np.diag(cov_th)), 4).tolist())),
            "relative_theta_error": round(rel, 4),
            "band_shares_true": np.round(sh_true, 4).tolist(),
            "band_shares_hat": np.round(sh_hat, 4).tolist(),
            "band_share_tv": round(tv, 4),
            "recovered": bool(rel <= 0.25 and tv <= 0.05),
            "identifiability": ident,
            "profiles": prof,
        }
        print(
            name,
            {k: out[name][k] for k in ("relative_theta_error", "band_share_tv", "recovered")},
            "ident",
            ident["passed"],
            "max|corr|",
            ident["max_abs_correlation"],
            "profiles bounded",
            all(p["bounded"] for p in prof.values()),
        )
    res = {
        "command": "python scripts/v3_synthetic.py",
        "noise_sd": NOISE,
        "players": out,
        "criterion_a_passed": all(v["recovered"] for v in out.values()),
        "identifiability_passed": all(v["identifiability"]["passed"] for v in out.values()),
    }
    with open(os.path.join(ROOT, "results", "v3_synthetic.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(
        "criterion (a):",
        res["criterion_a_passed"],
        " identifiability (a'):",
        res["identifiability_passed"],
    )


if __name__ == "__main__":
    main()
