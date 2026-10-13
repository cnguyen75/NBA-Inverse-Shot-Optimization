"""v2 step 3, criteria (b) and (c), on the three real players. Run only if (a) passed.

    python scripts/v2_real.py       # writes results/v2_real.json

Per player: fit the quadratic cost on training shots for each rho in
{0.01, 0.1, 1}; choose rho by validation MAE in shot distance. Report the
underfit check (training stationarity R^2 of grad EP explained), the ablation
without the stationarity term (validation only), and then score the test set
once: learned cost vs maximising EP alone vs the player's mean training
decision, with paired bootstrap intervals (1,000 resamples).
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nbashots.data import SHOT_LOGS  # noqa: E402
from nbashots.v2 import PLAYERS, context, ep_grid, fit_cost, load, solve, split_player  # noqa: E402
from v2_synthetic import RHOS, SEED, load_ep  # noqa: E402

N_BOOT = 1000


def tens(df, ix):
    d = torch.tensor(df.iloc[ix][["s", "t"]].to_numpy(), dtype=torch.float32)
    return d, torch.from_numpy(context(df.iloc[ix])), torch.tensor(df.iloc[ix]["k"].to_numpy(),
                                                                   dtype=torch.float32)


def stationarity_r2(ep, stat, d, ctx):
    """1 - (mean squared stationarity residual, fitted multipliers) / (variance of grad EP)."""
    dd = d.clone().requires_grad_(True)
    g_ep = torch.autograd.grad(ep.ep(dd[:, 0], dd[:, 1], ctx).sum(), dd)[0]
    return 1 - stat / float(((g_ep - g_ep.mean(0)) ** 2).sum(1).mean())


def describe(cost):
    th = cost.theta()
    b, Q = np.array(th[:2]), np.array([[th[2], th[3]], [th[3], th[4]]])
    out = {"b": np.round(b, 5).tolist(), "Q": np.round(Q, 6).tolist()}
    if np.linalg.det(Q) > 1e-12:
        out["cost_minimum_ft_s"] = np.round(-np.linalg.solve(Q, b), 2).tolist()
    out["cost_of_1ft_from_minimum_pts"] = round(0.5 * Q[0, 0], 5)
    out["cost_of_1s_from_minimum_pts"] = round(0.5 * Q[1, 1], 5)
    return out


def boot_diff(err_a, err_b, rng):
    n = len(err_a)
    d = [float((err_a[i] - err_b[i]).mean()) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main():
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    out = {}
    for i, p in enumerate(PLAYERS):
        tr, va, te = split_player(df, p, SEED + i)
        (d_tr, c_tr, k_tr), (d_va, c_va, k_va), (d_te, c_te, k_te) = (tens(df, x) for x in (tr, va, te))
        g_va, g_te = ep_grid(ep, c_va), ep_grid(ep, c_te)
        fits = {}
        for rho in RHOS:
            c, diag = fit_cost(ep, d_tr, c_tr, k_tr, rho, seed=SEED)
            pv = solve(g_va, k_va, c)
            fits[rho] = (c, diag, float((pv - d_va).abs()[:, 0].mean()), float((pv - d_va).abs()[:, 1].mean()))
        rho = min(RHOS, key=lambda r: fits[r][2])
        cost, diag, va_s, va_t = fits[rho]
        r2 = stationarity_r2(ep, diag["stationarity"], d_tr, c_tr)
        # (c) ablation: no stationarity term, validation only
        abl, _ = fit_cost(ep, d_tr, c_tr, k_tr, rho, stationarity=False, seed=SEED)
        pa = solve(g_va, k_va, abl)
        th, th_a = cost.theta(), abl.theta()
        ablation = {"theta_with": np.round(th, 5).tolist(), "theta_without": np.round(th_a, 5).tolist(),
                    "relative_difference": round(float(np.linalg.norm(th - th_a) / np.linalg.norm(th)), 4),
                    "val_mae_s_with": round(va_s, 3),
                    "val_mae_s_without": round(float((pa - d_va).abs()[:, 0].mean()), 3),
                    "val_mae_t_with": round(va_t, 3),
                    "val_mae_t_without": round(float((pa - d_va).abs()[:, 1].mean()), 3)}
        ablation["changes_what_is_learned"] = bool(
            ablation["relative_difference"] > 0.25 and ablation["val_mae_s_with"] != ablation["val_mae_s_without"])
        # (b) test set, scored once
        pred = {"learned_cost": solve(g_te, k_te, cost),
                "ep_only": solve(g_te, k_te, None),
                "player_mean": torch.minimum(d_tr.mean(0).expand(len(d_te), 2),
                                             torch.stack([torch.full_like(k_te, 35.0), k_te], 1))}
        err = {m: (v - d_te).abs().numpy() for m, v in pred.items()}
        rng = np.random.default_rng(SEED + 300 + i)
        test = {m: {"mae_s_ft": round(float(e[:, 0].mean()), 3), "mae_t_s": round(float(e[:, 1].mean()), 3),
                    "pred_s_mean": round(float(pred[m][:, 0].mean()), 2),
                    "pred_s_sd": round(float(pred[m][:, 0].std()), 2)} for m, e in err.items()}
        beats = {}
        for base in ("ep_only", "player_mean"):
            for j, var in ((0, "s"), (1, "t")):
                ci = boot_diff(err["learned_cost"][:, j], err[base][:, j], rng)
                beats[f"{var}_vs_{base}"] = {"diff_ci95": ci, "beats": bool(ci[1] < 0)}
        out[p] = {"n": {"train": len(tr), "val": len(va), "test": len(te)}, "rho": rho,
                  "train_residuals": diag, "stationarity_r2_train": round(r2, 4),
                  "quadratic_underfits": bool(r2 < 0.2), "cost": describe(cost),
                  "actual_test_mean": {"s": round(float(d_te[:, 0].mean()), 2), "t": round(float(d_te[:, 1].mean()), 2)},
                  "ablation_validation": ablation, "test": test, "test_comparisons": beats}
        print(p, json.dumps({"rho": rho, "r2": out[p]["stationarity_r2_train"], "test": test,
                             "beats": {k: v["beats"] for k, v in beats.items()},
                             "ablation": ablation["changes_what_is_learned"]}))
    works = sum(out[p]["test_comparisons"]["s_vs_ep_only"]["beats"]
                and out[p]["test_comparisons"]["s_vs_player_mean"]["beats"] for p in PLAYERS) >= 2
    res = {"command": "python scripts/v2_real.py", "criterion_b_works": bool(works), "players": out}
    with open(os.path.join(ROOT, "results", "v2_real.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("criterion (b) works:", works)


if __name__ == "__main__":
    main()
