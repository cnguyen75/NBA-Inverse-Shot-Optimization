"""v2 step 2, criterion (a): can the method recover known costs?

    python scripts/v2_synthetic.py      # writes results/v2_synthetic.json

Three synthetic players with fixed quadratic costs C(d) = 0.5 (d - mu)'Q(d - mu)
(so b = -Q mu), written here before any run and not changed afterwards. Each
uses one real player's training contexts (defender distance, shot clock at the
catch, period) and chooses d by solving the forward problem on the grid with
the frozen EP model, plus Gaussian noise (sd 1.0 ft, 0.25 s), clipped to the
feasible set. The cost is then fitted from (decision, context) pairs alone.

rho is chosen from {0.01, 0.1, 1} by decision MAE on the first half of the
synthetic validation contexts; the agreement criterion uses the second half.
Recovered (pre-registered): relative error of theta = (b, Q11, Q12, Q22) <= 0.25,
and median |s_hat - s_true| <= 1.0 ft and median |t_hat - t_true| <= 0.25 s.
A noise-free run is also reported, for diagnosis only.
"""

import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nbashots.data import SHOT_LOGS  # noqa: E402
from nbashots.v2 import (  # noqa: E402
    PLAYERS,
    S_MAX,
    EPModel,
    QuadCost,
    context,
    ep_grid,
    fit_cost,
    load,
    solve,
    split_player,
)


SEED = 42
RHOS = (0.01, 0.1, 1.0)
NOISE = (1.0, 0.25)
SYNTHETIC = {  # name: (mu in (ft, s), Q in points per ft^2 / s^2), fixed before any run
    "rim": ((3.0, 1.5), ((0.010, 0.0), (0.0, 0.020))),
    "three": ((25.0, 1.0), ((0.008, 0.002), (0.002, 0.030))),
    "midrange_pullup": ((15.0, 3.0), ((0.012, -0.003), (-0.003, 0.015))),
}


def load_ep() -> EPModel:
    ck = torch.load(os.path.join(ROOT, "models", "v2_ep.pt"), weights_only=False)
    sd = ck["state_dict"]
    m = EPModel(sd["mean"].numpy(), sd["std"].numpy(), ck["three_a"], ck["three_c"])
    m.load_state_dict(sd)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def true_cost(
    mu: tuple[float, float], Q: tuple[tuple[float, float], tuple[float, float]]
) -> QuadCost:
    Q = torch.tensor(Q)
    L = torch.linalg.cholesky(Q)
    b = -(Q @ torch.tensor(mu))
    return QuadCost(b=b.tolist(), L=L.tolist())


def noisy(d: torch.Tensor, k: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    s = np.clip(d[:, 0].numpy() + rng.normal(0, NOISE[0], len(d)), 0, S_MAX)
    t = np.clip(d[:, 1].numpy() + rng.normal(0, NOISE[1], len(d)), 0, k.numpy())
    return torch.tensor(np.stack([s, t], 1), dtype=torch.float32)


def run(
    ep: EPModel, df: pd.DataFrame, noise: bool, rng: np.random.Generator
) -> dict[str, dict[str, Any]]:
    out = {}
    for i, (name, (mu, Q)) in enumerate(SYNTHETIC.items()):
        tr, va, _ = split_player(df, PLAYERS[i], SEED + i)
        half = len(va) // 2
        sets = {"train": tr, "select": va[:half], "check": va[half:]}
        ctx = {k_: torch.from_numpy(context(df.iloc[ix])) for k_, ix in sets.items()}
        kk = {
            k_: torch.tensor(df.iloc[ix]["k"].to_numpy(), dtype=torch.float32)
            for k_, ix in sets.items()
        }
        grids = {k_: ep_grid(ep, ctx[k_]) for k_ in sets}
        truth = true_cost(mu, Q)
        d_true = {k_: solve(grids[k_], kk[k_], truth) for k_ in sets}
        d_obs = {k_: (noisy(d_true[k_], kk[k_], rng) if noise else d_true[k_]) for k_ in sets}
        fits = {}
        for rho in RHOS:
            c, diag = fit_cost(ep, d_obs["train"], ctx["train"], kk["train"], rho, seed=SEED)
            pred = solve(grids["select"], kk["select"], c)
            fits[rho] = (c, diag, float((pred[:, 0] - d_obs["select"][:, 0]).abs().mean()))
        rho = min(RHOS, key=lambda r: fits[r][2])
        c, diag, _ = fits[rho]
        th_hat, th = c.theta(), truth.theta()
        rel = float(np.linalg.norm(th_hat - th) / np.linalg.norm(th))
        p_hat = solve(grids["check"], kk["check"], c)
        ds = float((p_hat[:, 0] - d_true["check"][:, 0]).abs().median())
        dt = float((p_hat[:, 1] - d_true["check"][:, 1]).abs().median())
        out[name] = {
            "contexts_from": PLAYERS[i],
            "n_train": len(tr),
            "rho": rho,
            "theta_true": np.round(th, 5).tolist(),
            "theta_hat": np.round(th_hat, 5).tolist(),
            "relative_theta_error": round(rel, 4),
            "median_abs_diff_s_ft": round(ds, 3),
            "median_abs_diff_t_s": round(dt, 3),
            "true_decisions_train": {
                "s_mean": round(float(d_true["train"][:, 0].mean()), 2),
                "s_sd": round(float(d_true["train"][:, 0].std()), 2),
                "t_mean": round(float(d_true["train"][:, 1].mean()), 2),
                "t_sd": round(float(d_true["train"][:, 1].std()), 2),
            },
            "residuals": {k_: round(v, 6) for k_, v in diag.items()},
            "recovered": bool(rel <= 0.25 and ds <= 1.0 and dt <= 0.25),
        }
        print(
            name,
            "noise" if noise else "no noise",
            {
                k_: out[name][k_]
                for k_ in (
                    "relative_theta_error",
                    "median_abs_diff_s_ft",
                    "median_abs_diff_t_s",
                    "recovered",
                    "rho",
                )
            },
        )
    return out


def main() -> None:
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    res = {
        "command": "python scripts/v2_synthetic.py",
        "noise_sd": NOISE,
        "synthetic_costs": {k: {"mu": v[0], "Q": v[1]} for k, v in SYNTHETIC.items()},
        "with_noise": run(ep, df, True, np.random.default_rng(SEED)),
        "no_noise_diagnostic": run(ep, df, False, np.random.default_rng(SEED)),
    }
    res["criterion_a_passed"] = all(v["recovered"] for v in res["with_noise"].values())
    with open(os.path.join(ROOT, "results", "v2_synthetic.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("criterion (a) passed:", res["criterion_a_passed"])


if __name__ == "__main__":
    main()
