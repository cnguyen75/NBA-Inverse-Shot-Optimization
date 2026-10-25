"""v3b: redesigned synthetic test for the v3 choice model (README, "v3b").

    python scripts/v3b_synthetic.py     # writes results/v3b_synthetic.json

(a) Well-specified recovery, gating: three synthetic players, one per real
player (that player's training contexts and sample size), T = 0.15 and
alpha = 0.05, band costs solved so predicted band shares match the real
player's training shares within +-0.01. Five seeds each (0-4); decisions are
drawn from the model with no added noise. Passes only if all 15 datasets pass
the unchanged v3a checks.
(b) Robustness, not gating: the same draws with v2's noise added (sd 1.0 ft,
0.25 s), clipped and refitted.
"""

import json
import os
import sys
from typing import Any

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from v2_synthetic import SEED, load_ep  # noqa: E402
from v3_synthetic import profile  # noqa: E402

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
)


T_TRUE, ALPHA_TRUE = 0.15, 0.05
TARGET = {
    "lebron james": [0.171, 0.307, 0.192, 0.330],
    "stephen curry": [0.122, 0.196, 0.124, 0.558],
    "james harden": [0.168, 0.248, 0.156, 0.428],
}
SEEDS = range(5)
NOISE = (1.0, 0.25)


def solve_bands(
    data: ChoiceData, target: list[float], iters: int = 200
) -> tuple[torch.Tensor, np.ndarray]:
    """Band costs (linear form) so the predicted shares match `target`; rim fixed at 0."""
    a, c = 1 / T_TRUE, ALPHA_TRUE / T_TRUE
    b = np.zeros(3)
    tgt = np.array(target)
    for _ in range(iters):
        w = torch.tensor([a, *b, c], dtype=torch.float64)
        sh = band_shares(data, w)
        # raising a band's cost lowers its share; log-ratio step relative to the rim band
        step = np.log(sh[1:] / tgt[1:]) - np.log(sh[0] / tgt[0])
        b += step
        if np.abs(sh - tgt).max() < 0.002:
            break
    return torch.tensor([a, *b, c], dtype=torch.float64), sh


def checks(
    data_fit: ChoiceData, va_data: ChoiceData, w_true: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    w, ll, H, _ = fit(data_fit)
    th_hat, th_true = theta(w), theta(w_true)
    rel = float(np.linalg.norm(th_hat - th_true) / np.linalg.norm(th_true))
    tv = float(0.5 * np.abs(band_shares(va_data, w) - band_shares(va_data, w_true)).sum())
    ident = identifiability(w, H)
    return (
        w,
        H,
        {
            "theta_hat": dict(zip(THETA, np.round(th_hat, 4).tolist())),
            "relative_theta_error": round(rel, 4),
            "band_share_tv": round(tv, 4),
            "identifiability_passed": ident["passed"],
            "max_abs_correlation": ident["max_abs_correlation"],
            "se_linear_form": dict(zip(PARAMS, ident["se"])),
            "se_ok": dict(zip(PARAMS, ident["se_ok"])),
            "passed": bool(rel <= 0.25 and tv <= 0.05 and ident["passed"]),
        },
    )


def main() -> None:
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    out = {}
    for i, p in enumerate(PLAYERS):
        tr, va, _ = split_player(df, p, SEED + i)
        g_tr = ep_grid(ep, torch.from_numpy(context(df.iloc[tr])))
        g_va = ep_grid(ep, torch.from_numpy(context(df.iloc[va])))
        k_tr = torch.tensor(df.iloc[tr]["k"].to_numpy(), dtype=torch.float32)
        k_va = torch.tensor(df.iloc[va]["k"].to_numpy(), dtype=torch.float32)
        base, va_data = ChoiceData(g_tr, k_tr), ChoiceData(g_va, k_va)
        w_true, sh = solve_bands(base, TARGET[p])
        matched = bool(np.abs(sh - np.array(TARGET[p])).max() <= 0.01)
        res = {
            "n_train": len(tr),
            "target_shares": TARGET[p],
            "matched_shares": np.round(sh, 4).tolist(),
            "shares_within_0.01": matched,
            "theta_true": dict(zip(THETA, np.round(theta(w_true), 4).tolist())),
            "well_specified": {},
            "with_v2_noise": {},
        }
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            d = sample(base, w_true, rng)
            w, H, r = checks(ChoiceData(g_tr, k_tr, d), va_data, w_true)
            res["well_specified"][seed] = r
            if seed == 0:
                res["profiles_seed0"] = {
                    PARAMS[j]: profile(ChoiceData(g_tr, k_tr, d), w, H, j) for j in range(5)
                }
            nr = np.random.default_rng(100 + seed)
            s = np.clip(d[:, 0].numpy() + nr.normal(0, NOISE[0], len(d)), 0, S_MAX)
            t = np.clip(d[:, 1].numpy() + nr.normal(0, NOISE[1], len(d)), 0, k_tr.numpy())
            dn = torch.tensor(np.stack([s, t], 1), dtype=torch.float32)
            _, _, rn = checks(ChoiceData(g_tr, k_tr, dn), va_data, w_true)
            res["with_v2_noise"][seed] = rn
            print(
                p,
                seed,
                "well-specified",
                {
                    k: r[k]
                    for k in (
                        "relative_theta_error",
                        "band_share_tv",
                        "identifiability_passed",
                        "passed",
                    )
                },
                "| noise",
                {k: rn[k] for k in ("relative_theta_error", "band_share_tv", "passed")},
                flush=True,
            )
        out[p] = res
    a_pass = all(r["passed"] for v in out.values() for r in v["well_specified"].values()) and all(
        v["shares_within_0.01"] for v in out.values()
    )
    res = {
        "command": "python scripts/v3b_synthetic.py",
        "T_true": T_TRUE,
        "alpha_true": ALPHA_TRUE,
        "seeds": list(SEEDS),
        "players": out,
        "criterion_a_passed": a_pass,
        "robustness_passed_count": sum(
            r["passed"] for v in out.values() for r in v["with_v2_noise"].values()
        ),
        "datasets": 15,
    }
    with open(os.path.join(ROOT, "results", "v3b_synthetic.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(
        "v3b (a) passed:", a_pass, "| robustness passes:", res["robustness_passed_count"], "of 15"
    )


if __name__ == "__main__":
    main()
