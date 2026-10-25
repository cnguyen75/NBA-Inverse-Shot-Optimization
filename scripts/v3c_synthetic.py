"""v3c: the v3b synthetic test with the powered band-share limits (README, "v3c").

    python scripts/v3c_synthetic.py     # writes results/v3c_synthetic.json

Same players, true parameters, fitting and checks as scripts/v3b_synthetic.py
(imported, not copied). Only the band-share limit (results/v3c_limits.json),
the seeds (10-29) and the pass rule (at least 18 of 20 per player) differ.
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
from v3b_synthetic import TARGET, checks, solve_bands  # noqa: E402

from nbashots.data import SHOT_LOGS  # noqa: E402
from nbashots.v2 import PLAYERS, context, ep_grid, load, split_player  # noqa: E402
from nbashots.v3 import THETA, ChoiceData, sample, theta  # noqa: E402


SEEDS = range(10, 30)
MIN_PASS = 18


def main() -> None:
    torch.set_num_threads(4)
    with open(os.path.join(ROOT, "results", "v3c_limits.json")) as f:
        limits = json.load(f)["players"]
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    out = {}
    for i, p in enumerate(PLAYERS):
        lim = limits[p]["limit"]
        tr, va, _ = split_player(df, p, SEED + i)
        g_tr = ep_grid(ep, torch.from_numpy(context(df.iloc[tr])))
        g_va = ep_grid(ep, torch.from_numpy(context(df.iloc[va])))
        k_tr = torch.tensor(df.iloc[tr]["k"].to_numpy(), dtype=torch.float32)
        k_va = torch.tensor(df.iloc[va]["k"].to_numpy(), dtype=torch.float32)
        base, va_data = ChoiceData(g_tr, k_tr), ChoiceData(g_va, k_va)
        w_true, sh = solve_bands(base, TARGET[p])
        rows = {}
        for seed in SEEDS:
            d = sample(base, w_true, np.random.default_rng(seed))
            _, _, r = checks(ChoiceData(g_tr, k_tr, d), va_data, w_true)
            r["passed_v3c"] = bool(
                r["relative_theta_error"] <= 0.25
                and r["band_share_tv"] <= lim
                and r["identifiability_passed"]
            )
            rows[seed] = r
            print(
                p,
                seed,
                {
                    k: r[k]
                    for k in (
                        "relative_theta_error",
                        "band_share_tv",
                        "identifiability_passed",
                        "passed_v3c",
                    )
                },
                flush=True,
            )
        n_pass = sum(r["passed_v3c"] for r in rows.values())
        out[p] = {
            "limit": lim,
            "theta_true": dict(zip(THETA, np.round(theta(w_true), 4).tolist(), strict=False)),
            "matched_shares": np.round(sh, 4).tolist(),
            "datasets": rows,
            "n_passed": n_pass,
            "player_passes": bool(n_pass >= MIN_PASS),
        }
    passed = all(v["player_passes"] for v in out.values())
    res = {
        "command": "python scripts/v3c_synthetic.py",
        "seeds": list(SEEDS),
        "pass_rule": f"at least {MIN_PASS} of 20 per player",
        "players": out,
        "v3c_passed": passed,
    }
    with open(os.path.join(ROOT, "results", "v3c_synthetic.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("v3c passed:", passed, {p: v["n_passed"] for p, v in out.items()})


if __name__ == "__main__":
    main()
