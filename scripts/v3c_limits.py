"""v3c band-share limits and pass-rule power, by the rule fixed in the power-check note.

    python scripts/v3c_limits.py     # needs models/v3b_power_draws.npz from scripts/v3b_power.py

Limit per synthetic player: the 99th percentile of the ideal estimator's gap,
rounded up to the next 0.005. Pass rule: at least 18 of 20 datasets per player
pass all checks. Power: probability that the ideal estimator meets the rule
for all three players, with per-dataset pass probability from the same draws.
Written to results/v3c_limits.json before any v3c fit.
"""

import json
import math
import os

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYERS = ["lebron james", "stephen curry", "james harden"]
N_SEEDS, MIN_PASS = 20, 18


def main() -> None:
    z = np.load(os.path.join(ROOT, "models", "v3b_power_draws.npz"))
    out, power = {}, 1.0
    for p in PLAYERS:
        rel, gap, ident = z[f"{p}|rel"], z[f"{p}|gap"], bool(z[f"{p}|ident"])
        limit = math.ceil(np.quantile(gap, 0.99) / 0.005 - 1e-9) * 0.005
        q = float(((rel <= 0.25) & (gap <= limit)).mean()) * ident
        p_rule = sum(
            math.comb(N_SEEDS, k) * q**k * (1 - q) ** (N_SEEDS - k)
            for k in range(MIN_PASS, N_SEEDS + 1)
        )
        power *= p_rule
        out[p] = {
            "gap_q99": round(float(np.quantile(gap, 0.99)), 4),
            "limit": round(limit, 3),
            "ideal_per_dataset_pass": round(q, 4),
            "ideal_meets_18_of_20": round(p_rule, 4),
        }
    res = {
        "command": "python scripts/v3c_limits.py",
        "seeds_per_player": N_SEEDS,
        "pass_rule": f"at least {MIN_PASS} of {N_SEEDS} per player, all three players",
        "players": out,
        "power_ideal_estimator": round(power, 4),
        "power_at_least_0.80": bool(power >= 0.80),
    }
    with open(os.path.join(ROOT, "results", "v3c_limits.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
