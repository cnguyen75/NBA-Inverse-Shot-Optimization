"""Power check of the v3b gate (README, "Power check of the v3b gate").

    python scripts/v3b_power.py      # writes results/v3b_power.json

Ideal estimator: w_hat ~ N(w*, I(w*)^-1), with I(w*) the exact Fisher
information over each synthetic player's training contexts (it does not
depend on observed choices in this model). No data is drawn and no model is
fitted. The exact draws are also saved to models/v3b_power_draws.npz (ignored) for
the v3c limit rule. Primary: 2,000 exact draws per player (seed 2026) through the model's
band shares on validation contexts, scored as v3b scores them. Cross-check:
the delta method (linearised shares), 100,000 draws.
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
from nbashots.v2 import PLAYERS, context, ep_grid, load, split_player  # noqa: E402
from nbashots.v3 import ChoiceData, band_shares, identifiability, theta  # noqa: E402
from v2_synthetic import SEED, load_ep  # noqa: E402
from v3b_synthetic import TARGET, solve_bands  # noqa: E402

N_EXACT, N_LINEAR, MC_SEED = 2000, 100_000, 2026


def main():
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    rng = np.random.default_rng(MC_SEED)
    out, draws = {}, {}
    for i, p in enumerate(PLAYERS):
        tr, va, _ = split_player(df, p, SEED + i)
        g_tr = ep_grid(ep, torch.from_numpy(context(df.iloc[tr])))
        g_va = ep_grid(ep, torch.from_numpy(context(df.iloc[va])))
        k_tr = torch.tensor(df.iloc[tr]["k"].to_numpy(), dtype=torch.float32)
        k_va = torch.tensor(df.iloc[va]["k"].to_numpy(), dtype=torch.float32)
        base, va_data = ChoiceData(g_tr, k_tr), ChoiceData(g_va, k_va)
        w_true, _ = solve_bands(base, TARGET[p])                  # same w* as v3b
        _, _, info, _ = base.stats(w_true)                        # exact I(w*), no choices needed
        cov = np.linalg.inv(info.numpy())
        ident = identifiability(w_true, info)                     # evaluated at w*
        th_true = theta(w_true)
        sh_true = band_shares(va_data, w_true)
        L = np.linalg.cholesky(cov)
        # primary: exact Monte Carlo
        rel, gap = np.empty(N_EXACT), np.empty(N_EXACT)
        for j in range(N_EXACT):
            w = w_true.numpy() + L @ rng.standard_normal(5)
            rel[j] = np.linalg.norm(theta(torch.tensor(w)) - th_true) / np.linalg.norm(th_true)
            gap[j] = 0.5 * np.abs(band_shares(va_data, torch.tensor(w)) - sh_true).sum()
        draws[p] = (rel.copy(), gap.copy(), bool(ident["passed"]))
        ok = (rel <= 0.25) & (gap <= 0.05)
        p_ds = float(ok.mean()) * float(ident["passed"])
        # cross-check: delta method on the band shares
        eps = 1e-4
        J = np.stack([(band_shares(va_data, w_true + eps * torch.eye(5, dtype=torch.float64)[k]) - sh_true) / eps
                      for k in range(5)], 1)                      # (4, 5)
        z = rng.standard_normal((N_LINEAR, 5)) @ L.T
        gap_lin = 0.5 * np.abs(z @ J.T).sum(1)
        wl = w_true.numpy() + z
        th_l = np.stack([1 / wl[:, 0]] + [wl[:, k] / wl[:, 0] for k in range(1, 5)], 1)
        rel_lin = np.linalg.norm(th_l - th_true, axis=1) / np.linalg.norm(th_true)
        p_lin = float(((rel_lin <= 0.25) & (gap_lin <= 0.05)).mean()) * float(ident["passed"])
        se = np.sqrt(p_ds * (1 - p_ds) / N_EXACT)
        out[p] = {"n_train": len(tr), "n_val": len(va),
                  "identifiability_at_truth": {k: ident[k] for k in ("passed", "max_abs_correlation", "se", "estimate")},
                  "p_dataset_exact": round(p_ds, 4), "p_dataset_exact_ci95": [round(max(0, p_ds - 1.96 * se), 4),
                                                                              round(min(1, p_ds + 1.96 * se), 4)],
                  "p_theta_ok": round(float((rel <= 0.25).mean()), 4),
                  "p_gap_ok": round(float((gap <= 0.05).mean()), 4),
                  "gap_quantiles": {q: round(float(np.quantile(gap, q / 100)), 4) for q in (50, 90, 95, 99)},
                  "theta_error_quantiles": {q: round(float(np.quantile(rel, q / 100)), 4) for q in (50, 90, 99)},
                  "p_dataset_linearised": round(p_lin, 4),
                  "gap_quantiles_linearised": {q: round(float(np.quantile(gap_lin, q / 100)), 4) for q in (50, 90, 95, 99)}}
        print(p, json.dumps({k: out[p][k] for k in ("p_dataset_exact", "p_gap_ok", "p_theta_ok", "gap_quantiles",
                                                      "p_dataset_linearised")}), "ident", ident["passed"], flush=True)
    P = float(np.prod([out[p]["p_dataset_exact"] ** 5 for p in PLAYERS]))
    P_lo = float(np.prod([out[p]["p_dataset_exact_ci95"][0] ** 5 for p in PLAYERS]))
    P_hi = float(np.prod([out[p]["p_dataset_exact_ci95"][1] ** 5 for p in PLAYERS]))
    P_lin = float(np.prod([out[p]["p_dataset_linearised"] ** 5 for p in PLAYERS]))
    res = {"command": "python scripts/v3b_power.py", "players": out,
           "P_ideal_passes_v3b_gate": round(P, 4), "P_ci95_from_mc": [round(P_lo, 4), round(P_hi, 4)],
           "P_linearised": round(P_lin, 4),
           "decision": "limit fair: v3b stands, stop" if P >= 0.80 else "gate underpowered: run Step 2 (v3c)"}
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    np.savez(os.path.join(ROOT, "models", "v3b_power_draws.npz"),
             **{f"{p}|{k}": v for p, (r, g, idok) in draws.items()
                for k, v in (("rel", r), ("gap", g), ("ident", np.array(idok)))})
    with open(os.path.join(ROOT, "results", "v3b_power.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("P_ideal_passes_v3b_gate", "P_ci95_from_mc", "P_linearised", "decision")}))


if __name__ == "__main__":
    main()
