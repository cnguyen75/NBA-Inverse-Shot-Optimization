"""v3 criteria (b) and (c) on the three real players. Run only if (a) and (a') passed.

    python scripts/v3_real.py       # writes results/v3_real.json

Per player: fit the five-parameter choice model and the EP-only model (T only)
on training shots; build the empirical baseline from training histograms.
Identifiability (a') is reported for the real fits. Validation log-likelihood
is reported for information only (v3 has no settings to choose). The test set
is scored once: log-likelihood per shot with paired bootstrap intervals, MAE
of expected shot distance, and the pre-registered context-shift check (c).
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
from nbashots.v2 import PLAYERS, S_GRID, T_GRID, context, ep_grid, load, split_player  # noqa: E402
from nbashots.v3 import (  # noqa: E402
    THETA,
    ChoiceData,
    cell_index,
    expected_s,
    fit,
    identifiability,
    theta,
    theta_cov,
)


N_BOOT = 1000


def empirical(d_tr: torch.Tensor, k_tr: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    si, ti = cell_index(d_tr, k_tr)
    ps = np.bincount(si.numpy(), minlength=len(S_GRID)) + 1.0
    pt = np.bincount(ti.numpy(), minlength=len(T_GRID)) + 1.0
    return ps / ps.sum(), pt / pt.sum()


def empirical_ll(ps: np.ndarray, pt: np.ndarray, d: torch.Tensor, k: torch.Tensor) -> np.ndarray:
    si, ti = cell_index(d, k)
    cum = np.cumsum(pt)
    tmax = np.floor(k.numpy() / 0.1 + 1e-6).astype(int)
    return np.log(ps[si.numpy()]) + np.log(pt[ti.numpy()]) - np.log(cum[tmax])


def boot(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    n = len(x)
    m = [x[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def boot_diff_groups(y: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% interval of mean(y[g]) - mean(y[~g]), resampling within groups."""
    a, b = y[g], y[~g]
    m = [
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(N_BOOT)
    ]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main() -> None:
    torch.set_num_threads(4)
    df, _ = load(SHOT_LOGS)
    ep = load_ep()
    out = {}
    for i, p in enumerate(PLAYERS):
        rng = np.random.default_rng(SEED + 400 + i)
        sets = dict(zip(("train", "val", "test"), split_player(df, p, SEED + i), strict=False))
        D, K, G = {}, {}, {}
        for name, ix in sets.items():
            D[name] = torch.tensor(df.iloc[ix][["s", "t"]].to_numpy(), dtype=torch.float32)
            K[name] = torch.tensor(df.iloc[ix]["k"].to_numpy(), dtype=torch.float32)
            G[name] = ep_grid(ep, torch.from_numpy(context(df.iloc[ix])))
        data = {n: ChoiceData(G[n], K[n], D[n]) for n in sets}
        w, ll_tr, H, _ = fit(data["train"])
        w_ep, _, _, _ = fit(data["train"], free=[0])
        ps, pt = empirical(D["train"], K["train"])
        cov_th, _ = theta_cov(w, H)
        th, se = theta(w), np.sqrt(np.diag(cov_th))
        ident = identifiability(w, H)
        val = {
            "model": data["val"].stats(w, False)[0] / len(sets["val"]),
            "ep_only": data["val"].stats(w_ep, False)[0] / len(sets["val"]),
            "empirical": float(empirical_ll(ps, pt, D["val"], K["val"]).mean()),
        }
        # ---- test set, scored once
        ll = {
            "model": data["test"].stats(w, False)[3].numpy(),
            "ep_only": data["test"].stats(w_ep, False)[3].numpy(),
            "empirical": empirical_ll(ps, pt, D["test"], K["test"]),
        }
        es = {
            "model": expected_s(ChoiceData(G["test"], K["test"]), w),
            "ep_only": expected_s(ChoiceData(G["test"], K["test"]), w_ep),
            "empirical": np.full(len(sets["test"]), float((ps * S_GRID.double().numpy()).sum())),
        }
        s_true = D["test"][:, 0].numpy()
        test = {
            m: {
                "loglik_per_shot": round(float(v.mean()), 4),
                "loglik_ci95": boot(v, rng),
                "mae_expected_s_ft": round(float(np.abs(es[m] - s_true).mean()), 3),
            }
            for m, v in ll.items()
        }
        better = {}
        for base in ("ep_only", "empirical"):
            ci = boot(ll["model"] - ll[base], rng)
            better[base] = {
                "loglik_diff_ci95": ci,
                "better": bool(ci[0] > 0),
                "mae_diff_ft": round(
                    test["model"]["mae_expected_s_ft"] - test[base]["mae_expected_s_ft"], 3
                ),
            }
        # ---- (c) context shifts in mean shot distance, test set
        te = df.iloc[sets["test"]]
        med_def = float(df.iloc[sets["train"]]["def"].median())
        splits = {
            "defender_above_train_median": (te["def"] > med_def).to_numpy(),
            "shot_clock_at_catch_above_10s": (te["k"] > 10).to_numpy(),
        }
        shifts = {}
        for name, g in splits.items():
            obs = float(s_true[g].mean() - s_true[~g].mean())
            ci = boot_diff_groups(s_true, g, rng)
            pred = {m: round(float(es[m][g].mean() - es[m][~g].mean()), 3) for m in es}
            has = ci[0] > 0 or ci[1] < 0
            shifts[name] = {
                "n_above": int(g.sum()),
                "n_below": int((~g).sum()),
                "observed_diff_ft": round(obs, 3),
                "observed_ci95": ci,
                "predicted_diff_ft": pred,
                "something_to_explain": bool(has),
                "model_explains": bool(
                    has
                    and np.sign(pred["model"]) == np.sign(obs)
                    and ci[0] <= pred["model"] <= ci[1]
                ),
            }
        out[p] = {
            "n": {k_: len(v) for k_, v in sets.items()},
            "theta": {
                n: {
                    "estimate": round(float(th[j]), 4),
                    "ci95": [
                        round(float(th[j] - 1.96 * se[j]), 4),
                        round(float(th[j] + 1.96 * se[j]), 4),
                    ],
                }
                for j, n in enumerate(THETA)
            },
            "ep_only_T": round(float(1 / w_ep[0]), 4),
            "identifiability": ident,
            "validation_loglik_per_shot": {k_: round(v, 4) for k_, v in val.items()},
            "test": test,
            "model_vs_baselines": better,
            "context_shifts": shifts,
        }
        print(
            p,
            json.dumps(
                {
                    "theta": {n: v["estimate"] for n, v in out[p]["theta"].items()},
                    "ident": ident["passed"],
                    "test": test,
                    "better": {b: v["better"] for b, v in better.items()},
                    "shifts": {
                        n: (
                            v["observed_diff_ft"],
                            v["observed_ci95"],
                            v["predicted_diff_ft"]["model"],
                            v["model_explains"],
                        )
                        for n, v in shifts.items()
                    },
                }
            ),
        )
    works = (
        sum(
            all(out[p]["model_vs_baselines"][b]["better"] for b in ("ep_only", "empirical"))
            for p in PLAYERS
        )
        >= 2
    )
    adds = (
        sum(any(v["model_explains"] for v in out[p]["context_shifts"].values()) for p in PLAYERS)
        >= 2
    )
    res = {
        "command": "python scripts/v3_real.py",
        "criterion_b_works": bool(works),
        "criterion_c_adds_beyond_frequencies": bool(adds),
        "players": out,
    }
    with open(os.path.join(ROOT, "results", "v3_real.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("criterion (b):", works, " criterion (c):", adds)


if __name__ == "__main__":
    main()
