"""v2 step 1: fit and check the expected-points model, then freeze it.

    python scripts/v2_ep.py        # writes results/v2_ep.json and models/v2_ep.pt (ignored)

Training shots: all other players' shots except a held-out 10%, plus the three
players' training split. Held-out check: that 10% plus the three players'
validation split. The three players' test shots are never used here.
Pre-registered check: every decile of predicted P(make) within 0.03 of the
observed rate, expected calibration error <= 0.02, log loss below a logistic
regression on distance alone.
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nbashots.data import SHOT_LOGS  # noqa: E402
from nbashots.v2 import PLAYERS, context, fit_ep, load, split_player  # noqa: E402


SEED = 42


def splits(df):
    out = {}
    for i, p in enumerate(PLAYERS):
        out[p] = split_player(df, p, SEED + i)
    others = np.flatnonzero(~df["player_name"].isin(PLAYERS).to_numpy())
    others = np.random.default_rng(SEED).permutation(others)
    n = int(0.1 * len(others))
    return out, others[n:], others[:n]


def main():
    torch.set_num_threads(4)
    df, n0 = load(SHOT_LOGS)
    sp, oth_tr, oth_ho = splits(df)
    tr = np.concatenate([oth_tr] + [sp[p][0] for p in PLAYERS])
    ho = np.concatenate([oth_ho] + [sp[p][1] for p in PLAYERS])
    m, best = fit_ep(df.iloc[tr], df.iloc[ho], seed=SEED)
    h = df.iloc[ho]
    with torch.no_grad():
        p = torch.sigmoid(
            m.logit(
                torch.tensor(h["s"].to_numpy(), dtype=torch.float32),
                torch.tensor(h["t"].to_numpy(), dtype=torch.float32),
                torch.from_numpy(context(h)),
            )
        ).numpy()
    y = h["FGM"].to_numpy()
    q = np.quantile(p, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
    dec = [
        {
            "decile": int(b),
            "n": int((bins == b).sum()),
            "predicted": round(float(p[bins == b].mean()), 4),
            "observed": round(float(y[bins == b].mean()), 4),
        }
        for b in range(10)
    ]
    ece = float(sum(d["n"] * abs(d["predicted"] - d["observed"]) for d in dec) / len(p))
    base = LogisticRegression().fit(df.iloc[tr][["s"]], df.iloc[tr]["FGM"])
    ll_base = log_loss(y, base.predict_proba(h[["s"]])[:, 1])
    ll = log_loss(y, p)
    passed = (
        all(abs(d["predicted"] - d["observed"]) <= 0.03 for d in dec)
        and ece <= 0.02
        and ll < ll_base
    )
    out = {
        "command": "python scripts/v2_ep.py",
        "shots_after_filters": len(df),
        "shots_raw": n0,
        "per_player_after_filters": {p: int((df["player_name"] == p).sum()) for p in PLAYERS},
        "train_shots": len(tr),
        "heldout_shots": len(ho),
        "three_point_curve": {
            "slope_per_ft": round(m.three_a, 4),
            "midpoint_ft": round(m.three_c, 3),
        },
        "heldout_log_loss": round(ll, 4),
        "distance_only_log_loss": round(ll_base, 4),
        "ece": round(ece, 4),
        "deciles": dec,
        "check_passed": bool(passed),
    }
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    torch.save(
        {"state_dict": m.state_dict(), "three_a": m.three_a, "three_c": m.three_c},
        os.path.join(ROOT, "models", "v2_ep.pt"),
    )
    with open(os.path.join(ROOT, "results", "v2_ep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "deciles"}, indent=1))
    for d in dec:
        print(d)


if __name__ == "__main__":
    main()
