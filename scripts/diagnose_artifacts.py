"""Is the network learning player preference, or how the alternatives were generated?

    python scripts/diagnose_artifacts.py      # writes results/artifacts.json

Validation split only (same splits and seeds as scripts/evaluate.py); the test
set is not used.

1. Label/distance mismatch: a generated alternative keeps the zone it drew
   even when its clipped distance lands in another zone (as in the notebook).
   Real shots never mismatch. Counts the share of rows that do.
2. Open-shot share: how often the closest defender is more than 6 ft away,
   for real shots vs generated alternatives (all generated rows; no model).
3. Ablations of the chosen network (plain BCE), refitted on train and scored
   on validation, with logistic regression for reference:
   - all 12 features;
   - without the four zone flags (distance stays);
   - without zone flags, with the zone recomputed from distance (consistent labels).
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from evaluate import SEED, metrics, split_groups  # noqa: E402
from nbashots.alternatives import AlternativeGenerator  # noqa: E402
from nbashots.data import PLAYERS, ZONE_BINS, ZONES, load_shots  # noqa: E402
from nbashots.model import NUMERIC, feature_matrix, predict_logit, train  # noqa: E402

import pandas as pd  # noqa: E402


def consistent_zone(d):
    return pd.cut(d, bins=ZONE_BINS, labels=ZONES).astype(str)


def main():
    shots = load_shots()
    gen = AlternativeGenerator(shots, seed=SEED)
    out = {}
    for pi, player in enumerate(PLAYERS):
        ps = shots[shots["player_name"] == player].sample(n=800, random_state=42)
        alts = gen.for_shots(ps)
        rng = np.random.default_rng(SEED + pi)
        tr, va, _ = split_groups(alts["group_id"].nunique(), rng)
        alts["zone_mismatch"] = alts["shot_zone"] != consistent_zone(alts["SHOT_DIST"])
        mism = alts.groupby("IS_CHOSEN")["zone_mismatch"].mean().round(4).to_dict()
        open6 = (alts["CLOSE_DEF_DIST"] > 6).groupby(alts["IS_CHOSEN"]).mean().round(4).to_dict()
        part = {k: alts[alts["group_id"].isin(s)].reset_index(drop=True)
                for k, s in (("train", tr), ("val", va))}
        mean = part["train"][NUMERIC].mean().to_numpy()
        std = part["train"][NUMERIC].std().replace(0, 1).to_numpy()
        res = {}
        for name, prep, cols in (
                ("all features", lambda d: d, slice(None)),
                ("without zone flags", lambda d: d, [0, 1, 2, 3, 4, 9, 10, 11]),
                ("zone recomputed from distance",
                 lambda d: d.assign(shot_zone=consistent_zone(d["SHOT_DIST"])), slice(None))):
            X = {k: feature_matrix(prep(d), mean, std)[:, cols] for k, d in part.items()}
            Y = {k: d["IS_CHOSEN"].to_numpy() for k, d in part.items()}
            R = {k: d[NUMERIC].to_numpy(np.float32) for k, d in part.items()}
            G = part["val"]["group_id"].to_numpy()
            import nbashots.model as M
            # input width follows the column subset
            orig = M.KKTCostFunction.__init__.__defaults__
            M.KKTCostFunction.__init__.__defaults__ = (X["train"].shape[1], 3)
            try:
                m, _ = train(X["train"], Y["train"], R["train"], X["val"], Y["val"], R["val"],
                             variant="bce", seed=SEED)
            finally:
                M.KKTCostFunction.__init__.__defaults__ = orig
            pv = 1 / (1 + np.exp(-predict_logit(m, X["val"])))
            lr = LogisticRegression(max_iter=1000).fit(X["train"], Y["train"])
            res[name] = {"network": {k: round(v, 4) for k, v in metrics(Y["val"], pv, G).items()},
                         "logistic_regression_auc": round(float(roc_auc_score(
                             Y["val"], lr.predict_proba(X["val"])[:, 1])), 4)}
        out[player] = {"zone_label_mismatch_share": {"real_shots": mism.get(1, 0.0),
                                                     "alternatives": mism.get(0, 0.0)},
                       "defender_over_6ft_share": {"real_shots": open6.get(1), "alternatives": open6.get(0)},
                       "validation": res}
        print(player, out[player]["zone_label_mismatch_share"], out[player]["defender_over_6ft_share"],
              {k: (v["network"]["roc_auc"], v["network"]["top1_within_shot"]) for k, v in res.items()})
    with open(os.path.join(ROOT, "results", "artifacts.json"), "w") as f:
        json.dump({"command": "python scripts/diagnose_artifacts.py",
                   "split": "validation only; test not used", "players": out}, f, indent=1)


if __name__ == "__main__":
    main()
