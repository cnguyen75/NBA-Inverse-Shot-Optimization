"""Held-out comparison: majority class vs logistic regression vs the cost network.

    python scripts/evaluate.py            # writes results/results.json

Per player (LeBron James, Stephen Curry, James Harden):
  1. 800 of the player's shots (sampled with seed 42, as in the notebook), each
     with up to 3 generated alternatives (nbashots/alternatives.py).
  2. Split by shot, 60/20/20 (train/validation/test): a real shot and its
     alternatives are always in the same split. The notebook split rows at
     random, so near-copies of a validation shot were in training.
  3. Numeric features standardised with training-set statistics only.
  4. Model variants (nbashots/model.py: "repo", "bce", "kkt_fixed") are fitted
     on train with early stopping on validation. The variant with the lowest
     mean validation log loss across the three players is chosen. Nothing
     else is tuned.
  5. The test set is scored once, for: always "not chosen" (majority class),
     logistic regression on the same 12 features, the repo's model, and the
     chosen variant.

Metrics on test rows: accuracy (threshold 0.5), ROC AUC, log loss, and
top-1 within a shot (does the real shot get the highest score among its
options? chance = mean of 1/options). 95% intervals: percentile bootstrap
over test shots, 1,000 resamples. Feature importance (the notebook's mean
|gradient|) is averaged over 5 training seeds on train/validation only.
"""

from datetime import date
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nbashots.alternatives import AlternativeGenerator  # noqa: E402
from nbashots.data import PLAYERS, SHOT_LOGS, load_shots  # noqa: E402
from nbashots.model import (  # noqa: E402
    FEATURES,
    NUMERIC,
    feature_matrix,
    gradient_importance,
    predict_logit,
    train,
)


SEED = 42
VARIANTS = ("repo", "bce", "kkt_fixed")
N_BOOT = 1000
IMPORTANCE_SEEDS = 5


def split_groups(n_groups: int, rng: np.random.Generator) -> tuple[set[int], set[int], set[int]]:
    g = rng.permutation(n_groups)
    a, b = int(0.6 * n_groups), int(0.8 * n_groups)
    return set(g[:a]), set(g[a:b]), set(g[b:])


def metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    top, chance = [], []
    for g in np.unique(groups):
        m = groups == g
        top.append(float(y[m][np.argmax(p[m])] == 1))
        chance.append(1 / m.sum())
    return {
        "accuracy": float(((p >= 0.5) == y).mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top1_within_shot": float(np.mean(top)),
        "top1_chance": float(np.mean(chance)),
    }


def bootstrap(
    y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, rng: np.random.Generator
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Percentile intervals for each model's metrics and paired differences vs logistic regression."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    draws = {k: [] for k in preds}
    diffs = {k: [] for k in preds if k != "logistic_regression"}
    for _ in range(N_BOOT):
        pick = rng.choice(ug, len(ug), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        new_g = np.concatenate([np.full(len(idx[g]), i) for i, g in enumerate(pick)])
        m = {k: metrics(y[rows], p[rows], new_g) for k, p in preds.items()}
        for k in preds:
            draws[k].append(m[k])
        for k in diffs:
            diffs[k].append(
                {
                    s: m[k][s] - m["logistic_regression"][s]
                    for s in ("roc_auc", "top1_within_shot", "log_loss")
                }
            )

    def ci(rows: list[dict[str, float]], key: str) -> list[float]:
        v = np.array([r[key] for r in rows], float)
        v = v[~np.isnan(v)]
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    out = {
        k: {s: ci(draws[k], s) for s in ("accuracy", "roc_auc", "log_loss", "top1_within_shot")}
        for k in preds
    }
    out_d = {
        k: {s: ci(diffs[k], s) for s in ("roc_auc", "top1_within_shot", "log_loss")} for k in diffs
    }
    return out, out_d


def majority_ci(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Bootstrap interval for the accuracy of always predicting "not chosen"."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    acc = [
        1 - y[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])].mean()
        for _ in range(N_BOOT)
    ]
    return [round(float(np.percentile(acc, 2.5)), 4), round(float(np.percentile(acc, 97.5)), 4)]


def r4(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 4) for k, v in d.items()}


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(1)
    shots = load_shots()
    gen = AlternativeGenerator(shots, seed=SEED)
    per, val_scores = {}, {v: [] for v in VARIANTS}
    for pi, player in enumerate(PLAYERS):
        ps = shots[shots["player_name"] == player].sample(n=800, random_state=42)
        alts = gen.for_shots(ps)
        rng = np.random.default_rng(SEED + pi)
        tr, va, te = split_groups(alts["group_id"].nunique(), rng)
        part = {
            k: alts[alts["group_id"].isin(s)].reset_index(drop=True)
            for k, s in (("train", tr), ("val", va), ("test", te))
        }
        mean = part["train"][NUMERIC].mean().to_numpy()
        std = part["train"][NUMERIC].std().replace(0, 1).to_numpy()
        X = {k: feature_matrix(d, mean, std) for k, d in part.items()}
        Y = {k: d["IS_CHOSEN"].to_numpy() for k, d in part.items()}
        R = {k: d[NUMERIC].to_numpy(np.float32) for k, d in part.items()}
        G = {k: d["group_id"].to_numpy() for k, d in part.items()}
        models, hist, val = {}, {}, {}
        for v in VARIANTS:
            m, h = train(
                X["train"],
                Y["train"],
                R["train"],
                X["val"],
                Y["val"],
                R["val"],
                variant=v,
                seed=SEED,
            )
            models[v], hist[v] = m, h
            pv = 1 / (1 + np.exp(-predict_logit(m, X["val"])))
            val[v] = r4(metrics(Y["val"], pv, G["val"]))
            val_scores[v].append(val[v]["log_loss"])
        lr = LogisticRegression(max_iter=1000).fit(X["train"], Y["train"])
        val["logistic_regression"] = r4(
            metrics(Y["val"], lr.predict_proba(X["val"])[:, 1], G["val"])
        )
        out_of_support = (
            part["test"]
            .query("IS_CHOSEN == 1")
            .eval("CLOSE_DEF_DIST > 10 or SHOT_DIST > 35")
            .mean()
        )
        per[player] = dict(
            part=part,
            X=X,
            Y=Y,
            G=G,
            models=models,
            lr=lr,
            val=val,
            R=R,
            hist={
                v: {
                    "epochs": len(h["val_bce"]),
                    "final_val_bce": round(h["val_bce"][-1], 4),
                    "min_val_bce": round(min(h["val_bce"]), 4),
                    "final_val_comp_slackness": round(h["val_comp_slackness"][-1], 2),
                }
                for v, h in hist.items()
            },
            lr_coef=dict(zip(FEATURES, lr.coef_[0].round(4).tolist())),
            counts={
                k: {
                    "shots": int(d["group_id"].nunique()),
                    "rows": len(d),
                    "real_share": round(float(d["IS_CHOSEN"].mean()), 4),
                }
                for k, d in part.items()
            },
            real_test_shots_outside_generator_range=round(float(out_of_support), 4),
        )
        print(
            f"[{player}] validation log loss: "
            + ", ".join(
                f"{v} {val[v]['log_loss']:.4f}" for v in VARIANTS + ("logistic_regression",)
            )
        )

    chosen = min(VARIANTS, key=lambda v: np.mean(val_scores[v]))
    print(f"chosen on validation: {chosen}")

    # ---- test set: scored once, here
    results = {}
    for pi, player in enumerate(PLAYERS):
        d = per[player]
        X, Y, G = d["X"], d["Y"], d["G"]
        preds = {
            "majority_class": np.full(len(Y["test"]), 1e-7),
            "logistic_regression": d["lr"].predict_proba(X["test"])[:, 1],
            "model_repo": 1 / (1 + np.exp(-predict_logit(d["models"]["repo"], X["test"]))),
        }
        if chosen != "repo":
            preds[f"model_{chosen}"] = 1 / (
                1 + np.exp(-predict_logit(d["models"][chosen], X["test"]))
            )
        test = {k: r4(metrics(Y["test"], p, G["test"])) for k, p in preds.items()}
        test["majority_class"]["log_loss"] = (
            None  # a constant prediction has no meaningful log loss
        )
        test["majority_class"]["roc_auc"] = 0.5
        test["majority_class"]["top1_within_shot"] = None
        ci, diff = bootstrap(
            Y["test"],
            {k: v for k, v in preds.items() if k != "majority_class"},
            G["test"],
            np.random.default_rng(SEED + 100 + pi),
        )
        ci["majority_class"] = {
            "accuracy": majority_ci(Y["test"], G["test"], np.random.default_rng(SEED + 200 + pi))
        }
        # importance: chosen variant, 5 training seeds, train+val rows only
        imps = []
        for s in range(IMPORTANCE_SEEDS):
            m, _ = train(
                X["train"],
                Y["train"],
                d["R"]["train"],
                X["val"],
                Y["val"],
                d["R"]["val"],
                variant=chosen,
                seed=s,
            )
            imps.append(gradient_importance(m, np.concatenate([X["train"], X["val"]])))
        imp = {
            f: {
                "mean": round(float(np.mean([i[f] for i in imps])), 4),
                "min": round(float(np.min([i[f] for i in imps])), 4),
                "max": round(float(np.max([i[f] for i in imps])), 4),
            }
            for f in FEATURES
        }
        results[player] = {
            "counts": d["counts"],
            "validation": d["val"],
            "training": d["hist"],
            "test": test,
            "test_ci95": ci,
            "test_diff_vs_logistic_ci95": diff,
            "real_test_shots_outside_generator_range": d["real_test_shots_outside_generator_range"],
            "logistic_regression_coefficients": d["lr_coef"],
            "importance_chosen_model_5_seeds": imp,
        }
    commit = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out = {
        "command": "python scripts/evaluate.py",
        "git_commit": commit,
        "run_date": date.today().isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "seed": SEED,
        "data": os.path.basename(SHOT_LOGS),
        "chosen_variant": chosen,
        "validation_mean_log_loss": {v: round(float(np.mean(s)), 4) for v, s in val_scores.items()},
        "selection_rule": "lowest mean validation log loss across the three players",
        "runtime_seconds": round(time.time() - t0, 1),
        "players": results,
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    for p, r in results.items():
        print(
            p,
            {k: (v["accuracy"], v["roc_auc"], v["top1_within_shot"]) for k, v in r["test"].items()},
        )


if __name__ == "__main__":
    main()
