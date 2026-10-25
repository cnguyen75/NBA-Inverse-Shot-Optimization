"""The cost network and its training.

FEATURES are the notebook's 12 inputs: five numeric (standardised with
training-set statistics), four distance-zone flags and three game-stage flags.

Loss variants:
  "repo"      the notebook's loss, unchanged: BCE + 0.1 x (stationarity +
              complementary slackness + primal feasibility), where
              - the constraints are applied to the *standardised* features
                (so "shot clock >= 0" means "at least the average shot clock",
                and "defender distance <= 10" is in standard deviations);
              - "complementary slackness" is sum_j lambda_j * mean(g_j), a
                signed quantity. Since g_j < 0 for most rows, minimising it
                pushes lambda_j up and the term towards minus infinity: it is
                a reward, not a violation;
              - "stationarity" is the gradient of the batch-mean cost with
                respect to the inputs, which pushes the cost to be flat.
              Training stops early on validation BCE; the final epoch's
              weights are kept (as in the notebook).
  "bce"       BCE only; the best validation epoch's weights are restored.
  "kkt_fixed" BCE + 0.1 x (mean |lambda_j g_j| + mean relu(g_j)) with g_j in
              real units (seconds and feet), best weights restored.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


NUMERIC = ["SHOT_DIST", "CLOSE_DEF_DIST", "SHOT_CLOCK", "DRIBBLES", "player_fg_pct"]
FEATURES = [
    "Shot Distance",
    "Defender Distance",
    "Shot Clock",
    "Dribbles",
    "Player FG%",
    "Restricted Area",
    "Mid Range",
    "Long Range",
    "Deep Three",
    "Early Game",
    "Mid Game",
    "Late Game",
]


def feature_matrix(df: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    num = (df[NUMERIC].to_numpy(float) - mean) / std
    zone = np.stack(
        [
            (df["shot_zone"] == z).to_numpy(float)
            for z in ("restricted_area", "mid_range", "long_range", "deep_three")
        ],
        1,
    )
    stage = np.stack([(df["game_stage"] == s).to_numpy(float) for s in ("early", "mid", "late")], 1)
    return np.concatenate([num, zone, stage], 1).astype(np.float32)


class KKTCostFunction(nn.Module):
    """The notebook's network: 12 -> 64 -> 32 -> 16 -> 1, LeakyReLU(0.2),
    BatchNorm, Dropout 0.3/0.2/0.2, three softplus dual variables."""

    def __init__(self, input_dim: int = 12, n_constraints: int = 3) -> None:
        super().__init__()
        layers, prev = [], input_dim
        for i, h in enumerate([64, 32, 16]):
            layers += [
                nn.Linear(prev, h),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(h),
                nn.Dropout(0.3 if i == 0 else 0.2),
            ]
            prev = h
        last = nn.Linear(prev, 1)
        nn.init.xavier_uniform_(last.weight, gain=0.01)
        self.network = nn.Sequential(*layers, last)
        self.dual_vars = nn.Parameter(torch.zeros(n_constraints))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def duals(self) -> torch.Tensor:
        return nn.functional.softplus(self.dual_vars)


def _repo_penalties(model: KKTCostFunction, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The notebook's get_kkt_violations, unchanged."""
    x = x.detach().clone().requires_grad_(True)
    out = model(x)
    g = [-x[:, 2], x[:, 1] - 10.0, x[:, 0] - 35.0]
    lam = model.duals()
    lagr = out.mean() + sum(lam[i] * g[i].mean() for i in range(3))
    grad = torch.autograd.grad(lagr, x, create_graph=True)[0]
    return {
        "stationarity": grad.norm(dim=1).mean(),
        "comp_slackness": sum(lam[i] * g[i].mean() for i in range(3)),
        "primal_feasibility": sum(torch.relu(gi.mean()) for gi in g),
    }


def _fixed_penalties(model: KKTCostFunction, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Complementary slackness and primal feasibility as violations, in real units."""
    g = [-raw[:, 2], raw[:, 1] - 10.0, raw[:, 0] - 35.0]
    lam = model.duals()
    return {
        "comp_slackness": sum((lam[i] * g[i]).abs().mean() for i in range(3)),
        "primal_feasibility": sum(torch.relu(gi).mean() for gi in g),
    }


def train(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    raw_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    raw_va: np.ndarray,
    variant: str = "repo",
    seed: int = 0,
    lr: float = 1e-3,
    epochs: int = 200,
    batch: int = 32,
    patience: int = 20,
    kkt_weight: float = 0.1,
) -> tuple[KKTCostFunction, dict[str, list[float]]]:
    torch.manual_seed(seed)
    model = KKTCostFunction()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()
    X, Y, R = map(torch.from_numpy, (x_tr, y_tr.astype(np.float32), raw_tr.astype(np.float32)))
    Xv, Yv, Rv = map(torch.from_numpy, (x_va, y_va.astype(np.float32), raw_va.astype(np.float32)))
    gen = torch.Generator().manual_seed(seed)
    hist = {"train_bce": [], "val_bce": [], "val_comp_slackness": []}
    best, best_state, bad = float("inf"), None, 0
    for _ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=gen)
        tot = n = 0
        for i in range(0, len(X) - batch + 1, batch):  # drop_last, as in the notebook
            idx = perm[i : i + batch]
            out = model(X[idx])
            loss = bce(out, Y[idx])
            if variant == "repo":
                p = _repo_penalties(model, X[idx])
                loss = loss + kkt_weight * (
                    p["stationarity"] + p["comp_slackness"] + p["primal_feasibility"]
                )
            elif variant == "kkt_fixed":
                p = _fixed_penalties(model, R[idx])
                loss = loss + kkt_weight * (p["comp_slackness"] + p["primal_feasibility"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += bce(out.detach(), Y[idx]).item()
            n += 1
        model.eval()
        with torch.no_grad():
            vb = bce(model(Xv), Yv).item()
        cs = (
            _repo_penalties(model, Xv)["comp_slackness"].item()
            if variant == "repo"
            else _fixed_penalties(model, Rv)["comp_slackness"].item()
        )
        hist["train_bce"].append(tot / n)
        hist["val_bce"].append(vb)
        hist["val_comp_slackness"].append(cs)
        if vb < best:
            best, bad = vb, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if variant != "repo" and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist


def predict_logit(model: KKTCostFunction, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def gradient_importance(model: KKTCostFunction, x: np.ndarray) -> dict[str, float]:
    """The notebook's importance: mean |d cost / d input| per feature, normalised to sum 1."""
    model.eval()
    xt = torch.from_numpy(x).requires_grad_(True)
    out = model(xt).mean()
    g = torch.autograd.grad(out, xt)[0].abs().mean(0).numpy()
    g = g / (g.sum() + 1e-10)
    return dict(zip(FEATURES, g.round(4).tolist(), strict=False))
