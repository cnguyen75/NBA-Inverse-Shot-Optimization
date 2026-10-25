"""Counterfactual "shots not taken" around each real shot.

A port of the notebook's ShotAlternativeGenerator with a seeded generator.
For each real shot it draws up to `n_alternatives` perturbed shots:
  - shot distance: 70% from the same distance zone, 30% from a zone drawn by
    its frequency; normal around the zone's mean, clipped to [0, 35] ft;
  - defender distance: 70% near the real value, else around the zone's mean,
    clipped to [0, 10] ft;
  - shot clock: real value plus N(0, 2) s (N(0, 1) in the last 3 minutes of
    the 4th quarter or overtime), clipped to [0, 24] and to the game clock;
  - running FG%: real value plus N(0, 0.02);
  - dribbles: real value plus N(0, 1), at least 0.
A draw closer than 0.2 (scaled distance) to an existing option is rejected,
and at most 3 x n_alternatives draws are tried, so some shots get fewer.
The real shot has IS_CHOSEN = 1, its alternatives 0, sharing a group_id.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


SCALE = {"SHOT_DIST": 35.0, "CLOSE_DEF_DIST": 10.0, "SHOT_CLOCK": 24.0, "player_fg_pct": 1.0}


def _dist(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    return np.sqrt(sum(((a[k] - b[k]) / s) ** 2 for k, s in SCALE.items()))


class AlternativeGenerator:
    def __init__(self, shots: pd.DataFrame, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        z = shots.groupby("shot_zone")
        self.zone = {
            k: {
                "mean": g["SHOT_DIST"].mean(),
                "std": g["SHOT_DIST"].std(),
                "freq": len(g) / len(shots),
                "def_mean": g["CLOSE_DEF_DIST"].mean(),
                "def_std": g["CLOSE_DEF_DIST"].std(),
            }
            for k, g in z
        }
        self.stats = {"alternatives": 0, "rejected": 0}

    def _one(self, shot: pd.Series, clutch: bool, game_clock: float) -> dict[str, Any]:
        r = self.rng
        zones = list(self.zone)
        zone = (
            r.choice(zones, p=[self.zone[k]["freq"] for k in zones])
            if r.random() < 0.3
            else shot["shot_zone"]
        )
        zs = self.zone[zone]
        dist = float(np.clip(r.normal(zs["mean"], zs["std"]), 0, 35))
        if r.random() < 0.7:
            base, sd = shot["CLOSE_DEF_DIST"], zs["def_std"] * 0.5
        else:
            base, sd = zs["def_mean"], zs["def_std"]
        def_dist = float(np.clip(r.normal(base, sd), 0, 10))
        clock = float(np.clip(shot["SHOT_CLOCK"] + r.normal(0, 1 if clutch else 2), 0, 24))
        clock = min(clock, game_clock)
        return {
            "SHOT_DIST": dist,
            "CLOSE_DEF_DIST": def_dist,
            "SHOT_CLOCK": clock,
            "DRIBBLES": max(0, int(shot["DRIBBLES"] + r.normal(0, 1))),
            "player_fg_pct": float(np.clip(shot["player_fg_pct"] + r.normal(0, 0.02), 0, 1)),
            # the notebook kept the drawn zone label even when the clipped
            # distance falls outside it; kept for fidelity
            "shot_zone": zone,
        }

    def for_shots(
        self, shots: pd.DataFrame, n_alternatives: int = 3, min_difference: float = 0.2
    ) -> pd.DataFrame:
        rows = []
        for gid, (_, s) in enumerate(shots.iterrows()):
            real = {
                k: s[k]
                for k in (
                    "SHOT_DIST",
                    "CLOSE_DEF_DIST",
                    "SHOT_CLOCK",
                    "DRIBBLES",
                    "player_fg_pct",
                    "shot_zone",
                )
            }
            opts = [real]
            clutch = s["PERIOD"] >= 4 and s["GAME_CLOCK_SEC"] < 180
            for _ in range(n_alternatives * 3):
                if len(opts) >= n_alternatives + 1:
                    break
                alt = self._one(s, clutch, s["GAME_CLOCK_SEC"])
                if any(_dist(alt, o) < min_difference for o in opts):
                    self.stats["rejected"] += 1
                    continue
                opts.append(alt)
                self.stats["alternatives"] += 1
            for i, o in enumerate(opts):
                rows.append(
                    {
                        **o,
                        "game_stage": s["game_stage"],
                        "IS_CHOSEN": int(i == 0),
                        "group_id": gid,
                        "player_name": s["player_name"],
                    }
                )
        return pd.DataFrame(rows)
