"""Aggregate demo data for the project page: shots by distance zone, and factor importance.

    python scripts/export_demo.py OUT.json      # after scripts/evaluate.py

Aggregates only. For each player, every 2014-15 shot in the log is counted by
the notebook's distance zones (the data has no court x/y, so a "shot chart"
here is concentric distance bands). Importance comes from results/results.json
(the chosen model's mean |gradient|, 5 training seeds) with logistic-regression
coefficients alongside.
"""

from datetime import date
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nbashots.data import PLAYERS, ZONE_LABELS, ZONES, load_shots  # noqa: E402


DISPLAY = {
    "lebron james": "LeBron James",
    "stephen curry": "Stephen Curry",
    "james harden": "James Harden",
}


def main() -> None:
    shots = load_shots()
    res = json.load(open(os.path.join(ROOT, "results", "results.json")))
    players = []
    for p in PLAYERS:
        s = shots[shots["player_name"] == p]
        zones = []
        for z in ZONES:
            q = s[s["shot_zone"] == z]
            zones.append(
                {
                    "zone": z,
                    "label": ZONE_LABELS[z],
                    "attempts": int(len(q)),
                    "made": int(q["FGM"].sum()),
                    "fg_pct": round(float(q["FGM"].mean()), 3) if len(q) else None,
                    "three_point_attempts": int((q["PTS_TYPE"] == 3).sum()),
                    "median_defender_ft": round(float(q["CLOSE_DEF_DIST"].median()), 1)
                    if len(q)
                    else None,
                    "share_of_attempts": round(len(q) / len(s), 3),
                }
            )
        r = res["players"][p]
        players.append(
            {
                "id": p.replace(" ", "-"),
                "name": DISPLAY[p],
                "attempts": int(len(s)),
                "made": int(s["FGM"].sum()),
                "fg_pct": round(float(s["FGM"].mean()), 3),
                "zones": zones,
                "importance": {
                    "method": "mean |d score / d input|, normalised to sum to 1; "
                    "mean and range over 5 training seeds",
                    "values": r["importance_chosen_model_5_seeds"],
                },
                "logistic_regression_coefficients": r["logistic_regression_coefficients"],
                "test": {k: v for k, v in r["test"].items()},
                "test_ci95": r["test_ci95"],
            }
        )
    demo = {
        "schema": "nba-shot-decisions/1",
        "about": {
            "season": "2014-15 regular season (NBA shot logs, Kaggle: dansbecker/nba-shot-logs)",
            "zones": "distance bands from the course notebook: 0–4, 4–14, 14–22 and 22+ ft",
            "importance_model": res["chosen_variant"],
            "content": "aggregates only; no shot-level rows",
            "commands": ["python scripts/evaluate.py", "python scripts/export_demo.py demo.json"],
            "generated": date.today().isoformat(),
        },
        "players": players,
    }
    with open(sys.argv[1], "w") as f:
        json.dump(demo, f, indent=1, ensure_ascii=False)
    print(f"{len(players)} players -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
