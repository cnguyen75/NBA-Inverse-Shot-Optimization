"""Load the 2014-15 shot logs and add the features the model uses.

The data is not in this repository. Download it once (free Kaggle account):

    kaggle datasets download dansbecker/nba-shot-logs --unzip -p data/

and point SHOT_LOGS at it (default: data/shot_logs.csv).
"""

import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOT_LOGS = os.environ.get("SHOT_LOGS", os.path.join(ROOT, "data", "shot_logs.csv"))
SHOT_LOGS_SHA256 = "e9c7fcd71e6f81ade108841449fe1bccda072ed98fae2b5dc86a3b9f29fe5c7c"
PLAYERS = ["lebron james", "stephen curry", "james harden"]

# The notebook's distance zones (feet). The data has shot distance only, no
# court x/y, so these are distance bands, not left/right/corner zones.
ZONE_BINS = [-np.inf, 4, 14, 22, np.inf]
ZONES = ["restricted_area", "mid_range", "long_range", "deep_three"]
ZONE_LABELS = {
    "restricted_area": "0–4 ft",
    "mid_range": "4–14 ft",
    "long_range": "14–22 ft",
    "deep_three": "22+ ft",
}


def clock_to_sec(s: str | float) -> float:
    try:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    except (AttributeError, ValueError):
        return np.nan


def game_stage(period: int) -> str:
    return "early" if period in (1, 2) else ("mid" if period == 3 else "late")


def load_shots(path: str = SHOT_LOGS, players: list[str] = PLAYERS) -> pd.DataFrame:
    """Shot logs for `players`, with the derived columns.

    Differences from the course notebook (both were bugs there):
    - shots are ordered by game, period and seconds left; the notebook sorted
      the "M:SS" clock as text, so "9:59" came before "11:00";
    - the running field-goal percentage is shifted within each player-game;
      the notebook shifted across the whole table, so a game's first shot
      inherited the previous game's makes.
    """
    df = pd.read_csv(path)
    df = df[df["player_name"].isin(players)].copy()
    df["GAME_CLOCK_SEC"] = df["GAME_CLOCK"].map(clock_to_sec)
    df = df.sort_values(
        ["player_name", "GAME_ID", "PERIOD", "GAME_CLOCK_SEC"], ascending=[True, True, True, False]
    ).reset_index(drop=True)
    g = df.groupby(["player_name", "GAME_ID"])
    df["cum_fg_attempts"] = g.cumcount()
    df["cum_fg_made"] = g["FGM"].cumsum() - df["FGM"]
    df["player_fg_pct"] = np.where(
        df["cum_fg_attempts"] > 0, df["cum_fg_made"] / df["cum_fg_attempts"].clip(lower=1), 0.0
    )
    # As in the notebook: a missing shot clock (mostly when the game clock is
    # under 24 s and the shot clock is off) is set to 14 s.
    df["SHOT_CLOCK"] = pd.to_numeric(df["SHOT_CLOCK"], errors="coerce").fillna(14.0)
    df["game_stage"] = df["PERIOD"].map(game_stage)
    df["shot_zone"] = pd.cut(df["SHOT_DIST"], bins=ZONE_BINS, labels=ZONES).astype(str)
    return df
