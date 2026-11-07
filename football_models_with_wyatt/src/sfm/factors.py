"""Team-level factors built from football-data.co.uk results and closing odds.

Each match becomes two rows, one from each team's point of view. All
strength measures are "real time": they use only matches played on earlier
dates in the same season, in the spirit of the paper's Section 2.2.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fetch import LEAGUES, RAW, SEASONS, season_code

# Closing 1X2 odds, in order of preference, as (home, draw, away) column triples.
_CLOSING_ODDS = [("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA")]


def load_results(league: str, season: int) -> pd.DataFrame:
    div = LEAGUES[league]
    path = RAW / "football_data" / f"{div}_{season_code(season)}.csv"
    df = pd.read_csv(path, encoding="latin-1").dropna(subset=["HomeTeam"]).copy()
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    odds = pd.DataFrame(index=df.index, columns=["oH", "oD", "oA"], dtype=float)
    for cols in _CLOSING_ODDS:
        if all(c in df for c in cols):
            fill = df[list(cols)].astype(float)
            fill.columns = ["oH", "oD", "oA"]
            odds = odds.fillna(fill)
    inv = 1.0 / odds
    probs = inv.div(inv.sum(axis=1), axis=0)  # proportional de-vigging

    return pd.DataFrame(
        {
            "league": league,
            "season": season,
            "date": df["date"],
            "home_team": df["HomeTeam"].str.strip(),
            "away_team": df["AwayTeam"].str.strip(),
            "home_goals": df["FTHG"].astype(int),
            "away_goals": df["FTAG"].astype(int),
            "p_home": probs["oH"],
            "p_draw": probs["oD"],
            "p_away": probs["oA"],
        }
    )


def _team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "team": matches.home_team, "opponent": matches.away_team, "home": 1,
            "goals_for": matches.home_goals, "goals_against": matches.away_goals,
            "p_win": matches.p_home, "p_loss": matches.p_away,
        }
    )
    away = pd.DataFrame(
        {
            "team": matches.away_team, "opponent": matches.home_team, "home": 0,
            "goals_for": matches.away_goals, "goals_against": matches.home_goals,
            "p_win": matches.p_away, "p_loss": matches.p_home,
        }
    )
    keys = matches[["league", "season", "date", "p_draw"]]
    rows = pd.concat([keys.join(home), keys.join(away)], ignore_index=True)
    rows["points"] = np.select(
        [rows.goals_for > rows.goals_against, rows.goals_for == rows.goals_against], [3, 1], 0
    )
    return rows


def _pre_match_form(rows: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date points and games for each team, from earlier dates only."""
    daily = (
        rows.groupby(["league", "season", "team", "date"], as_index=False)
        .agg(points=("points", "sum"), games=("points", "size"))
        .sort_values("date")
    )
    g = daily.groupby(["league", "season", "team"])
    daily["pts_pre"] = g["points"].cumsum() - daily["points"]
    daily["games_pre"] = g["games"].cumsum() - daily["games"]
    return daily[["league", "season", "team", "date", "pts_pre", "games_pre"]]


def build_team_matches(seasons=SEASONS, leagues=LEAGUES) -> pd.DataFrame:
    matches = pd.concat([load_results(lg, s) for lg in leagues for s in seasons], ignore_index=True)
    rows = _team_rows(matches)
    form = _pre_match_form(rows)
    keys = ["league", "season", "date"]
    rows = rows.merge(form, on=keys + ["team"], how="left")
    opp = form.rename(columns={"team": "opponent", "pts_pre": "opp_pts_pre", "games_pre": "opp_games_pre"})
    rows = rows.merge(opp, on=keys + ["opponent"], how="left")

    # Points per game before kickoff; 0 before a team's first match.
    ppg = rows.pts_pre / rows.games_pre.replace(0, np.nan)
    opp_ppg = rows.opp_pts_pre / rows.opp_games_pre.replace(0, np.nan)
    rows["ppg_diff_pre"] = (ppg.fillna(0) - opp_ppg.fillna(0)).round(4)
    rows["pts_diff_pre"] = rows.pts_pre - rows.opp_pts_pre  # the paper's raw points gap
    rows["market_strength"] = np.log(rows.p_win / rows.p_loss).round(4)

    cols = [
        "league", "season", "date", "team", "opponent", "home", "goals_for", "goals_against",
        "points", "pts_pre", "games_pre", "opp_pts_pre", "opp_games_pre", "pts_diff_pre",
        "ppg_diff_pre", "p_win", "p_draw", "p_loss", "market_strength",
    ]
    out = rows[cols].sort_values(["league", "season", "date", "home"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)
