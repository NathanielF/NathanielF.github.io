"""Build the player panels from the Kaggle Understat dump plus football-data factors.

The Kaggle dump (data/understats/<League>/) has every match, every shot and
season-level player totals for 2014/15-2023/24, but **no per-match lineups**.
So we publish three player tables at different grains:

* shots            - one row per shot, with team factors attached.
* player_match_events - one row per player-match in which the player took a
                     shot or made a key pass (NOT every appearance; matches
                     where a player did neither are invisible in this source).
* player_seasons   - one row per player-league-season with full totals and
                     minutes, which *is* complete.

The match-level goals SFM needs every appearance; that waits on Understat
roster pages (see fetch.fetch_understat_match).
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import load_results
from .fetch import LEAGUES, PROJECT, SEASONS

KAGGLE = PROJECT / "data" / "understats"
# our Understat league slug -> Kaggle folder name
KAGGLE_DIRS = {"EPL": "EPL", "La_liga": "La_Liga", "Bundesliga": "Bundesliga", "Serie_A": "Serie_A", "Ligue_1": "Ligue_1"}


def read_kaggle(league: str, table: str) -> pd.DataFrame:
    path = KAGGLE / KAGGLE_DIRS[league] / f"{table}.csv"
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c == "" or c.startswith("Unnamed")])
    for col in ("player", "player_name", "player_assisted", "team_h", "team_a", "h_team", "a_team", "team_title"):
        if col in df:
            df[col] = df[col].map(lambda v: html.unescape(v) if isinstance(v, str) else v)
    df = df.drop(columns=["league"], errors="ignore")
    df.insert(0, "league", league)
    return df


# --------------------------------------------------------------------------
# Matches and the team-name map
# --------------------------------------------------------------------------

def load_matches(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        mi = read_kaggle(lg, "match_info")
        mi = mi[mi.season.isin(seasons)]
        frames.append(
            pd.DataFrame(
                {
                    "match_id": mi.id.astype(int),
                    "league": lg,
                    "season": mi.season.astype(int),
                    "kickoff": pd.to_datetime(mi.date),
                    "home_team_us": mi.team_h,
                    "away_team_us": mi.team_a,
                    "home_goals": mi.h_goals.astype(int),
                    "away_goals": mi.a_goals.astype(int),
                    "home_xg": mi.h_xg.astype(float),
                    "away_xg": mi.a_xg.astype(float),
                }
            )
        )
    m = pd.concat(frames, ignore_index=True)
    m["date"] = m.kickoff.dt.normalize()
    return m


def build_team_map(matches: pd.DataFrame, leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    """Learn Understat -> football-data team names from matches with the same
    league, date and score, and keep the most frequent pairing per team."""
    fd = pd.concat([load_results(lg, s) for lg in leagues for s in seasons], ignore_index=True)
    keys = ["league", "date", "home_goals", "away_goals"]
    j = matches.merge(fd, on=keys)
    pairs = pd.concat(
        [
            j[["league", "home_team_us", "home_team"]].set_axis(["league", "understat", "football_data"], axis=1),
            j[["league", "away_team_us", "away_team"]].set_axis(["league", "understat", "football_data"], axis=1),
        ]
    )
    counts = pairs.value_counts().rename("n").reset_index()
    best = counts.sort_values("n", ascending=False).drop_duplicates(["league", "understat"])
    totals = counts.groupby(["league", "understat"]).n.sum().rename("n_total")
    best = best.merge(totals, on=["league", "understat"])
    best["share"] = (best.n / best.n_total).round(3)
    return best.sort_values(["league", "understat"]).reset_index(drop=True)


def attach_fd_names(matches: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    lut = team_map.set_index(["league", "understat"]).football_data
    m = matches.copy()
    m["home_team"] = [lut.get((lg, t)) for lg, t in zip(m.league, m.home_team_us)]
    m["away_team"] = [lut.get((lg, t)) for lg, t in zip(m.league, m.away_team_us)]
    return m


def team_view(matches: pd.DataFrame, team_matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, side) with the pre-match factors for that side.

    Joined on (league, season, team, opponent, home): each home/away pairing
    happens once per season, which avoids Understat's kickoff times (ahead of
    UK time) pushing late games onto the next calendar date. The date column
    returned is football-data's, i.e. the local match date.
    """
    tm = team_matches.copy()
    tm["date"] = pd.to_datetime(tm.date)
    sides = []
    for side, team_col, opp_col in (("h", "home_team", "away_team"), ("a", "away_team", "home_team")):
        s = matches[["match_id", "league", "season", team_col, opp_col, f"{team_col}_us", f"{opp_col}_us"]]
        s = s.set_axis(["match_id", "league", "season", "team", "opponent", "team_us", "opponent_us"], axis=1)
        s = s.assign(h_a=side, home=int(side == "h"))
        sides.append(s)
    tv = pd.concat(sides, ignore_index=True)
    keys = ["league", "season", "team", "opponent", "home"]
    factor_cols = ["date", "pts_diff_pre", "ppg_diff_pre", "market_strength", "games_pre"]
    tv = tv.merge(tm[keys + factor_cols], on=keys, how="left", validate="one_to_one")
    return tv


# --------------------------------------------------------------------------
# Shots and player-match events
# --------------------------------------------------------------------------

def load_shots(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        sd = read_kaggle(lg, "shot_data")
        sd = sd[sd.season.isin(seasons)]
        frames.append(sd)
    sd = pd.concat(frames, ignore_index=True)
    sd = sd[sd.result != "OwnGoal"]  # own goals are not the listed player's shot
    out = pd.DataFrame(
        {
            "shot_id": sd.id.astype(int),
            "match_id": sd.match_id.astype(int),
            "league": sd.league,
            "season": sd.season.astype(int),
            "minute": sd.minute.astype(int),
            "player_id": sd.player_id.astype(int),
            "player": sd.player,
            "h_a": sd.h_a,
            "result": sd.result,
            "is_goal": (sd.result == "Goal").astype(int),
            "xG": sd.xG.astype(float),
            "situation": sd.situation,
            "is_penalty": (sd.situation == "Penalty").astype(int),
            "shot_type": sd.shotType,
            "X": sd.X.astype(float),
            "Y": sd.Y.astype(float),
            "assister": sd.player_assisted,
            "last_action": sd.lastAction,
        }
    )
    return out.reset_index(drop=True)


def _assister_ids(shots: pd.DataFrame, player_seasons: pd.DataFrame) -> pd.Series:
    """Resolve assister names to Understat player ids within league-season."""
    names = player_seasons[["league", "season", "player_name", "player_id"]].drop_duplicates(
        ["league", "season", "player_name"], keep=False  # drop ambiguous same-name pairs
    )
    lut = names.set_index(["league", "season", "player_name"]).player_id
    return pd.Series(
        [lut.get((lg, s, a)) if isinstance(a, str) else None for lg, s, a in zip(shots.league, shots.season, shots.assister)],
        index=shots.index,
        dtype="Int64",
    )


def build_player_match_events(shots: pd.DataFrame, player_seasons: pd.DataFrame, tv: pd.DataFrame,
                               matches: pd.DataFrame) -> pd.DataFrame:
    s = shots.copy()
    s["np"] = 1 - s.is_penalty
    shooting = s.groupby(["match_id", "player_id", "h_a"], as_index=False).agg(
        player=("player", "first"),
        shots=("shot_id", "size"),
        np_shots=("np", "sum"),
        goals=("is_goal", "sum"),
        np_goals=("is_goal", lambda g: int((g * s.loc[g.index, "np"]).sum())),
        xG=("xG", "sum"),
        npxG=("xG", lambda x: float((x * s.loc[x.index, "np"]).sum())),
        pens=("is_penalty", "sum"),
    )
    s["assister_id"] = _assister_ids(s, player_seasons)
    kp = (
        s.dropna(subset=["assister_id"])
        .groupby(["match_id", "assister_id", "h_a"], as_index=False)
        .agg(key_passes=("shot_id", "size"), xA=("xG", "sum"), assists=("is_goal", "sum"))
        .rename(columns={"assister_id": "player_id"})
    )
    kp["player_id"] = kp.player_id.astype(int)
    ev = shooting.merge(kp, on=["match_id", "player_id", "h_a"], how="outer")
    count_cols = ["shots", "np_shots", "goals", "np_goals", "pens", "key_passes", "assists"]
    ev[count_cols] = ev[count_cols].fillna(0).astype(int)
    ev[["xG", "npxG", "xA"]] = ev[["xG", "npxG", "xA"]].fillna(0.0).round(5)
    names = player_seasons.drop_duplicates("player_id").set_index("player_id").player_name
    ev["player"] = ev.player.fillna(ev.player_id.map(names))

    ev = ev.merge(tv, on=["match_id", "h_a"], how="left")
    ev = ev.merge(matches[["match_id", "kickoff"]], on="match_id", how="left")
    cols = ["league", "season", "match_id", "kickoff", "date", "player_id", "player", "team", "opponent", "home",
            "pts_diff_pre", "ppg_diff_pre", "market_strength", "games_pre"] + count_cols + ["xG", "npxG", "xA"]
    return ev[cols].sort_values(["league", "season", "kickoff", "match_id", "player_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Player seasons and the player pool
# --------------------------------------------------------------------------

def load_player_seasons(leagues=LEAGUES, seasons=SEASONS) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        p = read_kaggle(lg, "player")
        frames.append(p[p.year.isin(seasons)])
    p = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "player_id": p.id.astype(int),
            "player_name": p.player_name,
            "league": p.league,
            "season": p.year.astype(int),
            "team_title": p.team_title,
            "position": p.position,
            "games": p.games.astype(int),
            "minutes": p.time.astype(int),
            "goals": p.goals.astype(int),
            "npg": p.npg.astype(int),
            "xG": p.xG.astype(float).round(4),
            "npxG": p.npxG.astype(float).round(4),
            "shots": p.shots.astype(int),
            "key_passes": p.key_passes.astype(int),
            "assists": p.assists.astype(int),
            "xA": p.xA.astype(float).round(4),
            "xGChain": p.xGChain.astype(float).round(4),
            "xGBuildup": p.xGBuildup.astype(float).round(4),
        }
    )
    return out


def enrich_player_seasons(ps: pd.DataFrame, events: pd.DataFrame, team_map: pd.DataFrame,
                          team_matches: pd.DataFrame) -> pd.DataFrame:
    ps = ps.copy()
    # non-penalty shots come from the shot table
    ev = events.groupby(["league", "season", "player_id"]).agg(np_shots=("np_shots", "sum"),
                                                                  shot_matches=("shots", lambda x: int((x > 0).sum())))
    ps = ps.merge(ev, on=["league", "season", "player_id"], how="left")
    ps[["np_shots", "shot_matches"]] = ps[["np_shots", "shot_matches"]].fillna(0).astype(int)
    ps["zero_shot_games"] = (ps.games - ps.shot_matches).clip(lower=0)

    # season-average team strength: final points per game of the player's team(s)
    tm = team_matches.groupby(["league", "season", "team"]).points.mean().rename("team_ppg")
    lut = team_map.set_index(["league", "understat"]).football_data

    def team_ppg(row):
        vals = [tm.get((row.league, row.season, lut.get((row.league, t.strip())))) for t in str(row.team_title).split(",")]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else np.nan

    ps["team_ppg"] = ps.apply(team_ppg, axis=1)

    # Pre-match strength, as in the paper: the team's average points-per-game gap to
    # its opponents, each gap measured before kickoff (0 for a team's first match).
    pre = team_matches.groupby(["league", "season", "team"]).ppg_diff_pre.mean()

    def team_pre(row):
        vals = [pre.get((row.league, row.season, lut.get((row.league, t.strip())))) for t in str(row.team_title).split(",")]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else np.nan

    ps["team_strength_pre"] = ps.apply(team_pre, axis=1)
    league_avg = team_matches.groupby(["league", "season"]).points.mean().rename("league_ppg")
    ps = ps.merge(league_avg, on=["league", "season"], how="left")
    ps["team_strength"] = (ps.team_ppg - ps.league_ppg).round(4)
    ps = ps.drop(columns="league_ppg")

    # N^S: the number of seasons the player has been observed in, up to and including this one
    first = ps.groupby("player_id").season.transform("min")
    seen = ps[["player_id", "season"]].drop_duplicates().sort_values(["player_id", "season"])
    seen["season_index"] = seen.groupby("player_id").cumcount() + 1
    ps = ps.merge(seen, on=["player_id", "season"], how="left")
    ps["first_season"] = first
    return ps.sort_values(["player_id", "season", "league"]).reset_index(drop=True)


MAPS = PROJECT / "data" / "maps"


def load_wyatt_picks() -> pd.DataFrame:
    """data/maps/wyatt_picks.csv: Understat player names Wyatt wants included
    even below the main minutes threshold (they still need `PICK_MIN_MINUTES`)."""
    path = MAPS / "wyatt_picks.csv"
    return pd.read_csv(path, comment="#") if path.exists() else pd.DataFrame(columns=["player_name"])


def load_display_names() -> dict:
    """data/maps/display_names.csv: overrides for Understat names (e.g. Mbappe-Lottin)."""
    path = MAPS / "display_names.csv"
    if not path.exists():
        return {}
    d = pd.read_csv(path, comment="#")
    return dict(zip(d.player_name, d.display_name))


POOL_MIN_MINUTES = 6000
PICK_MIN_MINUTES = 2500


def build_player_pool(ps: pd.DataFrame, min_minutes: int = POOL_MIN_MINUTES,
                      picks: pd.DataFrame | None = None, display_names: dict | None = None) -> pd.DataFrame:
    """Forwards: players with at least `min_minutes` league minutes in the window,
    the majority of which came in seasons where Understat lists them at F,
    plus Wyatt's picks with at least PICK_MIN_MINUTES (any position)."""
    picks = load_wyatt_picks() if picks is None else picks
    display_names = load_display_names() if display_names is None else display_names
    ps = ps.copy()
    ps["is_fw_season"] = ps.position.fillna("").str.split().map(lambda toks: "F" in toks)
    agg = ps.groupby("player_id").agg(
        player_name=("player_name", "last"),
        minutes=("minutes", "sum"),
        fw_minutes=("minutes", lambda m: int(m[ps.loc[m.index, "is_fw_season"]].sum())),
        games=("games", "sum"),
        goals=("goals", "sum"),
        npg=("npg", "sum"),
        npxG=("npxG", "sum"),
        np_shots=("np_shots", "sum"),
        key_passes=("key_passes", "sum"),
        seasons=("season", "nunique"),
        first_season=("season", "min"),
        last_season=("season", "max"),
        leagues=("league", lambda s: ",".join(sorted(set(s)))),
        teams=("team_title", lambda s: ",".join(dict.fromkeys(t.strip() for v in s for t in str(v).split(",")))),
    )
    agg["fw_share"] = (agg.fw_minutes / agg.minutes).round(3)
    main = (agg.minutes >= min_minutes) & (agg.fw_share >= 0.5)
    picked = agg.player_name.isin(set(picks.player_name)) & (agg.minutes >= PICK_MIN_MINUTES)
    missing = set(picks.player_name) - set(agg.player_name)
    if missing:
        print(f"wyatt_picks not found in Understat names: {sorted(missing)}")
    pool = agg[main | picked].copy()
    pool["pool_reason"] = np.where(main[main | picked], "minutes", "wyatt_pick")
    pool.insert(1, "display_name", pool.player_name.map(lambda n: display_names.get(n, n)))
    per90 = pool.minutes / 90
    pool["npg_p90"] = (pool.npg / per90).round(3)
    pool["npxG_p90"] = (pool.npxG / per90).round(3)
    pool["np_shots_p90"] = (pool.np_shots / per90).round(3)
    pool["key_passes_p90"] = (pool.key_passes / per90).round(3)
    pool["npxG_per_shot"] = (pool.npxG / pool.np_shots).round(4)
    pool["npg_minus_npxG"] = (pool.npg - pool.npxG).round(2)
    pool["npxG"] = pool.npxG.round(3)
    return pool.reset_index().sort_values("npg", ascending=False).reset_index(drop=True)
