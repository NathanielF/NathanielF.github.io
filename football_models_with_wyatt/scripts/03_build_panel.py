"""Build the hosted player tables from the Kaggle Understat dump.

Requires data/processed/team_matches.csv (scripts/02_team_matches.py).

Outputs in data/processed/:
    team_name_map.csv             Understat -> football-data team names (also in data/maps/)
    matches.csv                   one row per match, with Understat ids and xG
    shots.parquet                 one row per shot, with team factors
    player_match_events.parquet   player-matches with >= 1 shot or key pass
    player_seasons.csv            complete season totals with minutes
    players.csv                   the forward pool (>= 3,000 minutes, mostly at F)
    forwards_player_match_events.csv   the events table restricted to the pool
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sfm import panel  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

OUT = PROJECT / "data" / "processed"


def main() -> None:
    team_matches = pd.read_csv(OUT / "team_matches.csv", parse_dates=["date"])

    matches = panel.load_matches()
    team_map = panel.build_team_map(matches)
    matches = panel.attach_fd_names(matches, team_map)
    assert matches[["home_team", "away_team"]].notna().all().all(), "unmapped team names"
    tv = panel.team_view(matches, team_matches)
    assert tv.date.notna().all(), "some Understat matches have no football-data counterpart"

    # use football-data's local match date everywhere
    home_dates = tv[tv.h_a == "h"].set_index("match_id").date
    matches["date"] = matches.match_id.map(home_dates)

    shots = panel.load_shots()
    shots = shots.merge(tv[["match_id", "h_a", "team", "opponent", "home", "date", "ppg_diff_pre", "pts_diff_pre"]],
                        on=["match_id", "h_a"], how="left", validate="many_to_one")
    assert shots.team.notna().all()

    seasons = panel.load_player_seasons()
    events = panel.build_player_match_events(shots, seasons, tv, matches)
    seasons = panel.enrich_player_seasons(seasons, events, team_map, team_matches)
    pool = panel.build_player_pool(seasons)

    team_map[["league", "understat", "football_data"]].to_csv(PROJECT / "data" / "maps" / "team_names.csv", index=False)
    matches.drop(columns=["kickoff"]).to_csv(OUT / "matches.csv", index=False, float_format="%.4f")
    shots.drop(columns=["h_a"]).to_parquet(OUT / "shots.parquet", index=False)
    events.to_parquet(OUT / "player_match_events.parquet", index=False)
    seasons.to_csv(OUT / "player_seasons.csv", index=False)
    pool.to_csv(OUT / "players.csv", index=False)
    events[events.player_id.isin(pool.player_id)].to_csv(OUT / "forwards_player_match_events.csv", index=False,
                                                        float_format="%.5f")

    print(f"matches: {len(matches):,}  shots: {len(shots):,}  events: {len(events):,}")
    print(f"player-seasons: {len(seasons):,}  forward pool: {len(pool):,}")


if __name__ == "__main__":
    main()
