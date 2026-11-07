# Data dictionary

**Window:** seasons 2014/15 to 2023/24 (`season` is the start year), top five European leagues. `league` uses Understat slugs: `EPL`, `La_liga`, `Bundesliga`, `Serie_A`, `Ligue_1`.

**Sources:**
- **Understat** (match, shot and player-season data), taken from the Kaggle dump [mexwell/understat-database](https://www.kaggle.com/datasets/mexwell/understat-database). It is kept locally in `data/understats/`, which is gitignored.
- **[football-data.co.uk](https://www.football-data.co.uk/)**: results and closing odds.
- **Build:** `scripts/02_team_matches.py`, then `scripts/03_build_panel.py`.

> **The key limitation: no lineups.** The Kaggle dump has no per-match lineups or minutes. A player's appearance in which he neither shot nor made a key pass does not appear at match level. Across the forward pool that is **29% of appearances**. Season totals (`player_seasons.csv`) *are* complete. The paper's match-level goals model needs every appearance, so it waits on Understat roster pages. The four-skill model can be built from these tables.

## team_matches.csv

Each match appears twice, once from each team's point of view. All `*_pre` fields use only matches on **earlier dates in the same season**, so they are known before kickoff.

| column | meaning |
|---|---|
| date | local match date |
| team, opponent | football-data.co.uk team names |
| home | 1 if `team` played at home |
| goals_for, goals_against, points | the result from `team`'s side |
| pts_pre, games_pre, opp_pts_pre, opp_games_pre | season-to-date points and games before kickoff |
| pts_diff_pre | `pts_pre - opp_pts_pre`: the paper's points-difference factor |
| ppg_diff_pre | difference in points per game before kickoff (0 for a team with no games yet); robust to postponed matches |
| p_win, p_draw, p_loss | closing 1X2 probabilities, de-vigged proportionally (average closing odds, falling back to Pinnacle and then Bet365 closing odds). Missing for 2 matches. |
| market_strength | `log(p_win / p_loss)`: market-implied strength, for robustness checks only (correlates about 0.7 with the points factors) |

## matches.csv

One row per match: `match_id` (Understat id), `league`, `season`, `date` (football-data's local date; Understat's own timestamps run ahead of UK time and push about 1.5% of late games onto the next day), Understat and football-data team names, goals, and team xG. Understat lacks 3 of football-data's 18,085 matches.

## shots.parquet

One row per shot (453,015). Own goals are excluded.

| column | meaning |
|---|---|
| shot_id, match_id, minute | identifiers and timing |
| player_id, player | the shooter (Understat id and name) |
| team, opponent, home, date | from the joined match |
| result | `Goal`, `SavedShot`, `MissedShots`, `BlockedShot`, `ShotOnPost` |
| is_goal, xG | outcome and Understat expected goals |
| situation, is_penalty | `OpenPlay`, `FromCorner`, `SetPiece`, `DirectFreekick`, `Penalty` |
| shot_type, X, Y | foot or head; pitch coordinates in 0–1 |
| assister, last_action | name of the player who made the key pass, and the action before the shot |
| ppg_diff_pre, pts_diff_pre | the pre-match strength factors for the shooter's team |

## player_match_events.parquet

One row per player-match **in which the player shot or made a key pass** (318,210 rows), aggregated from `shots`. **This is not every appearance** (see the limitation above).

| column | meaning |
|---|---|
| shots, np_shots, pens | all shots, non-penalty shots, penalties taken |
| goals, np_goals | goals, non-penalty goals |
| xG, npxG | expected goals, all and non-penalty |
| key_passes, assists, xA | shots assisted, goals assisted, xG of the shots assisted. Assister names are resolved to ids within league-season; about 0.5% stay unresolved (same-name clashes or spelling) |
| home, pts_diff_pre, ppg_diff_pre, market_strength, games_pre | the pre-match team factors |

`forwards_player_match_events.csv` is the same table restricted to the players in `players.csv`.

## player_seasons.csv

One row per player, league and season (27,025). It is **complete**, including minutes. Totals come from Understat's season table. They reconcile exactly with the shot table for goals and non-penalty goals, and for 99.8% of rows for shots.

| column | meaning |
|---|---|
| team_title | club(s); comma-separated after a mid-season move |
| position | the Understat positions played (e.g. `F M S`; S = substitute) |
| games, minutes | appearances and minutes |
| goals, npg, xG, npxG, shots, key_passes, assists, xA, xGChain, xGBuildup | Understat season totals |
| np_shots | non-penalty shots (from the shot table) |
| shot_matches, zero_shot_games | matches with at least one shot, and `games - shot_matches` |
| team_ppg, team_strength | the team's points per game that season, and that minus the league average |
| team_strength_pre | the team's pre-match points-per-game gap to each opponent, averaged over its season; the team factor the model uses (the paper's factor, known before each kickoff) |
| season_index | N^S: the number of seasons the player has been observed **in this window** up to and including this one. This is left-censored, so Messi's 2014/15 is season 1. |
| first_season | first season observed in the window |

## players.csv

The player pool has 593 players:

- **587 forwards** (`pool_reason = minutes`): at least 6,000 league minutes in the window, with at least half of them in seasons where Understat lists the player at F.
- **6 of Wyatt's picks** (`pool_reason = wyatt_pick`): players named in `data/maps/wyatt_picks.csv` with at least 2,500 minutes, in any position. Currently Palmer, Kvaratskhelia, Núñez, Højlund, Bellingham and Musiala. With fewer minutes, their skill estimates will be noticeably less certain.

`display_name` applies the overrides in `data/maps/display_names.csv` (for example, Understat's "Kylian Mbappe-Lottin" becomes "Kylian Mbappé"). The file has career totals plus per-90 rates (`npg_p90`, `npxG_p90`, `np_shots_p90`, `key_passes_p90`), shot quality (`npxG_per_shot`), `npg_minus_npxG` and `fw_share`.
