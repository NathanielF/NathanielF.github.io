"""Write data/model_table.csv: the one table reproduce.ipynb needs.

One row per player-season used by the model, with the season totals, the
pre-match team factor and the three finishing statistics (goals, xG and
sum p(1-p) over the season's non-penalty shots). About 0.5 MB, so the
notebook runs without the Kaggle dump.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sfm import models  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

PROC = PROJECT / "data" / "processed"

if __name__ == "__main__":
    d, coords, _ = models.prepare(pd.read_csv(PROC / "player_seasons.csv"), pd.read_csv(PROC / "players.csv"),
                                  pd.read_parquet(PROC / "shots.parquet"))
    names = dict(zip(coords["player_id"], coords["player"]))
    out = pd.DataFrame({
        "player_id": d.player_id, "player": d.player_id.map(names), "league": d.league, "season": d.season,
        "team": d.team_title, "minutes": d.minutes, "season_index": d.season_index,
        "team_strength": d.ts.round(4), "np_shots": d.np_shots, "npxG": d.npxG.round(4), "npg": d.npg,
        "key_passes": d.key_passes, "fin_goals": d.fin_goals.astype(int), "fin_xg": d.fin_xg.round(5),
        "fin_var": d.fin_var.round(5),
    })
    path = PROJECT / "data" / "model_table.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path.relative_to(PROJECT)}: {len(out):,} rows, {path.stat().st_size / 1e6:.2f} MB")
