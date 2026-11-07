"""SAR / PAR table from the fitted four-skill model -> idata/sar_table.csv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az  # noqa: E402
import pandas as pd  # noqa: E402

from sfm import sar  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

if __name__ == "__main__":
    idata = az.from_netcdf(PROJECT / "idata" / "sfm_skills_full.nc")
    rows = pd.read_csv(PROJECT / "idata" / "sfm_skills_rows.csv")
    players = pd.read_csv(PROJECT / "data" / "processed" / "players.csv")
    tab = sar.sar_table(idata.posterior, rows, players)
    tab.round(4).to_csv(PROJECT / "idata" / "sar_table.csv", index=False)
    print(f"replacement pool: {tab.is_replacement.sum()} players, "
          f"level-field goals/90 {tab[tab.is_replacement].goals90_level.min():.3f}-{tab[tab.is_replacement].goals90_level.max():.3f}")
    print(tab.head(12)[["SAR_rank", "player", "SAR", "SAR_lo83", "SAR_hi83", "SAR_minus_PAR", "minutes"]].round(3).to_string(index=False))
