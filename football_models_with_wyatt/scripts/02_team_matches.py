"""Build data/processed/team_matches.csv from the cached football-data CSVs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfm.factors import build_team_matches  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

if __name__ == "__main__":
    tm = build_team_matches()
    out = PROJECT / "data" / "processed" / "team_matches.csv"
    tm.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out.relative_to(PROJECT)}: {len(tm):,} rows")
