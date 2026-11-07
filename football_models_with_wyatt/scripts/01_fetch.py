"""Download (once, with caching) all raw data for the project window.

Usage:
    python scripts/01_fetch.py odds         # football-data.co.uk CSVs
    python scripts/01_fetch.py understat    # league pages, then every match page
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfm import fetch  # noqa: E402


def fetch_odds() -> None:
    for league, div in fetch.LEAGUES.items():
        for season in fetch.SEASONS:
            path = fetch.fetch_football_data(div, season)
            print(f"{league:11s} {season}: {path.name} ({path.stat().st_size:,} bytes)")


def fetch_understat() -> None:
    if not fetch.understat_available():
        sys.exit("Understat is serving its maintenance page; try again later.")
    for league in fetch.LEAGUES:
        for season in fetch.SEASONS:
            data = fetch.fetch_understat_league(league, season)
            dates = data.get("datesData", data.get("dates", []))
            played = [m for m in dates if m.get("isResult")]
            print(f"{league:11s} {season}: {len(played)} played matches")
            for i, match in enumerate(played, 1):
                fetch.fetch_understat_match(match["id"])
                if i % 50 == 0:
                    print(f"    {i}/{len(played)}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "odds"
    {"odds": fetch_odds, "understat": fetch_understat}[what]()
