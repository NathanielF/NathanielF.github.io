"""Polite, cached fetchers for the two public data sources.

Every response is written to data/raw/ the first time it is fetched and read
from disk afterwards, so the remote sites are hit at most once per resource.

* football-data.co.uk: one results+odds CSV per league-season.
* Understat: one league page per league-season (the fixture list with match ids)
  and one page per match (player rosters and shot-level data).
"""

from __future__ import annotations

import codecs
import json
import re
import time
from pathlib import Path

import requests

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / "data" / "raw"

USER_AGENT = "football-models-with-wyatt (personal research; slow, cached scrape)"
SLEEP_SECONDS = 2.0

SEASONS = list(range(2014, 2024))  # start years: 2014/15 ... 2023/24 (Kaggle Understat coverage)

# Understat league slug -> football-data.co.uk division code
LEAGUES = {
    "EPL": "E0",
    "La_liga": "SP1",
    "Bundesliga": "D1",
    "Serie_A": "I1",
    "Ligue_1": "F1",
}


class SourceUnavailable(RuntimeError):
    """The remote site is serving a maintenance or error page."""


_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_last_request = 0.0


def _get(url: str, **headers) -> requests.Response:
    global _last_request
    wait = SLEEP_SECONDS - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = _session.get(url, headers=headers, timeout=30)
    _last_request = time.monotonic()
    resp.raise_for_status()
    return resp


def _cached(path: Path, fetch) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


# --------------------------------------------------------------------------
# football-data.co.uk
# --------------------------------------------------------------------------

def season_code(start_year: int) -> str:
    """2024 -> '2425'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def fetch_football_data(div: str, start_year: int) -> Path:
    path = RAW / "football_data" / f"{div}_{season_code(start_year)}.csv"
    url = f"https://www.football-data.co.uk/mmz4281/{season_code(start_year)}/{div}.csv"
    _cached(path, lambda: _get(url).content)
    return path


# --------------------------------------------------------------------------
# Understat
# --------------------------------------------------------------------------

_MAINTENANCE_MARKER = "We will come back soon"


def _understat_json(url: str, cache: Path) -> dict:
    """Return the JSON payload for an Understat page.

    Understat has served data in two ways: JSON embedded in the HTML as
    `var name = JSON.parse('<hex-escaped>')`, and (more recently) AJAX
    endpoints that return JSON directly. We cache the raw response and parse
    whichever shape comes back. A maintenance page raises SourceUnavailable
    and is never cached.
    """
    if cache.exists() and cache.stat().st_size > 0:
        raw = cache.read_text(encoding="utf-8")
    else:
        raw = _get(url, **{"X-Requested-With": "XMLHttpRequest"}).text
        if _MAINTENANCE_MARKER in raw:
            raise SourceUnavailable(f"Understat is in maintenance mode ({url})")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    return parse_understat(raw)


def parse_understat(raw: str) -> dict:
    stripped = raw.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    out = {}
    for name, payload in re.findall(r"var\s+(\w+)\s*=\s*JSON\.parse\('(.*?)'\)", raw, re.S):
        out[name] = json.loads(codecs.decode(payload, "unicode_escape"))
    if not out:
        raise ValueError("No Understat JSON found in response")
    return out


def fetch_understat_league(league: str, start_year: int) -> dict:
    cache = RAW / "understat" / "league" / f"{league}_{start_year}.html"
    return _understat_json(f"https://understat.com/league/{league}/{start_year}", cache)


def fetch_understat_match(match_id: int | str) -> dict:
    cache = RAW / "understat" / "match" / f"{match_id}.html"
    return _understat_json(f"https://understat.com/match/{match_id}", cache)


def understat_available() -> bool:
    try:
        text = _get("https://understat.com/").text
    except requests.RequestException:
        return False
    return _MAINTENANCE_MARKER not in text
