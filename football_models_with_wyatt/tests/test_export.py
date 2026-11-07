"""Checks on the exported web files and thinned posterior.

Run from football_models_with_wyatt/:  python -m pytest tests -q
"""

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB, IDATA = ROOT / "web", ROOT / "idata"
SKILLS = ["shoot", "fox", "sniper", "create"]


@pytest.fixture(scope="module")
def web():
    return {p.stem: json.loads(p.read_text()) for p in WEB.glob("*.json")}


def test_all_files_present(web):
    assert {"players", "draws", "curves", "skill_corr", "model_card", "wyatt_guesses", "ppc"} <= set(web)


def test_players_schema_and_order(web):
    players = web["players"]
    assert len(players) == web["model_card"]["data"]["players"]
    assert [p["i"] for p in players] == list(range(len(players)))
    assert len({p["id"] for p in players}) == len(players)
    for p in players:
        assert set(p["skills"]) == set(SKILLS)
        for k in SKILLS:
            s = p["skills"][k]
            assert s["lo83"] <= s["mean"] <= s["hi83"]
            assert 1 <= s["stars"] <= 5
        assert p["SAR"]["lo83"] <= p["SAR"]["mean"] <= p["SAR"]["hi83"]
        assert p["seasons"], p["name"]


def test_sniper_stars_follow_intervals(web):
    for p in web["players"]:
        s = p["skills"]["sniper"]
        if s["lo83"] <= 0 <= s["hi83"]:
            assert s["stars"] == 3, p["name"]
        elif s["lo83"] > 0:
            assert s["stars"] in (4, 5), p["name"]
        else:
            assert s["stars"] in (1, 2), p["name"]


def test_draw_arrays_shape(web):
    d, n = web["draws"], len(web["players"])
    for k in SKILLS:
        assert len(d["alpha"][k]) == n
        assert all(len(row) == d["n_draws"] for row in d["alpha"][k])
    assert len(d["goals90_level"]) == n and len(d["goals90_actual"]) == n
    assert all(len(row) == d["n_draws"] for row in d["goals90_level"])


def test_sar_recomputable_from_web_draws(web):
    """The page recomputes SAR client-side; check that matches the exported summary."""
    level = np.array(web["draws"]["goals90_level"])
    rlp = np.array([p["is_replacement"] for p in web["players"]])
    sar_web = (level - level[rlp].mean(axis=0)).mean(axis=1)
    sar_exported = np.array([p["SAR"]["mean"] for p in web["players"]])
    assert np.abs(sar_web - sar_exported).max() < 0.02
    assert np.corrcoef(sar_web, sar_exported)[0, 1] > 0.999


def test_sar_matches_sar_table(web):
    tab = pd.read_csv(IDATA / "sar_table.csv").set_index("player_id")
    exported = pd.Series({p["id"]: p["SAR"]["mean"] for p in web["players"]})
    assert np.allclose(tab.loc[exported.index, "SAR"], exported, atol=1e-3)


def test_web_draw_skills_match_summaries(web):
    for k in SKILLS:
        draws = np.array(web["draws"]["alpha"][k]).mean(axis=1)
        means = np.array([p["skills"][k]["mean"] for p in web["players"]])
        assert np.corrcoef(draws, means)[0, 1] > 0.99


def test_curves_and_corr(web):
    c = web["curves"]
    assert len(c["season_index"]) == 10
    for k in SKILLS:
        assert len(c[k]["mean"]) == 10
    corr = np.array(web["skill_corr"]["mean"])
    assert corr.shape == (4, 4) and np.allclose(np.diag(corr), 1) and np.allclose(corr, corr.T)


def test_thinned_idata():
    path = IDATA / "sfm_skills.nc"
    assert path.stat().st_size < 50e6
    idata = az.from_netcdf(path)
    assert idata.posterior.sizes["draw"] == 1000
    assert "z" not in idata.posterior and "alpha" in idata.posterior


def test_ppc_structure(web):
    ppc = web["ppc"]
    for v, d in ppc["distributions"].items():
        assert len(d["observed"]) == len(d["edges"]) - 1
        assert len(d["replicates"]) == ppc["n_replicates"]
        assert all(len(r) == len(d["observed"]) for r in d["replicates"])
        assert sum(ppc["pit"][v]["counts"]) == ppc["pit"][v]["n"]
        assert 0.5 < ppc["coverage83"][v] <= 1
    assert len(ppc["careers"]) == len(web["players"])
    for name, rows in ppc["groups"].items():
        for r in rows:
            assert r["lo83"] <= r["mean"] <= r["hi83"], (name, r)
