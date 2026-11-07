"""Posterior predictive checks for the four-skill model, summarised for the web.

Four kinds of check, each comparing observed data with replicated data drawn
from the posterior predictive distribution:

1. distributions  - histogram of each outcome vs histograms of replicated datasets
2. pit            - probability integral transform; uniform if calibrated,
                    hump-shaped if predictions are too wide, U-shaped if too narrow
3. groups         - non-penalty goals per 90 by team-strength band, league and
                    season observed: does the model reproduce the team factor?
4. careers        - each player's observed career non-penalty goals vs his
                    predictive interval
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import LEAGUE_ORDER

N_REPLICATES = 40
LO, HI = 0.085, 0.915  # central 83%

OUTCOMES = {
    # name: (label, transform for display, bins)
    "np_shots": ("Non-penalty shots per season", None, np.arange(0, 230, 10)),
    "key_passes": ("Key passes per season", None, np.arange(0, 140, 5)),
    "log_xg_per_shot": ("xG per shot (log scale)", "exp", np.linspace(np.log(0.02), np.log(0.6), 31)),
    "npg": ("Non-penalty goals per season", None, np.arange(0, 42, 2)),
}


def _flat(da) -> np.ndarray:
    """(chain, draw, obs) -> (samples, obs)."""
    v = da.values
    return v.reshape(-1, v.shape[-1])


def _hist(x: np.ndarray, bins: np.ndarray) -> list[int]:
    return np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9), bins=bins)[0].astype(int).tolist()


def _pit(y: np.ndarray, yrep: np.ndarray, discrete: bool, rng) -> np.ndarray:
    below = (yrep < y).mean(axis=0)
    if not discrete:
        return below
    equal = (yrep == y).mean(axis=0)
    return below + rng.uniform(size=y.shape) * equal  # randomised PIT for counts


def _interval(x: np.ndarray) -> dict:
    return {"mean": round(float(x.mean()), 4), "lo83": round(float(np.quantile(x, LO)), 4),
            "hi83": round(float(np.quantile(x, HI)), 4)}


def build_ppc(idata, rows: pd.DataFrame, player_names: list[str], seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    pp, ob = idata.posterior_predictive, idata.observed_data
    n_s = pp.sizes["chain"] * pp.sizes["draw"]
    reps = np.sort(rng.choice(n_s, size=N_REPLICATES, replace=False))
    sub = ((rows.np_shots > 0) & (rows.npxG > 0)).to_numpy()  # rows used by fox and sniper

    out = {"n_replicates": N_REPLICATES, "distributions": {}, "pit": {}, "coverage83": {}}
    pit_bins = np.linspace(0, 1, 21)
    for v, (label, transform, bins) in OUTCOMES.items():
        y, yrep = ob[v].values, _flat(pp[v])
        lo, hi = np.quantile(yrep, [LO, HI], axis=0)
        out["coverage83"][v] = round(float(((y >= lo) & (y <= hi)).mean()), 3)
        edges = np.exp(bins) if transform == "exp" else bins
        out["distributions"][v] = {
            "label": label, "log_x": transform == "exp",
            "edges": [round(float(e), 4) for e in edges],
            "observed": _hist(y, bins),
            "replicates": [_hist(yrep[j], bins) for j in reps],
        }
        pit = _pit(y, yrep, discrete=v != "log_xg_per_shot", rng=rng)
        out["pit"][v] = {"label": label, "counts": np.histogram(pit, bins=pit_bins)[0].astype(int).tolist(),
                         "n": int(len(y))}

    # goals per 90 by group, from the npg replicates (subset rows)
    npg_rep = _flat(pp["npg"])  # (samples, subset rows)
    r = rows[sub].reset_index(drop=True)
    mins = r.minutes.to_numpy()
    groupings = {
        "team_strength": pd.qcut(r.team_strength, 5, labels=["weakest fifth", "2nd", "middle", "4th", "strongest fifth"]),
        "league": r.league.map(dict(zip(LEAGUE_ORDER, ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]))),
        "season_index": r.season_index.clip(upper=8).map(lambda s: f"{s}" if s < 8 else "8+"),
    }
    out["groups"] = {}
    for name, g in groupings.items():
        cats = list(g.cat.categories) if hasattr(g, "cat") else sorted(g.unique(), key=lambda s: (len(s), s))
        rows_out = []
        for c in cats:
            m = (g == c).to_numpy()
            per90 = 90 * npg_rep[:, m].sum(axis=1) / mins[m].sum()
            rows_out.append({"group": str(c), "observed": round(float(90 * r.npg[m].sum() / mins[m].sum()), 4),
                             **_interval(per90), "n": int(m.sum())})
        out["groups"][name] = rows_out

    # career totals per player
    p = r.p_idx.to_numpy()
    n_players = len(player_names)
    tot_rep = np.zeros((npg_rep.shape[0], n_players))
    for j in range(n_players):  # sum each player's rows within each draw
        tot_rep[:, j] = npg_rep[:, p == j].sum(axis=1)
    obs_tot = np.bincount(p, weights=r.npg.to_numpy(), minlength=n_players)
    lo, hi = np.quantile(tot_rep, [LO, HI], axis=0)
    careers = [{"i": j, "name": player_names[j], "observed": int(obs_tot[j]),
                "mean": round(float(tot_rep[:, j].mean()), 2), "lo83": round(float(lo[j]), 1), "hi83": round(float(hi[j]), 1)}
               for j in range(n_players)]
    out["careers"] = careers
    out["career_coverage83"] = round(float(np.mean([lo[j] <= obs_tot[j] <= hi[j] for j in range(n_players)])), 3)
    return out
