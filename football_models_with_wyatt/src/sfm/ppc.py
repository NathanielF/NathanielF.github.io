"""Posterior predictive checks for the four-skill model, summarised for the web.

Per-outcome checks (each likelihood on its own, conditional on its inputs):

1. distributions  - histogram of each outcome vs histograms of replicated datasets
2. pit            - leave-one-out PIT (PSIS-weighted, randomised for counts);
                    uniform if calibrated, hump-shaped if too wide, U-shaped if too narrow
3. coverage       - share of observations inside the 83% interval, in-sample and leave-one-out

"Goals" here are a season's non-penalty goals given the shots he actually took
(the Bernoulli finishing model summed over his shots).

Full-chain checks (goals simulated through every skill: replicated shot counts,
then replicated xG per shot, then replicated goals):

4. groups   - non-penalty goals per 90 by team-strength band, league and season observed
5. careers  - each player's career non-penalty goals vs his predictive interval

The full chain draws goals as Binomial(shots, p) with p set by the replicated
average xG per shot, which ignores the spread of chance quality within a season.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from . import models

N_REPLICATES = 40
LO, HI = 0.085, 0.915
CHUNK = 100

OUTCOMES = {
    "np_shots": ("Non-penalty shots per season", None, np.arange(0, 230, 10)),
    "key_passes": ("Key passes per season", None, np.arange(0, 140, 5)),
    "log_xg_per_shot": ("xG per shot (log scale)", "exp", np.linspace(np.log(0.02), np.log(0.6), 31)),
    "goals": ("Non-penalty goals per season, given his shots", None, np.arange(0, 42, 2)),
}


def _flat_post(post) -> dict:
    out = {}
    for v in post.data_vars:
        out[v] = post[v].stack(s=("chain", "draw")).transpose(..., "s").values
    return out


def _hist(x, bins):
    return np.histogram(np.clip(x, bins[0], bins[-1] - 1e-9), bins=bins)[0].astype(int).tolist()


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def goal_loglik_and_reps(eta_sniper: np.ndarray, shots: pd.DataFrame, rows_with_shots: np.ndarray, rng):
    """Per-row log-likelihood and replicated goals for the Bernoulli finishing model.
    eta_sniper is (rows, draws). Returns (draws, n_rows_with_shots) arrays."""
    r = shots.row.to_numpy()
    lx, y = shots.logit_xg.to_numpy(), shots.is_goal.to_numpy()
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])  # shots are sorted by row
    assert np.array_equal(r[starts], rows_with_shots)
    n_s = eta_sniper.shape[1]
    ll = np.empty((n_s, len(starts)))
    reps = np.empty((n_s, len(starts)), dtype=np.int32)
    for a in range(0, n_s, CHUNK):
        b = min(a + CHUNK, n_s)
        logit_p = lx[:, None] + eta_sniper[r, a:b]
        logp = -np.logaddexp(0, -logit_p)          # log sigmoid
        log1mp = -np.logaddexp(0, logit_p)         # log(1 - sigmoid)
        ll[a:b] = np.add.reduceat(np.where(y[:, None] == 1, logp, log1mp), starts, axis=0).T
        draws = rng.random(logit_p.shape) < np.exp(logp)
        reps[a:b] = np.add.reduceat(draws.astype(np.int32), starts, axis=0).T
    return ll, reps


def _loo_weights(loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-smoothed, normalised leave-one-out weights, (draws, obs), and Pareto k."""
    lw, k = az.psislw(-loglik.T)            # psislw expects (obs, draws)
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return w.T, np.asarray(k)


def _weighted_quantiles(yrep: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    order = np.argsort(yrep, axis=0)
    ys = np.take_along_axis(yrep, order, axis=0)
    cw = np.cumsum(np.take_along_axis(w, order, axis=0), axis=0)
    out = []
    for q in qs:
        idx = (cw < q).sum(axis=0).clip(max=yrep.shape[0] - 1)
        out.append(ys[idx, np.arange(yrep.shape[1])])
    return out


def _pit(y, yrep, w, discrete, rng):
    below = (w * (yrep < y)).sum(axis=0)
    if not discrete:
        return below
    return below + rng.uniform(size=y.shape) * (w * (yrep == y)).sum(axis=0)


def build_ppc(idata, rows: pd.DataFrame, shots: pd.DataFrame, player_names: list[str], seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    post = _flat_post(idata.posterior)
    n_s = post["mu_shoot"].shape[0]
    rows_with_shots = np.unique(shots.row.to_numpy())

    eta = {k: models.eta_rows(post, rows, k) for k in models.SKILLS}
    goal_ll, goal_rep = goal_loglik_and_reps(eta["sniper"], shots, rows_with_shots, rng)
    goal_obs = shots.groupby("row").is_goal.sum().reindex(rows_with_shots).to_numpy()

    data = {}
    for v in ["np_shots", "key_passes", "log_xg_per_shot"]:
        yrep = idata.posterior_predictive[v].values.reshape(n_s, -1)
        ll = idata.log_likelihood[v].values.reshape(n_s, -1)
        data[v] = (idata.observed_data[v].values, yrep, ll)
    data["goals"] = (goal_obs, goal_rep, goal_ll)

    reps = np.sort(rng.choice(n_s, size=N_REPLICATES, replace=False))
    out = {"n_replicates": N_REPLICATES, "distributions": {}, "pit": {}, "coverage83": {}, "pareto_k_over_0_7": {}}
    pit_bins = np.linspace(0, 1, 21)
    for v, (label, transform, bins) in OUTCOMES.items():
        y, yrep, ll = data[v]
        discrete = v != "log_xg_per_shot"
        lo, hi = np.quantile(yrep, [LO, HI], axis=0)
        w, k = _loo_weights(ll)
        llo, lhi = _weighted_quantiles(yrep, w, [LO, HI])
        out["coverage83"][v] = {"in_sample": round(float(((y >= lo) & (y <= hi)).mean()), 3),
                                "loo": round(float(((y >= llo) & (y <= lhi)).mean()), 3)}
        out["pareto_k_over_0_7"][v] = int((k > 0.7).sum())
        edges = np.exp(bins) if transform == "exp" else bins
        out["distributions"][v] = {"label": label, "log_x": transform == "exp",
                                   "edges": [round(float(e), 4) for e in edges],
                                   "observed": _hist(y, bins), "replicates": [_hist(yrep[j], bins) for j in reps]}
        pit = np.clip(_pit(y, yrep, w, discrete, rng), 0, 1)  # weights can overshoot 1 by rounding
        out["pit"][v] = {"label": label, "kind": "leave-one-out",
                         "counts": np.histogram(pit, bins=pit_bins)[0].astype(int).tolist(), "n": int(len(y))}

    # ---- full chain: shots -> xG per shot -> goals --------------------------------
    shots_rep = data["np_shots"][1]                                   # (draws, rows)
    sigma_fox = post["sigma_fox"]
    n = np.maximum(shots_rep, 1)
    log_m = eta["fox"].T + rng.standard_normal(n.shape) * sigma_fox[:, None] / np.sqrt(n)
    p = _sigmoid(models.logit(np.exp(log_m)) + eta["sniper"].T)
    goals_full = rng.binomial(shots_rep, np.clip(p, 0, 1))           # (draws, rows)

    mins = rows.minutes.to_numpy()
    groupings = {
        "team_strength": pd.qcut(rows.ts, 5, labels=["weakest fifth", "2nd", "middle", "4th", "strongest fifth"]),
        "league": rows.league.map(dict(zip(models.LEAGUE_ORDER, ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]))),
        "season_index": rows.season_index.clip(upper=8).map(lambda s: f"{s}" if s < 8 else "8+"),
    }
    out["groups"] = {}
    for name, g in groupings.items():
        cats = list(g.cat.categories) if hasattr(g, "cat") else sorted(g.unique(), key=lambda s: (len(s), s))
        res = []
        for c in cats:
            m = (g == c).to_numpy()
            per90 = 90 * goals_full[:, m].sum(axis=1) / mins[m].sum()
            res.append({"group": str(c), "observed": round(float(90 * rows.npg[m].sum() / mins[m].sum()), 4),
                        "mean": round(float(per90.mean()), 4), "lo83": round(float(np.quantile(per90, LO)), 4),
                        "hi83": round(float(np.quantile(per90, HI)), 4), "n": int(m.sum())})
        out["groups"][name] = res

    pidx = rows.p_idx.to_numpy()
    n_players = len(player_names)
    tot = np.zeros((n_s, n_players))
    for j in range(n_players):
        tot[:, j] = goals_full[:, pidx == j].sum(axis=1)
    obs_tot = np.bincount(pidx, weights=rows.npg.to_numpy(), minlength=n_players)
    lo, hi = np.quantile(tot, [LO, HI], axis=0)
    out["careers"] = [{"i": j, "name": player_names[j], "observed": int(obs_tot[j]), "mean": round(float(tot[:, j].mean()), 2),
                       "lo83": round(float(lo[j]), 1), "hi83": round(float(hi[j]), 1)} for j in range(n_players)]
    out["career_coverage83"] = round(float(np.mean((obs_tot >= lo) & (obs_tot <= hi))), 3)
    return out
