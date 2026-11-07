"""The four-skill Soccer Factor Model (season level), in PyMC.

Each player p has a correlated skill vector (shoot, fox, sniper, create).
For player-season row r (league l, pre-match team strength TS, seasons
observed s) and skill k:

    eta_k[r] = mu_k + league_k[l] + beta_k * TS[r] + gamma_k * TSbar[p]
               + h_k(s[r]) + alpha[p, k]

    shooting   np_shots           ~ NegBinomial(exp(log(minutes/90) + eta_shoot), phi_shoot)
    fox        log(npxG/np_shots)  ~ Normal(eta_fox, sigma_fox / sqrt(np_shots))
    sniper     goal on each shot   ~ Bernoulli(logit^-1(logit(xG_shot) + eta_sniper)),
               fitted through a second-order expansion in eta_sniper (see below)
    creating   key_passes         ~ NegBinomial(exp(log(minutes/90) + eta_create), phi_create)

TS is the team's season average of its pre-match points-per-game gap to its
opponents (the paper's factor, known before each kickoff). TSbar is the
player's minutes-weighted average TS across his seasons: a Mundlak term. It
absorbs the tendency of better forwards to play for better teams, so beta is
identified from changes in team strength within a player's career.

A player's skill is theta = alpha + gamma * (TSbar - mean): everything about
him that does not change with the team he plays for in a given season.

Finishing approximation. For a season with shots i (xG p_i) and goals G, the
exact Bernoulli log-likelihood in eta = eta_sniper is

    sum_i [ y_i (logit p_i + eta) - log(1 + exp(logit p_i + eta)) ].

Expanding to second order around eta = 0 gives, up to a constant,

    (G - sum_i p_i) * eta - 0.5 * sum_i p_i (1 - p_i) * eta^2,

a Gaussian likelihood for eta with centre (G - xG) / V and variance 1 / V,
where V = sum_i p_i (1 - p_i) is the variance of the season's goals given its
chances. Three numbers per season replace ~180,000 shot terms. The posterior
predictive checks still use the exact Bernoulli.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

SKILLS = ["shoot", "fox", "sniper", "create"]
LEAGUE_ORDER = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
MIN_ROW_MINUTES = 270
SEASON_GRID = np.arange(1, 11, dtype=float)
SEASON_CENTRE = 5.5
TS_COL = "team_strength_pre"

# prior centres for the intercepts (log scale; sniper is a log-odds shift on xG)
_MU_PRIOR = {"shoot": np.log(2.2), "fox": np.log(0.12), "sniper": 0.0, "create": np.log(1.3)}


def logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def prepare(player_seasons: pd.DataFrame, players: pd.DataFrame, shots: pd.DataFrame | None = None):
    """Rows used by the model, coordinates and (optionally) the matched shots."""
    d = player_seasons[player_seasons.player_id.isin(players.player_id)]
    d = d[d.minutes >= MIN_ROW_MINUTES].copy()
    order = players.sort_values("player_id")
    names = dict(zip(order.player_id, order.display_name))
    pid = sorted(d.player_id.unique())
    d["p_idx"] = d.player_id.map({p: i for i, p in enumerate(pid)})
    d["l_idx"] = d.league.map({lg: i for i, lg in enumerate(LEAGUE_ORDER)})
    d["exposure"] = d.minutes / 90.0
    d["ts"] = d[TS_COL]
    d = d.reset_index(drop=True)
    # Mundlak: each player's minutes-weighted mean team strength, centred
    w = d.groupby("p_idx").apply(lambda g: np.average(g.ts, weights=g.minutes), include_groups=False)
    tsbar = w.reindex(range(len(pid))).to_numpy()
    d["tsbar"] = tsbar[d.p_idx] - tsbar.mean()
    coords = {
        "player": [names[p] for p in pid],
        "player_id": pid,
        "league": LEAGUE_ORDER,
        "skill": SKILLS,
        "skill_bis": SKILLS,
        "season": SEASON_GRID.astype(int).tolist(),
        "row": np.arange(len(d)),
    }
    out_shots = None
    if shots is not None:
        s = shots[shots.is_penalty == 0][["player_id", "league", "season", "xG", "is_goal"]]
        key = d[["player_id", "league", "season"]].assign(row=np.arange(len(d)))
        s = s.merge(key, on=["player_id", "league", "season"], how="inner")
        out_shots = pd.DataFrame({"row": s.row.to_numpy(), "logit_xg": logit(s.xG.to_numpy()),
                                  "is_goal": s.is_goal.to_numpy().astype(int)}).sort_values("row").reset_index(drop=True)
        d = d.join(finishing_stats(out_shots, len(d)))
    return d, coords, out_shots


def finishing_stats(shots: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Per-row sufficient statistics for the finishing approximation."""
    p = 1 / (1 + np.exp(-shots.logit_xg.to_numpy()))
    g = pd.DataFrame({"row": shots.row, "fin_goals": shots.is_goal, "fin_xg": p, "fin_var": p * (1 - p)})
    return g.groupby("row")[["fin_goals", "fin_xg", "fin_var"]].sum().reindex(range(n_rows), fill_value=0.0)


def _hsgp_curve(name: str, s_centered: np.ndarray, m: int = 8, c: float = 2.0):
    ell = pm.InverseGamma(f"ell_{name}", alpha=5, beta=20)
    eta = pm.Exponential(f"amp_{name}", lam=1 / 0.15)
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov)
    grid = SEASON_GRID - SEASON_CENTRE
    x = np.concatenate([s_centered, grid])[:, None]
    phi, sqrt_psd = gp.prior_linearized(X=x)
    # centre each basis function over the data rows so the curve's level
    # cannot trade off against mu_k
    phi = phi - phi[: len(s_centered)].mean(axis=0, keepdims=True)
    beta = pm.Normal(f"hsgp_w_{name}", 0, 1, shape=m)
    f = phi @ (beta * sqrt_psd)
    n = len(s_centered)
    pm.Deterministic(f"curve_{name}", f[n:], dims="season")
    return f[:n]


def build_skill_model(d: pd.DataFrame, coords: dict, shots: pd.DataFrame | None = None) -> pm.Model:
    p, l = d.p_idx.to_numpy(), d.l_idx.to_numpy()
    ts = d.ts.to_numpy()
    tsbar_row = d.tsbar.to_numpy()
    tsbar_player = d.groupby("p_idx").tsbar.first().reindex(range(len(coords["player"]))).to_numpy()
    s_c = d.season_index.to_numpy(dtype=float) - SEASON_CENTRE
    log_exp = np.log(d.exposure.to_numpy())

    m_c = {k: v for k, v in coords.items() if k != "player_id"}
    with pm.Model(coords=m_c) as model:
        chol, corr, sds = pm.LKJCholeskyCov(
            "chol", n=4, eta=2.0, sd_dist=pm.Exponential.dist(3.0), compute_corr=True
        )
        pm.Deterministic("alpha_corr", corr, dims=("skill", "skill_bis"))
        pm.Deterministic("alpha_sd", sds, dims="skill")
        z = pm.Normal("z", 0, 1, dims=("player", "skill"))
        alpha = pm.Deterministic("alpha", z @ chol.T, dims=("player", "skill"))
        gamma = pm.Normal("gamma", 0, 0.5, dims="skill")
        pm.Deterministic("skill_theta", alpha + tsbar_player[:, None] * gamma[None, :], dims=("player", "skill"))

        eta = {}
        for k, skill in enumerate(SKILLS):
            mu = pm.Normal(f"mu_{skill}", _MU_PRIOR[skill], 0.5)
            lg = pm.ZeroSumNormal(f"league_{skill}", sigma=0.2, dims="league")
            beta = pm.Normal(f"beta_{skill}", 0, 0.5)
            h = _hsgp_curve(skill, s_c)
            eta[skill] = mu + lg[l] + beta * ts + gamma[k] * tsbar_row + h + alpha[p, k]

        pm.NegativeBinomial(
            "np_shots", mu=pt.exp(log_exp + eta["shoot"]), alpha=pm.Gamma("phi_shoot", 2, 0.1),
            observed=d.np_shots.to_numpy(), dims="row",
        )
        pm.NegativeBinomial(
            "key_passes", mu=pt.exp(log_exp + eta["create"]), alpha=pm.Gamma("phi_create", 2, 0.1),
            observed=d.key_passes.to_numpy(), dims="row",
        )
        idx_q = np.flatnonzero((d.np_shots.to_numpy() > 0) & (d.npxG.to_numpy() > 0))
        n_q = d.np_shots.to_numpy()[idx_q]
        sigma_fox = pm.HalfNormal("sigma_fox", 1.0)
        pm.Normal(
            "log_xg_per_shot", mu=eta["fox"][idx_q], sigma=sigma_fox / np.sqrt(n_q),
            observed=np.log(d.npxG.to_numpy()[idx_q] / n_q),
        )
        has = np.flatnonzero(d.fin_var.to_numpy() > 0)
        resid = (d.fin_goals - d.fin_xg).to_numpy()[has]
        var = d.fin_var.to_numpy()[has]
        e = eta["sniper"][has]
        pm.Potential("goal_approx", pt.sum(resid * e - 0.5 * var * e**2))
    return model


def eta_rows(post_flat: dict, d: pd.DataFrame, skill: str) -> np.ndarray:
    """Linear predictor for every row and draw, from flattened posterior arrays
    (each value has draws on the last axis). Returns (rows, draws)."""
    k = SKILLS.index(skill)
    p, l = d.p_idx.to_numpy(), d.l_idx.to_numpy()
    s = d.season_index.to_numpy() - 1
    return (post_flat[f"mu_{skill}"][None, :]
            + post_flat[f"league_{skill}"][l]
            + post_flat[f"beta_{skill}"][None, :] * d.ts.to_numpy()[:, None]
            + post_flat["gamma"][k][None, :] * d.tsbar.to_numpy()[:, None]
            + post_flat[f"curve_{skill}"][s]
            + post_flat["alpha"][p, k])


# --------------------------------------------------------------------------
# A numpy simulator mirroring the model, for the recovery check
# --------------------------------------------------------------------------

TRUE_PARAMS = {
    "sd": np.array([0.30, 0.25, 0.13, 0.35]),   # spreads set near the real estimates
    "corr": np.array(
        [[1.0, 0.2, -0.1, -0.2],
         [0.2, 1.0, -0.3, -0.5],
         [-0.1, -0.3, 1.0, 0.2],
         [-0.2, -0.5, 0.2, 1.0]]
    ),
    "mu": {"shoot": np.log(2.2), "fox": np.log(0.12), "sniper": 0.0, "create": np.log(1.3)},
    "beta": {"shoot": 0.10, "fox": 0.15, "sniper": 0.03, "create": 0.12},
    "gamma": {"shoot": 0.15, "fox": 0.20, "sniper": 0.02, "create": 0.10},
    "league": {"shoot": [0.05, -0.05, 0.08, 0.0, -0.08], "fox": [0.0, 0.03, -0.02, 0.01, -0.02],
               "sniper": [0.0] * 5, "create": [0.0, 0.05, -0.05, 0.02, -0.02]},
    "phi": {"shoot": 50.0, "create": 35.0},
    "sigma_fox": 1.5,
}


def hump(s: np.ndarray, height: float) -> np.ndarray:
    return height * (np.exp(-((s - 4.0) ** 2) / 8.0) - 0.45)


def simulate(d: pd.DataFrame, shots: pd.DataFrame, n_players: int, seed: int = 1, params=TRUE_PARAMS):
    """Simulate every outcome. Shots keep their real xG; goals are redrawn."""
    rng = np.random.default_rng(seed)
    cov = np.outer(params["sd"], params["sd"]) * params["corr"]
    alpha = rng.multivariate_normal(np.zeros(4), cov, size=n_players)
    p, l, s = d.p_idx.to_numpy(), d.l_idx.to_numpy(), d.season_index.to_numpy()
    ts, tsbar = d.ts.to_numpy(), d.tsbar.to_numpy()
    eta = {}
    for k, skill in enumerate(SKILLS):
        eta[skill] = (params["mu"][skill] + np.asarray(params["league"][skill])[l] + params["beta"][skill] * ts
                      + params["gamma"][skill] * tsbar + hump(s, 0.15 if skill != "sniper" else 0.05) + alpha[p, k])
    sim = d.copy()
    mu_s = d.exposure.to_numpy() * np.exp(eta["shoot"])
    phi = params["phi"]["shoot"]
    sim["np_shots"] = rng.negative_binomial(phi, phi / (phi + mu_s))
    mu_c = d.exposure.to_numpy() * np.exp(eta["create"])
    phi = params["phi"]["create"]
    sim["key_passes"] = rng.negative_binomial(phi, phi / (phi + mu_c))
    n = np.maximum(sim.np_shots.to_numpy(), 1)
    log_xgps = rng.normal(eta["fox"], params["sigma_fox"] / np.sqrt(n))
    sim["npxG"] = np.where(sim.np_shots > 0, np.exp(log_xgps) * sim.np_shots, 0.0)
    sim_shots = shots.copy()
    pr = 1 / (1 + np.exp(-(shots.logit_xg.to_numpy() + eta["sniper"][shots.row.to_numpy()])))
    sim_shots["is_goal"] = rng.binomial(1, pr)
    sim = sim.drop(columns=["fin_goals", "fin_xg", "fin_var"]).join(finishing_stats(sim_shots, len(sim)))
    theta = alpha + d.groupby("p_idx").tsbar.first().reindex(range(n_players)).to_numpy()[:, None] * \
        np.array([params["gamma"][k] for k in SKILLS])[None, :]
    return sim, sim_shots, alpha, theta
