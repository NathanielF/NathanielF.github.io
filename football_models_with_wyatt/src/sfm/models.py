"""The four-skill Soccer Factor Model (season-level), in PyMC.

Each player p has a correlated skill vector alpha_p = (shoot, fox, sniper, create).
For player-season row r (league l, team strength ts, seasons-observed s):

    eta_k[r] = mu_k + league_k[l] + beta_k * ts[r] + h_k(s[r]) + alpha[p, k]

    shooting   np_shots      ~ NegBinomial(exp(log(minutes/90) + eta_shoot), phi_shoot)
    fox        log(npxG/np_shots) ~ Normal(eta_fox, sigma_fox / sqrt(np_shots))
    sniper     npg           ~ Poisson(npxG * exp(eta_sniper))
    creating   key_passes    ~ NegBinomial(exp(log(minutes/90) + eta_create), phi_create)

As in the paper, team strength enters as a factor (beta_k), so alpha is the
skill *net* of the team, and h_k is a Matern-5/2 HSGP "maturity" curve over
seasons observed. Because shots/90 x xG/shot x (goals/xG) = goals/90, the
first three skills multiply into expected non-penalty goals per 90, which is
what SAR and PAR are computed on.
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

# prior centres for the intercepts on the log scale
_MU_PRIOR = {"shoot": np.log(2.2), "fox": np.log(0.12), "sniper": 0.0, "create": np.log(1.3)}


def prepare(player_seasons: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rows used by the model and the coordinate labels."""
    d = player_seasons[player_seasons.player_id.isin(players.player_id)]
    d = d[d.minutes >= MIN_ROW_MINUTES].copy()
    order = players.sort_values("player_id")
    names = dict(zip(order.player_id, order.display_name))
    pid = sorted(d.player_id.unique())
    d["p_idx"] = d.player_id.map({p: i for i, p in enumerate(pid)})
    d["l_idx"] = d.league.map({lg: i for i, lg in enumerate(LEAGUE_ORDER)})
    d["exposure"] = d.minutes / 90.0
    d = d.reset_index(drop=True)
    coords = {
        "player": [names[p] for p in pid],
        "player_id": pid,
        "league": LEAGUE_ORDER,
        "skill": SKILLS,
        "skill_bis": SKILLS,
        "season": SEASON_GRID.astype(int).tolist(),
        "row": np.arange(len(d)),
    }
    return d, coords


def _hsgp_curve(name: str, s_centered: np.ndarray, m: int = 8, c: float = 2.0):
    ell = pm.InverseGamma(f"ell_{name}", alpha=5, beta=20)       # bulk roughly 2-10 seasons
    eta = pm.Exponential(f"amp_{name}", lam=1 / 0.15)             # curve height on the log scale
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov)
    # evaluate on the data rows and on the season grid in one basis, so the
    # stored curve (dims "season") is exactly the function used for the rows
    grid = SEASON_GRID - SEASON_CENTRE
    x = np.concatenate([s_centered, grid])[:, None]
    phi, sqrt_psd = gp.prior_linearized(X=x)
    # centre each basis function over the data rows: the curve then averages
    # zero across player-seasons, so its level cannot trade off against mu_k
    phi = phi - phi[: len(s_centered)].mean(axis=0, keepdims=True)
    beta = pm.Normal(f"hsgp_w_{name}", 0, 1, shape=m)
    f = phi @ (beta * sqrt_psd)
    n = len(s_centered)
    pm.Deterministic(f"curve_{name}", f[n:], dims="season")
    return f[:n]


def build_skill_model(d: pd.DataFrame, coords: dict) -> pm.Model:
    p, l = d.p_idx.to_numpy(), d.l_idx.to_numpy()
    ts = d.team_strength.to_numpy()
    s_c = d.season_index.to_numpy(dtype=float) - SEASON_CENTRE
    log_exp = np.log(d.exposure.to_numpy())

    m_c = {k: v for k, v in coords.items() if k not in ("player_id",)}
    with pm.Model(coords=m_c) as model:
        chol, corr, sds = pm.LKJCholeskyCov(
            "chol", n=4, eta=2.0, sd_dist=pm.Exponential.dist(3.0), compute_corr=True
        )
        pm.Deterministic("skill_corr", corr, dims=("skill", "skill_bis"))
        pm.Deterministic("skill_sd", sds, dims="skill")
        z = pm.Normal("z", 0, 1, dims=("player", "skill"))
        alpha = pm.Deterministic("alpha", z @ chol.T, dims=("player", "skill"))

        eta = {}
        for k, skill in enumerate(SKILLS):
            mu = pm.Normal(f"mu_{skill}", _MU_PRIOR[skill], 0.5)
            lg = pm.ZeroSumNormal(f"league_{skill}", sigma=0.2, dims="league")
            beta = pm.Normal(f"beta_{skill}", 0, 0.5)
            h = _hsgp_curve(skill, s_c)
            eta[skill] = mu + lg[l] + beta * ts + h + alpha[p, k]

        pm.NegativeBinomial(
            "np_shots", mu=pt.exp(log_exp + eta["shoot"]), alpha=pm.Gamma("phi_shoot", 2, 0.1),
            observed=d.np_shots.to_numpy(), dims="row",
        )
        pm.NegativeBinomial(
            "key_passes", mu=pt.exp(log_exp + eta["create"]), alpha=pm.Gamma("phi_create", 2, 0.1),
            observed=d.key_passes.to_numpy(), dims="row",
        )

        has_shots = d.np_shots.to_numpy() > 0
        has_xg = (d.npxG.to_numpy() > 0) & has_shots
        idx_q = np.flatnonzero(has_shots & has_xg)
        n_q = d.np_shots.to_numpy()[idx_q]
        sigma_fox = pm.HalfNormal("sigma_fox", 1.0)  # per-shot spread of log xG around the season mean
        pm.Normal(
            "log_xg_per_shot", mu=eta["fox"][idx_q], sigma=sigma_fox / np.sqrt(n_q),
            observed=np.log(d.npxG.to_numpy()[idx_q] / n_q),
        )
        pm.Poisson(
            "npg", mu=d.npxG.to_numpy()[idx_q] * pt.exp(eta["sniper"][idx_q]),
            observed=d.npg.to_numpy()[idx_q],
        )
    return model


# --------------------------------------------------------------------------
# A numpy simulator mirroring the model, for the recovery check
# --------------------------------------------------------------------------

TRUE_PARAMS = {
    "sd": np.array([0.35, 0.20, 0.12, 0.40]),
    "corr": np.array(
        [[1.0, -0.3, 0.1, 0.3],
         [-0.3, 1.0, 0.0, -0.2],
         [0.1, 0.0, 1.0, 0.1],
         [0.3, -0.2, 0.1, 1.0]]
    ),
    "mu": {"shoot": np.log(2.2), "fox": np.log(0.12), "sniper": 0.0, "create": np.log(1.3)},
    "beta": {"shoot": 0.25, "fox": 0.10, "sniper": 0.03, "create": 0.20},
    "league": {"shoot": [0.05, -0.05, 0.08, 0.0, -0.08], "fox": [0.0, 0.03, -0.02, 0.01, -0.02],
               "sniper": [0.0] * 5, "create": [0.0, 0.05, -0.05, 0.02, -0.02]},
    "phi": {"shoot": 25.0, "create": 15.0},
    "sigma_fox": 1.1,
}


def hump(s: np.ndarray, height: float) -> np.ndarray:
    """A career curve peaking around season 4, centred near zero."""
    return height * (np.exp(-((s - 4.0) ** 2) / 8.0) - 0.45)


def simulate(d: pd.DataFrame, n_players: int, seed: int = 1, params=TRUE_PARAMS) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cov = np.outer(params["sd"], params["sd"]) * params["corr"]
    alpha = rng.multivariate_normal(np.zeros(4), cov, size=n_players)
    p, l, ts, s = d.p_idx.to_numpy(), d.l_idx.to_numpy(), d.team_strength.to_numpy(), d.season_index.to_numpy()
    eta = {}
    for k, skill in enumerate(SKILLS):
        eta[skill] = (params["mu"][skill] + np.asarray(params["league"][skill])[l] + params["beta"][skill] * ts
                      + hump(s, 0.15 if skill != "sniper" else 0.05) + alpha[p, k])
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
    sim["npg"] = rng.poisson(sim.npxG * np.exp(eta["sniper"]))
    return sim, alpha
