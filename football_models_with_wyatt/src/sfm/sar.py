"""Skill- and Performance-Above-Replacement from the four-skill posterior.

Expected non-penalty goals per 90 for player p in posterior draw j:

    level field (SAR):  exp(mu_shoot + theta_shoot) * exp(mu_fox + theta_fox) * conv(mu_sniper + theta_sniper)
                        average league, team strength 0, career curve at its average
    actual (PAR):       the same, with beta_k * TS added for each of p's player-seasons,
                        minutes-weighted (league and career stage held at average)

theta is the player's skill (alpha plus the Mundlak term). Finishing is a log-odds
shift on each shot's xG, so its effect on goals per xG depends on the mix of
chances. conv(eta) is goals per unit of xG for a reference mix of chances (all
non-penalty shots in the model) when every shot's log-odds is shifted by eta.

SAR_p = level_p - mean(level over replacement pool); PAR likewise with actual.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

SCORING = ["shoot", "fox", "sniper"]
RLP_QUANTILES = (0.40, 0.60)
_ETA_GRID = np.linspace(-2.0, 2.0, 401)


def _stack(post: xr.Dataset, name: str) -> np.ndarray:
    return post[name].stack(s=("chain", "draw")).transpose(..., "s").values


def conversion_curve(logit_xg: np.ndarray, grid: np.ndarray = _ETA_GRID, n_ref: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Goals per xG when every reference shot's log-odds is shifted by eta."""
    ref = np.quantile(logit_xg, (np.arange(n_ref) + 0.5) / n_ref)
    base = (1 / (1 + np.exp(-ref))).mean()
    conv = np.array([(1 / (1 + np.exp(-(ref + e)))).mean() for e in grid]) / base
    return grid, conv


def _conv(eta: np.ndarray, curve) -> np.ndarray:
    grid, conv = curve
    return np.interp(eta, grid, conv)


def goals_per90(post: xr.Dataset, rows: pd.DataFrame, curve) -> tuple[np.ndarray, np.ndarray]:
    """Return (level, actual) arrays of shape (players, draws)."""
    theta = _stack(post, "skill_theta")  # (player, skill, s)
    skills = list(post.skill.values)
    p = rows.p_idx.to_numpy()
    ts = rows.ts.to_numpy()[:, None]
    lvl_eta, row_eta = {}, {}
    for k in SCORING:
        mu = _stack(post, f"mu_{k}")
        a = theta[:, skills.index(k), :]
        lvl_eta[k] = mu + a
        row_eta[k] = mu + _stack(post, f"beta_{k}") * ts + a[p]   # only the team factor differs
    level = np.exp(lvl_eta["shoot"] + lvl_eta["fox"]) * _conv(lvl_eta["sniper"], curve)
    row_rate = np.exp(row_eta["shoot"] + row_eta["fox"]) * _conv(row_eta["sniper"], curve)
    w = rows.minutes.to_numpy()[:, None] * row_rate
    actual = np.zeros_like(level)
    np.add.at(actual, p, w)
    mins = np.bincount(p, weights=rows.minutes.to_numpy(), minlength=level.shape[0])
    return level, actual / mins[:, None]


def above_replacement(values: np.ndarray, replacement_mask: np.ndarray) -> np.ndarray:
    return values - values[replacement_mask].mean(axis=0, keepdims=True)


def hdi83(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(x, 0.085, axis=-1), np.quantile(x, 0.915, axis=-1)


def replacement_pool(level: np.ndarray, quantiles=RLP_QUANTILES) -> np.ndarray:
    m = level.mean(axis=1)
    lo, hi = np.quantile(m, quantiles)
    return (m >= lo) & (m <= hi)


def sar_table(post: xr.Dataset, rows: pd.DataFrame, players: pd.DataFrame, curve) -> pd.DataFrame:
    level, actual = goals_per90(post, rows, curve)
    rlp = replacement_pool(level)
    S, P = above_replacement(level, rlp), above_replacement(actual, rlp)
    ids = rows.drop_duplicates("p_idx").sort_values("p_idx").player_id.to_numpy()
    out = pd.DataFrame({"player_id": ids, "player": post.player.values})
    for name, x in (("goals90_level", level), ("goals90_actual", actual), ("SAR", S), ("PAR", P),
                    ("SAR_minus_PAR", S - P), ("team_lift_own", actual - level)):
        lo, hi = hdi83(x)
        out[name] = x.mean(1)
        out[f"{name}_lo83"], out[f"{name}_hi83"] = lo, hi
    out["is_replacement"] = rlp
    out = out.merge(players[["player_id", "minutes", "pool_reason"]], on="player_id", how="left")
    out["SAR_rank"] = out.SAR.rank(ascending=False, method="min").astype(int)
    return out.sort_values("SAR_rank").reset_index(drop=True)
