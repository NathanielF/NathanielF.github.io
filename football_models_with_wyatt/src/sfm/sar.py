"""Skill- and Performance-Above-Replacement from the four-skill posterior.

Expected non-penalty goals per 90 for player p in posterior draw j:

    level field (SAR):  exp(mu_shoot + a_shoot) * exp(mu_fox + a_fox) * exp(mu_sniper + a_sniper)
                        i.e. average league, team strength 0, career curve at its average
    actual (PAR):       the same product plus beta_k * team_strength for each of p's
                        player-seasons, minutes-weighted (league and career stage held at average)

SAR_p = level_p - mean(level over replacement pool); PAR likewise with actual.
SAR - PAR < 0 means the team(s) boosted the player's numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

SCORING = ["shoot", "fox", "sniper"]


def _stack(post: xr.Dataset, name: str) -> np.ndarray:
    return post[name].stack(s=("chain", "draw")).transpose(..., "s").values


def goals_per90(post: xr.Dataset, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (level, actual) arrays of shape (players, draws)."""
    alpha = _stack(post, "alpha")  # (player, skill, s)
    skills = list(post.skill.values)
    level = np.zeros(alpha[:, 0, :].shape)
    eta_rows = np.zeros((len(rows), alpha.shape[-1]))
    p = rows.p_idx.to_numpy()
    ts = rows.team_strength.to_numpy()[:, None]
    for k in SCORING:
        mu = _stack(post, f"mu_{k}")
        a = alpha[:, skills.index(k), :]
        level += mu + a
        # only the team factor differs from the level field, as in the paper's PAR
        eta_rows += mu + _stack(post, f"beta_{k}") * ts + a[p]
    level = np.exp(level)
    w = rows.minutes.to_numpy()[:, None] * np.exp(eta_rows)
    n_players = alpha.shape[0]
    actual = np.zeros_like(level)
    np.add.at(actual, p, w)
    mins = np.bincount(p, weights=rows.minutes.to_numpy(), minlength=n_players)
    return level, actual / mins[:, None]


def above_replacement(values: np.ndarray, replacement_mask: np.ndarray) -> np.ndarray:
    return values - values[replacement_mask].mean(axis=0, keepdims=True)


def hdi83(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(x, 0.085, axis=-1), np.quantile(x, 0.915, axis=-1)


# The default replacement-level pool: players whose level-field skill (posterior
# mean goals/90) sits between the 40th and 60th percentiles of the pool, i.e. a
# typical squad forward. The web page lets the viewer pick a different pool.
RLP_QUANTILES = (0.40, 0.60)


def replacement_pool(level: np.ndarray, quantiles=RLP_QUANTILES) -> np.ndarray:
    m = level.mean(axis=1)
    lo, hi = np.quantile(m, quantiles)
    return (m >= lo) & (m <= hi)


def sar_table(post: xr.Dataset, rows: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    level, actual = goals_per90(post, rows)
    rlp = replacement_pool(level)
    S, P = above_replacement(level, rlp), above_replacement(actual, rlp)
    ids = rows.drop_duplicates("p_idx").sort_values("p_idx").player_id.to_numpy()  # p_idx order = player coord
    out = pd.DataFrame({"player_id": ids, "player": post.player.values})
    for name, x in (("goals90_level", level), ("goals90_actual", actual), ("SAR", S), ("PAR", P), ("SAR_minus_PAR", S - P)):
        lo, hi = hdi83(x)
        out[name] = x.mean(1)
        out[f"{name}_lo83"], out[f"{name}_hi83"] = lo, hi
    out["P_team_boosted"] = ((S - P) < 0).mean(1)
    out["is_replacement"] = rlp
    out = out.merge(players[["player_id", "minutes", "pool_reason"]], on="player_id", how="left")
    out["SAR_rank"] = out.SAR.rank(ascending=False, method="min").astype(int)
    return out.sort_values("SAR_rank").reset_index(drop=True)
