"""Export the fitted four-skill model for the web page (web/*.json) and a
thinned, committable posterior (idata/sfm_skills.nc).

Skill values are exported on the log scale (alpha); the page shows them as
multipliers of a typical forward, exp(alpha) ("1.3x as many shots").
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from . import models, sar

KID_NAMES = {"shoot": "Shooter", "fox": "Fox", "sniper": "Sniper", "create": "Playmaker"}
SKILL_BLURBS = {
    "shoot": "How many shots he takes (non-penalty shots per 90 minutes).",
    "fox": "How good his chances are (expected goals per shot): finding the right spot.",
    "sniper": "Scoring more than his chances say he should (goals per expected goal).",
    "create": "Setting up team-mates' shots (key passes per 90 minutes).",
}
N_WEB_DRAWS = 200
N_NC_DRAWS = 1000
KEEP_VARS = ["skill_theta", "gamma", "alpha_corr", "alpha_sd", "phi_shoot", "phi_create", "sigma_fox"] + [
    f"{v}_{k}" for k in models.SKILLS for v in ("mu", "beta", "league", "curve", "ell", "amp")
]


def _flat(post, var):
    return post[var].stack(s=("chain", "draw")).transpose(..., "s").values


def _thin_index(n_total: int, n: int, seed: int = 11) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(n_total, size=n, replace=False))


def _round(x, nd=3):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def _stars(pct: np.ndarray) -> np.ndarray:
    """Percentile in the pool -> 1..5 stars (quintiles)."""
    return np.clip(np.ceil(pct * 5), 1, 5).astype(int)


SNIPER_CLEAR_EDGE = 0.05  # a 5% edge in goals per expected goal


def _interval_stars(lo: np.ndarray, hi: np.ndarray, edge: float = SNIPER_CLEAR_EDGE) -> np.ndarray:
    """Stars for a skill that barely varies (finishing): 3 unless the 83%
    interval excludes zero (4 / 2), and 5 / 1 only if it also excludes +/- edge.
    Percentile stars would turn tiny, uncertain gaps into 1-vs-5 star contrasts."""
    stars = np.full(lo.shape, 3)
    stars[lo > 0], stars[hi < 0] = 4, 2
    stars[lo > edge], stars[hi < -edge] = 5, 1
    return stars


def thinned_idata(idata, n: int = N_NC_DRAWS) -> az.InferenceData:
    post = idata.posterior[KEEP_VARS].stack(s=("chain", "draw"))
    keep = _thin_index(post.sizes["s"], n)
    post = post.isel(s=keep).drop_vars(["chain", "draw", "s"]).rename(s="draw")
    post = post.assign_coords(draw=np.arange(n)).expand_dims(chain=[0]).astype("float32")
    return az.InferenceData(posterior=post, observed_data=idata.observed_data)


def build_exports(idata, rows: pd.DataFrame, players: pd.DataFrame, player_seasons: pd.DataFrame,
                  curve, results: dict) -> dict:
    post = idata.posterior
    alpha = _flat(post, "skill_theta")  # (player, skill, s): each player's skill, theta
    n_s = alpha.shape[-1]
    web_idx = _thin_index(n_s, N_WEB_DRAWS)
    ids = rows.drop_duplicates("p_idx").sort_values("p_idx").player_id.to_numpy()
    names = list(post.player.values)

    level, actual = sar.goals_per90(post, rows, curve)
    rlp = sar.replacement_pool(level)
    S, P = sar.above_replacement(level, rlp), sar.above_replacement(actual, rlp)
    team_lift = P - S  # goals/90 added by the player's teams (positive = boosted)

    meta = players.set_index("player_id")
    ps = player_seasons[player_seasons.player_id.isin(ids)]
    season_lists = {
        pid: [
            {"season": int(r.season), "league": r.league, "team": r.team_title, "minutes": int(r.minutes),
             "season_index": int(r.season_index), "npg": int(r.npg), "np_shots": int(r.np_shots),
             "npxG": round(float(r.npxG), 2), "key_passes": int(r.key_passes),
             "team_strength": round(float(r.team_strength_pre), 3)}
            for r in g.sort_values(["season", "league"]).itertuples()
        ]
        for pid, g in ps.groupby("player_id")
    }

    def summary(x):
        lo, hi = sar.hdi83(x)
        m = x.mean(-1)
        pct = (m.argsort().argsort() + 0.5) / len(m)
        return m, lo, hi, pct

    sk = {k: summary(alpha[:, i, :]) for i, k in enumerate(models.SKILLS)}
    sar_s, lift_s, lvl_s = summary(S), summary(team_lift), summary(level)
    star_arrays = {k: _stars(sk[k][3]) for k in models.SKILLS}
    # use the same 4-dp rounding as the exported bounds, so stars always agree with the file
    star_arrays["sniper"] = _interval_stars(np.round(sk["sniper"][1], 4), np.round(sk["sniper"][2], 4))

    player_records = []
    for i, pid in enumerate(ids):
        m = meta.loc[pid]
        rec = {
            "i": i, "id": int(pid), "name": names[i], "understat_name": m.player_name,
            "pool_reason": m.pool_reason, "is_replacement": bool(rlp[i]),
            "teams": m.teams.split(","), "leagues": m.leagues.split(","),
            "first_season": int(m.first_season), "last_season": int(m.last_season),
            "minutes": int(m.minutes), "games": int(m.games),
            "raw": {"npg": int(m.npg), "npxG": round(float(m.npxG), 2), "npg_p90": float(m.npg_p90),
                    "npxG_p90": float(m.npxG_p90), "np_shots_p90": float(m.np_shots_p90),
                    "key_passes_p90": float(m.key_passes_p90), "npxG_per_shot": float(m.npxG_per_shot),
                    "npg_minus_npxG": float(m.npg_minus_npxG)},
            "skills": {k: {"mean": round(float(sk[k][0][i]), 4), "lo83": round(float(sk[k][1][i]), 4),
                           "hi83": round(float(sk[k][2][i]), 4), "pct": round(float(sk[k][3][i]), 3),
                           "stars": int(star_arrays[k][i])} for k in models.SKILLS},
            "goals90_level": {"mean": round(float(lvl_s[0][i]), 4), "lo83": round(float(lvl_s[1][i]), 4),
                              "hi83": round(float(lvl_s[2][i]), 4)},
            "SAR": {"mean": round(float(sar_s[0][i]), 4), "lo83": round(float(sar_s[1][i]), 4),
                    "hi83": round(float(sar_s[2][i]), 4), "pct": round(float(sar_s[3][i]), 3),
                    "stars": int(_stars(sar_s[3])[i])},
            "team_lift": {"mean": round(float(lift_s[0][i]), 4), "lo83": round(float(lift_s[1][i]), 4),
                          "hi83": round(float(lift_s[2][i]), 4), "pct": round(float(lift_s[3][i]), 3),
                          "stars": int(_stars(lift_s[3])[i]), "p_boosted": round(float((team_lift[i] > 0).mean()), 3)},
            "seasons": season_lists.get(pid, []),
        }
        player_records.append(rec)

    draws = {
        "n_draws": N_WEB_DRAWS,
        "skills": models.SKILLS,
        "note": "Per player (index i in players.json): skill[skill] log-scale skill (theta) draws, and expected "
                "non-penalty goals per 90 on a level field (goals90_level) and with his teams (goals90_actual).",
        "skill": {k: [_round(alpha[i, j, web_idx], 3) for i in range(len(ids))] for j, k in enumerate(models.SKILLS)},
        "goals90_level": [_round(level[i, web_idx], 4) for i in range(len(ids))],
        "goals90_actual": [_round(actual[i, web_idx], 4) for i in range(len(ids))],
    }

    grid = models.SEASON_GRID.astype(int).tolist()
    curves = {"season_index": grid, "note": "Population career curves on the log scale (multiplier = exp)."}
    for k in models.SKILLS:
        c = _flat(post, f"curve_{k}")
        lo, hi = sar.hdi83(c)
        curves[k] = {"mean": _round(c.mean(-1), 4), "lo83": _round(lo, 4), "hi83": _round(hi, 4),
                     "draws": [_round(c[:, j], 4) for j in _thin_index(n_s, 20)]}

    # correlations and spreads of the skills themselves (theta), across players, in each draw
    thin = _thin_index(n_s, 1000)
    corr = np.stack([np.corrcoef(alpha[:, :, j].T) for j in thin], axis=-1)
    lo, hi = sar.hdi83(corr)
    skill_corr = {"skills": models.SKILLS, "of": "player skills theta (alpha plus the Mundlak term), across players",
                  "mean": _round(corr.mean(-1), 3), "lo83": _round(lo, 3), "hi83": _round(hi, 3)}
    theta_sd = alpha.std(axis=0)                     # (skill, s)
    sd_lo, sd_hi = sar.hdi83(theta_sd)

    def scalar(v):
        x = _flat(post, v)
        lo, hi = sar.hdi83(x)
        return {"mean": _round(x.mean(-1), 4), "lo83": _round(lo, 4), "hi83": _round(hi, 4)}

    model_card = {
        "title": "Four-skill Soccer Factor Model (season level)",
        "paper": {"authors": "Alexandre Andorra and Maximilian Göbel", "year": 2024,
                  "title": "Unveiling True Talent: The Soccer Factor Model for Skill Evaluation",
                  "url": "https://arxiv.org/abs/2412.05911"},
        "data": {"window": "2014/15-2023/24", "leagues": models.LEAGUE_ORDER, "players": len(ids),
                 "player_seasons": int(len(rows)), "min_row_minutes": models.MIN_ROW_MINUTES,
                 "sources": ["Understat (via Kaggle: mexwell/understat-database)", "football-data.co.uk"]},
        "skills": {k: {"kid_name": KID_NAMES[k], "blurb": SKILL_BLURBS[k]} for k in models.SKILLS},
        "intervals": "83% central posterior intervals, as in the paper",
        "stars": {"default": "pool quintiles of the posterior mean (1 = bottom 20%, 5 = top 20%)",
                  "sniper": "3 unless the 83% interval excludes 0 (4 above, 2 below); 5 or 1 only if it also "
                            f"excludes +/-{SNIPER_CLEAR_EDGE} (a 5% edge in goals per expected goal), because "
                            "finishing skill barely varies and percentiles would exaggerate tiny gaps"},
        "replacement_pool": {"rule": "players between the 40th and 60th percentile of level-field goals/90",
                             "n_players": int(rlp.sum())},
        "fit": results.get("fit_diagnostics", {}),
        "prior_predictive": results.get("prior_predictive", {}),
        "recovery": results.get("recovery", {}),
        "ppc_coverage_83": results.get("ppc_coverage", {}),
        "params": {**{f"mu_{k}": scalar(f"mu_{k}") for k in models.SKILLS},
                   **{f"beta_{k}": scalar(f"beta_{k}") for k in models.SKILLS},
                   **{f"league_{k}": scalar(f"league_{k}") for k in models.SKILLS},
                   "gamma": scalar("gamma"),
                   "alpha_sd": scalar("alpha_sd"),
                   "alpha_corr": scalar("alpha_corr"),
                   "skill_sd": {"mean": _round(theta_sd.mean(-1), 4), "lo83": _round(sd_lo, 4), "hi83": _round(sd_hi, 4)}},
        "departures_from_paper": [
            "Season-level outcomes rather than per-match goals: the Kaggle Understat dump has no per-match lineups.",
            "Four correlated skills (shooting, chance quality, finishing, creating) instead of a single goals alpha; "
            "the first three multiply into expected non-penalty goals per 90, on which SAR/PAR are computed.",
            "Team strength is the team's season average of its pre-match points-per-game gap to opponents, the paper's "
            "factor averaged over a season; no home factor at season level.",
            "A Mundlak term (each player's average team strength) separates within-player team effects from the "
            "tendency of better forwards to play for better teams.",
            "Finishing is a log-odds shift on each shot's xG (Bernoulli per shot), not a goals count.",
            "Player effects are correlated (MVN with an LKJ prior) rather than summing to zero.",
            "The forward pool is any player Understat lists at F for most of his minutes, with at least 6,000 minutes; the "
            "replacement pool is the 40th-60th percentile of level-field skill (paper: hand-picked strikers).",
            "One population HSGP career curve per skill over seasons observed in the window (paper: per-player "
            "within- and cross-season HSGPs).",
            "Penalties excluded throughout.",
        ],
    }

    guesses = {
        "note": "Wyatt's ranking before seeing the model, from best (1) down. Fill in 'players' by display name.",
        "question": "Who is the best goal scorer if every team were equally good?",
        "players": [],
    }
    return {"players": player_records, "draws": draws, "curves": curves, "skill_corr": skill_corr,
            "model_card": model_card, "wyatt_guesses": guesses}


def write_json(obj, path: Path) -> int:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size
