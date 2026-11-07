"""Fit the four-skill SFM.

    python scripts/04_fit_skills.py prior      # prior predictive summary -> results/prior_predictive.json
    python scripts/04_fit_skills.py recovery   # fit simulated data with known skills -> results/recovery.json
    python scripts/04_fit_skills.py real       # fit the real panel -> idata/ (gitignored)
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

from sfm import models  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

PROC = PROJECT / "data" / "processed"
IDATA = PROJECT / "idata"
RESULTS = PROJECT / "results"
SAMPLE_KW = dict(draws=1500, tune=1500, chains=4, target_accept=0.95, nuts_sampler="nutpie",
                 random_seed=2024, progressbar=False)
SEASON_VARS = ["np_shots", "key_passes", "log_xg_per_shot"]


def load():
    ps = pd.read_csv(PROC / "player_seasons.csv")
    players = pd.read_csv(PROC / "players.csv")
    shots = pd.read_parquet(PROC / "shots.parquet")
    return models.prepare(ps, players, shots)


def write_result(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2))
    print(json.dumps(obj, indent=2))


def diagnostics(idata) -> tuple[dict, pd.DataFrame]:
    summ = az.summary(idata, var_names=["~z", "~hsgp_w", "~alpha", "~skill_theta", "~chol"], round_to=4)
    th_rhat = float(az.rhat(idata, var_names=["skill_theta"]).skill_theta.max())
    th_ess = float(az.ess(idata, var_names=["skill_theta"]).skill_theta.min())
    return {"divergences": int(idata.sample_stats.diverging.sum()),
            "max_r_hat": float(summ.r_hat.max()), "max_r_hat_param": str(summ.r_hat.idxmax()),
            "min_ess_bulk": float(summ.ess_bulk.min()), "min_ess_bulk_param": str(summ.ess_bulk.idxmin()),
            "skill_max_r_hat": th_rhat, "skill_min_ess_bulk": th_ess}, summ


def prior():
    d, coords, shots = load()
    with models.build_skill_model(d, coords, shots):
        pp = pm.sample_prior_predictive(500, var_names=SEASON_VARS, random_seed=1)
    out = {}
    for v in ["np_shots", "key_passes"]:
        x = pp.prior_predictive[v].values.ravel()
        out[v] = {"prior_median": float(np.median(x)), "prior_90": [float(q) for q in np.quantile(x, [0.05, 0.95])],
                  "data_median": float(np.median(d[v])), "data_90": [float(q) for q in np.quantile(d[v], [0.05, 0.95])]}
    x = np.exp(pp.prior_predictive["log_xg_per_shot"].values.ravel())
    obs = (d.npxG / d.np_shots.clip(lower=1))[d.np_shots > 0]
    out["xg_per_shot"] = {"prior_median": float(np.median(x)), "prior_90": [float(q) for q in np.quantile(x, [0.05, 0.95])],
                          "data_median": float(np.median(obs)), "data_90": [float(q) for q in np.quantile(obs, [0.05, 0.95])]}
    write_result("prior_predictive", out)


def recovery():
    d, coords, shots = load()
    n_p = len(coords["player"])
    sim, sim_shots, alpha_true, theta_true = models.simulate(d, shots, n_players=n_p, seed=7)
    t0 = time.time()
    with models.build_skill_model(sim, coords, sim_shots):
        idata = pm.sample(**SAMPLE_KW)
    diag, _ = diagnostics(idata)
    post = idata.posterior
    th = post.skill_theta.stack(s=("chain", "draw")).transpose("player", "skill", "s").values
    lo, hi = np.quantile(th, [0.085, 0.915], axis=-1)
    cover = ((theta_true >= lo) & (theta_true <= hi)).mean(axis=0)
    corr_mean = [float(np.corrcoef(th[:, k, :].mean(-1), theta_true[:, k])[0, 1]) for k in range(4)]
    tp = models.TRUE_PARAMS
    iu = np.triu_indices(4, 1)
    out = {
        "seconds": round(time.time() - t0),
        "diagnostics": diag,
        "skill_coverage83": dict(zip(models.SKILLS, np.round(cover, 3).tolist())),
        "skill_corr_with_truth": dict(zip(models.SKILLS, np.round(corr_mean, 3).tolist())),
        "alpha_sd_true": tp["sd"].tolist(),
        "alpha_sd_est": np.round(post.alpha_sd.mean(("chain", "draw")).values, 3).tolist(),
        "alpha_corr_true": tp["corr"][iu].round(2).tolist(),
        "alpha_corr_est": post.alpha_corr.mean(("chain", "draw")).values[iu].round(2).tolist(),
        "beta_true": tp["beta"], "beta_est": {k: round(float(post[f"beta_{k}"].mean()), 3) for k in models.SKILLS},
        "gamma_true": tp["gamma"],
        "gamma_est": dict(zip(models.SKILLS, np.round(post.gamma.mean(("chain", "draw")).values, 3).tolist())),
    }
    write_result("recovery", out)


def real():
    d, coords, shots = load()
    model = models.build_skill_model(d, coords, shots)
    t0 = time.time()
    with model:
        idata = pm.sample(**SAMPLE_KW)
        pm.sample_posterior_predictive(idata, var_names=SEASON_VARS, extend_inferencedata=True,
                                       random_seed=2024, progressbar=False)
        pm.compute_log_likelihood(idata, var_names=SEASON_VARS, extend_inferencedata=True, progressbar=False)
    print(f"sampled in {time.time() - t0:.0f}s")
    IDATA.mkdir(exist_ok=True)
    idata.to_netcdf(IDATA / "sfm_skills_full.nc")   # save before anything else can fail
    diag, summ = diagnostics(idata)
    summ.to_csv(IDATA / "sfm_skills_summary.csv")
    d.to_csv(IDATA / "sfm_skills_rows.csv", index=False)
    shots.to_parquet(IDATA / "sfm_skills_shots.parquet", index=False)
    write_result("fit_diagnostics", {"seconds": round(time.time() - t0), **diag})


if __name__ == "__main__":
    {"prior": prior, "recovery": recovery, "real": real}[sys.argv[1]]()
