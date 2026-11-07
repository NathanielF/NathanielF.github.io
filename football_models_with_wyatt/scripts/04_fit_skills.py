"""Fit the four-skill SFM.

    python scripts/04_fit_skills.py recovery   # fit to simulated data with known skills
    python scripts/04_fit_skills.py real       # fit to the real player-season panel

Posteriors are written to idata/ (full-size files are gitignored; export.py
writes the thinned, committed versions).
"""

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
SAMPLE_KW = dict(draws=1500, tune=1500, chains=4, target_accept=0.95, nuts_sampler="nutpie", random_seed=2024, progressbar=False)


def load():
    ps = pd.read_csv(PROC / "player_seasons.csv")
    players = pd.read_csv(PROC / "players.csv")
    return models.prepare(ps, players)


def summarise_fit(idata, label):
    div = int(idata.sample_stats.diverging.sum())
    summ = az.summary(idata, var_names=["~z", "~hsgp_w", "~alpha", "~chol"], round_to=3)
    alpha_rhat = az.rhat(idata, var_names=["alpha"]).alpha.max().item()
    alpha_ess = az.ess(idata, var_names=["alpha"]).alpha.min().item()
    print(f"[{label}] divergences={div}  max r_hat={summ.r_hat.max():.3f} (alpha {alpha_rhat:.3f})  "
          f"min bulk ESS={summ.ess_bulk.min():.0f} (alpha {alpha_ess:.0f})")
    return summ


def recovery():
    d, coords = load()
    sim, alpha_true = models.simulate(d, n_players=len(coords["player"]), seed=7)
    model = models.build_skill_model(sim, coords)
    t0 = time.time()
    with model:
        idata = pm.sample(**SAMPLE_KW)
    print(f"sampled in {time.time() - t0:.0f}s")
    summ = summarise_fit(idata, "recovery")
    print(summ.sort_values("r_hat").tail(6)[["mean", "sd", "ess_bulk", "r_hat"]])

    post = idata.posterior
    a = post.alpha.stack(s=("chain", "draw")).transpose("player", "skill", "s").values
    lo, hi = np.quantile(a, [0.085, 0.915], axis=-1)  # central 83% interval
    cover = ((alpha_true >= lo) & (alpha_true <= hi)).mean(axis=0)
    corr_mean = np.array([np.corrcoef(a[:, k, :].mean(-1), alpha_true[:, k])[0, 1] for k in range(4)])
    print("alpha 83% coverage by skill:", dict(zip(models.SKILLS, cover.round(3))))
    print("corr(posterior mean, truth):", dict(zip(models.SKILLS, corr_mean.round(3))))
    sd_est = post.skill_sd.mean(("chain", "draw")).values
    print("skill sd  true:", models.TRUE_PARAMS["sd"], " est:", sd_est.round(3))
    corr_est = post.skill_corr.mean(("chain", "draw")).values
    print("skill corr true vs est (upper triangle):")
    iu = np.triu_indices(4, 1)
    print("  true", models.TRUE_PARAMS["corr"][iu].round(2), "\n  est ", corr_est[iu].round(2))
    for s in models.SKILLS:
        print(f"  beta_{s}: true {models.TRUE_PARAMS['beta'][s]:+.2f}  est {post[f'beta_{s}'].mean().item():+.3f}"
              f"  mu_{s}: true {models.TRUE_PARAMS['mu'][s]:+.3f} est {post[f'mu_{s}'].mean().item():+.3f}")
    true_curve = models.hump(models.SEASON_GRID, 0.15)
    est_curve = post.curve_shoot.mean(("chain", "draw")).values
    print("shoot curve (true, centred):", (true_curve - true_curve.mean()).round(3))
    print("shoot curve (est,  centred):", (est_curve - est_curve.mean()).round(3))


def real():
    d, coords = load()
    model = models.build_skill_model(d, coords)
    t0 = time.time()
    with model:
        idata = pm.sample(**SAMPLE_KW, idata_kwargs={"log_likelihood": True})
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=2024, progressbar=False)
    print(f"sampled in {time.time() - t0:.0f}s")
    summ = summarise_fit(idata, "real")
    IDATA.mkdir(exist_ok=True)
    idata.to_netcdf(IDATA / "sfm_skills_full.nc")
    summ.to_csv(IDATA / "sfm_skills_summary.csv")
    d.to_csv(IDATA / "sfm_skills_rows.csv", index=False)
    print(summ.loc[[i for i in summ.index if not i.startswith(("curve", "league"))]].to_string())


if __name__ == "__main__":
    {"recovery": recovery, "real": real}[sys.argv[1]]()
