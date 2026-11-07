"""Export web/*.json for the page and the thinned idata/sfm_skills.nc."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sfm import export  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

IDATA, WEB = PROJECT / "idata", PROJECT / "web"


def ppc_coverage(idata) -> dict:
    out = {}
    for v in ["np_shots", "key_passes", "log_xg_per_shot", "npg"]:
        x = idata.posterior_predictive[v].stack(s=("chain", "draw")).values
        o = idata.observed_data[v].values
        lo, hi = np.quantile(x, [0.085, 0.915], axis=-1)
        out[v] = round(float(((o >= lo) & (o <= hi)).mean()), 3)
    out["npg_total_observed"] = int(idata.observed_data["npg"].sum())
    out["npg_total_predicted"] = round(float(idata.posterior_predictive["npg"].sum("npg_dim_0").mean()), 1)
    return out


def fit_stats(idata) -> dict:
    summ = pd.read_csv(IDATA / "sfm_skills_summary.csv", index_col=0)
    return {"sampler": "nutpie", "chains": int(idata.posterior.sizes["chain"]),
            "draws_per_chain": int(idata.posterior.sizes["draw"]),
            "divergences": int(idata.sample_stats.diverging.sum()),
            "max_r_hat": round(float(summ.r_hat.max()), 3), "min_ess_bulk": int(summ.ess_bulk.min())}


if __name__ == "__main__":
    idata = az.from_netcdf(IDATA / "sfm_skills_full.nc")
    rows = pd.read_csv(IDATA / "sfm_skills_rows.csv")
    proc = PROJECT / "data" / "processed"
    players, seasons = pd.read_csv(proc / "players.csv"), pd.read_csv(proc / "player_seasons.csv")

    out = export.build_exports(idata, rows, players, seasons, ppc_coverage(idata), fit_stats(idata))
    WEB.mkdir(exist_ok=True)
    guesses_path = WEB / "wyatt_guesses.json"
    if guesses_path.exists() and json.loads(guesses_path.read_text()).get("players"):
        out.pop("wyatt_guesses")  # never overwrite Wyatt's real guesses
    for name, obj in out.items():
        size = export.write_json(obj, WEB / f"{name}.json")
        print(f"web/{name}.json  {size / 1e6:.2f} MB")

    thin = export.thinned_idata(idata)
    thin.to_netcdf(IDATA / "sfm_skills.nc")
    print(f"idata/sfm_skills.nc  {(IDATA / 'sfm_skills.nc').stat().st_size / 1e6:.1f} MB")
