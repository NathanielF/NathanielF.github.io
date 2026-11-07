"""Export web/*.json for the page and the thinned idata/sfm_skills.nc."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import arviz as az  # noqa: E402
import pandas as pd  # noqa: E402

from sfm import export, ppc, sar  # noqa: E402
from sfm.fetch import PROJECT  # noqa: E402

IDATA, WEB = PROJECT / "idata", PROJECT / "web"


def load_results() -> dict:
    out = {}
    for name in ["fit_diagnostics", "prior_predictive", "recovery"]:
        f = PROJECT / "results" / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


if __name__ == "__main__":
    idata = az.from_netcdf(IDATA / "sfm_skills_full.nc")
    rows = pd.read_csv(IDATA / "sfm_skills_rows.csv")
    shots = pd.read_parquet(IDATA / "sfm_skills_shots.parquet")
    curve = sar.conversion_curve(shots.logit_xg.to_numpy())
    names = [str(n) for n in idata.posterior.player.values]
    checks = ppc.build_ppc(idata, rows, shots, names)
    results = load_results()
    results["ppc_coverage"] = checks["coverage83"]
    proc = PROJECT / "data" / "processed"
    players, seasons = pd.read_csv(proc / "players.csv"), pd.read_csv(proc / "player_seasons.csv")
    sar.sar_table(idata.posterior, rows, players, curve).round(4).to_csv(IDATA / "sar_table.csv", index=False)

    out = export.build_exports(idata, rows, players, seasons, curve, results)
    WEB.mkdir(exist_ok=True)
    guesses_path = WEB / "wyatt_guesses.json"
    if guesses_path.exists() and json.loads(guesses_path.read_text()).get("players"):
        out.pop("wyatt_guesses")  # never overwrite Wyatt's real guesses
    for name, obj in out.items():
        size = export.write_json(obj, WEB / f"{name}.json")
        print(f"web/{name}.json  {size / 1e6:.2f} MB")

    size = export.write_json(checks, WEB / "ppc.json")
    print(f"web/ppc.json  {size / 1e6:.2f} MB")

    thin = export.thinned_idata(idata)
    thin.to_netcdf(IDATA / "sfm_skills.nc")
    print(f"idata/sfm_skills.nc  {(IDATA / 'sfm_skills.nc').stat().st_size / 1e6:.1f} MB")
