import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

import numpy as np  # noqa: E402

from cc import kwikcluster, lp_kwikcluster, solve_ilp, solve_lp  # noqa: E402


def evaluate(inst, R=200, seed=0, ilp_time=60.0, want_opt=True):
    """Run both algorithms R times on one instance and return a result row."""
    rng = np.random.default_rng(seed)
    lp, D, info = solve_lp(inst)
    opt = None
    if want_opt:
        opt, _, _ = solve_ilp(inst, time_limit=ilp_time, init_cuts=info["_cuts"])
    piv = np.array([inst.cost(kwikcluster(inst, rng)) for _ in range(R)])
    lpp = np.array([inst.cost(lp_kwikcluster(inst, D, rng)) for _ in range(R)])
    row = dict(name=inst.name, n=inst.n, lp=lp, opt=opt, lp_time=info["time"],
               pivot_mean=piv.mean(), pivot_min=piv.min(), pivot_max=piv.max(),
               lppivot_mean=lpp.mean(), lppivot_min=lpp.min(), lppivot_max=lpp.max())
    ref = opt if opt is not None else lp
    row["ref"] = "OPT" if opt is not None else "LP"
    for k in ("pivot_mean", "pivot_max", "lppivot_mean", "lppivot_max"):
        row[k + "/ref"] = row[k] / ref if ref > 0 else (1.0 if row[k] == 0 else np.inf)
        row[k + "/lp"] = row[k] / lp if lp > 0 else (1.0 if row[k] == 0 else np.inf)
    return row, D
