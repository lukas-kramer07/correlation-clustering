"""Run all experiments:  python experiments/run_experiments.py

Writes CSV files to results/.  Runtime is roughly 10-15 minutes on a laptop.
Seeds are fixed, so reruns give identical instances; HiGHS may return a different
optimal LP vertex on other platforms, which can change LP rounding numbers slightly.
"""
import time

import numpy as np
import pandas as pd

from common import RESULTS, evaluate
from cc import kwikcluster, lp_kwikcluster, solve_lp
from cc.datasets import similarity_instance
from cc.generators import complete_minus_edge, planted, random_signed, star


def star_lp_pivot_expectation(n):
    """E[LP-pivot] on star(n) with the (unique) LP solution d = 1/2 on spokes, 1 elsewhere."""
    L = 0.0
    for m in range(1, n + 1):
        L = (m / 2 + m * (m - 1) / 8) / (m + 1) + m / (m + 1) * ((m - 1) / 2 + (1 + L) / 2)
    return L


def exp_tight():
    rows = []
    rng = np.random.default_rng(0)
    for n in (5, 10, 20, 40, 80):
        inst = complete_minus_edge(n)
        lp, D, _ = solve_lp(inst)
        mc = np.mean([inst.cost(kwikcluster(inst, rng)) for _ in range(4000)])
        mcl = np.mean([inst.cost(lp_kwikcluster(inst, D, rng)) for _ in range(4000)])
        rows.append(dict(family="Kn-e", n=n, opt=1.0, lp=lp, e_pivot=3 - 6 / n,
                         mc_pivot=mc, mc_lppivot=mcl))
    for m in (5, 10, 20, 40, 80):
        inst = star(m)
        lp, D, _ = solve_lp(inst)
        mc = np.mean([inst.cost(kwikcluster(inst, rng)) for _ in range(4000)])
        mcl = np.mean([inst.cost(lp_kwikcluster(inst, D, rng)) for _ in range(4000)])
        rows.append(dict(family="star", n=m + 1, opt=m - 1.0, lp=lp,
                         e_pivot=1.5 * m * (m - 1) / (m + 1), mc_pivot=mc,
                         e_lppivot=star_lp_pivot_expectation(m), mc_lppivot=mcl))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tight.csv", index=False)
    return df


def exp_synthetic():
    rows = []
    for p in (0.2, 0.5, 0.8):
        for s in range(5):
            r, _ = evaluate(random_signed(20, p, s), seed=s)
            rows.append(dict(family=f"random p={p}", eps=np.nan, **r))
    for eps in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
        for s in range(5):
            inst, _ = planted([10, 10, 10], eps, s)
            r, _ = evaluate(inst, seed=s)
            rows.append(dict(family="planted n=30", eps=eps, **r))
            inst, _ = planted([30, 20, 20, 10], eps, 100 + s)
            r, _ = evaluate(inst, seed=s, want_opt=False)
            rows.append(dict(family="planted n=80", eps=eps, **r))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "synthetic.csv", index=False)
    return df


def exp_real(n=100):
    rows = []
    for name in ("iris", "wine", "breast_cancer"):
        inst, _ = similarity_instance(name, n, rng=0)
        r, _ = evaluate(inst, want_opt=False)
        rows.append(dict(dataset=name, **r))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "real.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    for f in (exp_tight, exp_synthetic, exp_real):
        t = time.time()
        df = f()
        print(f"== {f.__name__} ({time.time() - t:.0f}s)")
        print(df.round(3).to_string(max_rows=80))
