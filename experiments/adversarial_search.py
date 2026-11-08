"""Hill-climb over small unweighted instances to find bad cases for each algorithm.

All quantities are exact: OPT by enumerating set partitions, E[cost] by the DP in
cc/exact.py.  Objectives:
    pivot    : E[KwikCluster] / OPT          (theorem: <= 3)
    lp_lp    : E[LP-KwikCluster] / LP        (theorem: <= 2.5)
    lp_opt   : E[LP-KwikCluster] / OPT
Usage: python experiments/adversarial_search.py [n] [restarts] [steps]
"""
import sys

import numpy as np
import pandas as pd

from common import RESULTS
from cc import Instance, solve_lp
from cc.exact import brute_force_opt, expected_kwikcluster, expected_lp_kwikcluster, set_partitions


def score(plus, objective, parts):
    inst = Instance.from_signs(plus)
    opt, _ = brute_force_opt(inst, parts)
    if opt == 0:
        return 1.0, None
    if objective == "pivot":
        return expected_kwikcluster(inst) / opt, None
    lp, D, _ = solve_lp(inst)
    e = expected_lp_kwikcluster(inst, D)
    return (e / lp if objective == "lp_lp" else e / opt), lp


def search(n, objective, restarts, steps, seed=0):
    rng = np.random.default_rng(seed)
    parts = set_partitions(n)
    iu = np.triu_indices(n, k=1)
    best_overall = (0.0, None)
    for r in range(restarts):
        x = rng.random(len(iu[0])) < rng.uniform(0.2, 0.8)
        plus = np.zeros((n, n), bool)
        plus[iu] = x
        cur, _ = score(plus | plus.T, objective, parts)
        for t in range(steps):
            y = x.copy()
            flips = rng.choice(len(y), size=rng.integers(1, 3), replace=False)
            y[flips] = ~y[flips]
            plus = np.zeros((n, n), bool)
            plus[iu] = y
            s, _ = score(plus | plus.T, objective, parts)
            if s >= cur:
                x, cur = y, s
        if cur > best_overall[0]:
            best_overall = (cur, x.copy())
    return best_overall


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    rows = []
    for obj in ("pivot", "lp_lp", "lp_opt"):
        val, x = search(n, obj, restarts, steps)
        edges = [(int(i), int(j)) for i, j, b in zip(*np.triu_indices(n, k=1), x) if b]
        print(f"{obj}: best ratio {val:.4f} with + edges {edges}")
        rows.append(dict(n=n, objective=obj, ratio=val, plus_edges=str(edges)))
    pd.DataFrame(rows).to_csv(RESULTS / f"adversarial_n{n}.csv", index=False)
