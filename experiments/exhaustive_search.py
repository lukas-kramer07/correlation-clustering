"""Exhaustive worst-case check on small instances.

Evaluates every sign pattern on n vertices up to relabelling: one graph of "+" pairs per
isomorphism class (4, 11, 34, 156, 1044 classes for n = 3..7). For each it computes
    pivot/opt       : E[KwikCluster] / OPT        (theorem: <= 3)
    lp_rounding/lp  : E[LP-KwikCluster] / LP      (theorem: <= 2.5)
    lp_rounding/opt : E[LP-KwikCluster] / OPT
All quantities are exact: OPT by enumerating set partitions, E[cost] by the DPs in
cc/exact.py. E[LP-KwikCluster] is for the LP optimum that the solver returns. Instances
with OPT = 0 (disjoint cliques) are listed but have no ratio.
Usage: python experiments/exhaustive_search.py [max_n]      (default 7, about 15 seconds)
"""
import sys

import numpy as np
import pandas as pd

from common import RESULTS
from cc import Instance, solve_lp
from cc.exact import (brute_force_opt, expected_kwikcluster, expected_lp_kwikcluster,
                      graph_classes, set_partitions)

OBJECTIVES = ("pivot/opt", "lp_rounding/lp", "lp_rounding/opt")
TOL = 1e-9


def evaluate(plus, parts):
    n = len(plus)
    inst = Instance.from_signs(plus)
    edges = [(int(i), int(j)) for i, j in zip(*np.triu_indices(n, 1)) if plus[i, j]]
    opt, _ = brute_force_opt(inst, parts)
    row = dict(n=n, plus_pairs=len(edges), plus_edges=str(edges), opt=opt)
    if opt == 0:
        return row
    lp, D, _ = solve_lp(inst)
    e_pivot = expected_kwikcluster(inst)
    e_lp = expected_lp_kwikcluster(inst, D)
    row.update(lp=lp, e_pivot=e_pivot, e_lp_rounding=e_lp)
    row.update(zip(OBJECTIVES, (e_pivot / opt, e_lp / lp, e_lp / opt)))
    return row


if __name__ == "__main__":
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    rows = []
    for n in range(3, max_n + 1):
        parts = set_partitions(n)
        reps = graph_classes(n)
        rows += [evaluate(plus, parts) for plus in reps]
        print(f"n={n}: {len(reps)} classes done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "exhaustive.csv", index=False)
    for n, g in df.groupby("n"):
        for obj in OBJECTIVES:
            best = g[obj].max()
            ties = g[g[obj] >= best - TOL]
            print(f"n={n} {obj:15s} max {best:.4f}  attained by {len(ties)} class(es), "
                  f"e.g. + edges {ties.iloc[0]['plus_edges']}")
