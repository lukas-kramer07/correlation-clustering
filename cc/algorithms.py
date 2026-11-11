"""Pivot-based algorithms of Ailon, Charikar and Newman (J. ACM 2008).

kwikcluster     : KwikCluster.  Pick a uniformly random pivot among the unclustered
                  vertices, cluster it with all its unclustered '+' neighbours, recurse.
                  On weighted instances it runs on the majority instance (ACN Def. 2.1).
lp_kwikcluster  : LP-KwikCluster.  Same recursion, but j joins pivot k's cluster
                  independently with probability x+_kj = 1 - d_kj, where d is an optimal
                  solution of the LP relaxation (computed once, reused in every call).

Scanning a uniformly random permutation and treating the first unclustered vertex as the
next pivot is equivalent to choosing each pivot uniformly among the remaining vertices.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance


def _rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def kwikcluster(inst: Instance, rng=None, plus: np.ndarray | None = None) -> np.ndarray:
    rng = _rng(rng)
    plus = inst.majority() if plus is None else plus
    n = inst.n
    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    c = 0
    for v in rng.permutation(n):
        if not unassigned[v]:
            continue
        members = unassigned & plus[v]
        members[v] = True
        labels[members] = c
        unassigned &= ~members
        c += 1
    return labels


def lp_kwikcluster(inst: Instance, dist: np.ndarray, rng=None) -> np.ndarray:
    """dist[i, j] = LP distance x-_ij in [0, 1]; join probability is 1 - dist."""
    rng = _rng(rng)
    n = inst.n
    join = 1.0 - np.clip(dist, 0.0, 1.0)
    labels = np.full(n, -1, dtype=int)
    unassigned = np.ones(n, dtype=bool)
    c = 0
    for v in rng.permutation(n):
        if not unassigned[v]:
            continue
        members = unassigned & (rng.random(n) < join[v])
        members[v] = True
        labels[members] = c
        unassigned &= ~members
        c += 1
    return labels


# Names used in our trailer presentation (identical functions).
pivot = kwikcluster          # "Pivot algorithm"  = ACN's KwikCluster
lp_rounding = lp_kwikcluster  # "LP rounding algorithm" = ACN's LP-KwikCluster; pass x^- from solve_lp
