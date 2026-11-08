"""Exact tools for tiny instances: brute-force optimum and exact expected costs.

These avoid Monte Carlo noise when checking the approximation guarantees.
  * brute_force_opt: enumerate all set partitions (Bell(n) of them), n <= 10.
  * expected_kwikcluster: E[cost] of KwikCluster via a DP over vertex subsets (2^n states).
  * expected_lp_kwikcluster: E[cost] of LP-KwikCluster; enumerates the random cluster
    formed around each pivot, O(n 3^n) work, n <= 10.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .instance import Instance


def set_partitions(n: int) -> np.ndarray:
    """All restricted-growth strings of length n (one per set partition)."""
    out = []
    a = [0] * n

    def rec(i, m):
        if i == n:
            out.append(a.copy())
            return
        for v in range(m + 1):
            a[i] = v
            rec(i + 1, max(m, v + 1))
    if n == 0:
        return np.zeros((1, 0), dtype=int)
    a[0] = 0
    rec(1, 1)
    return np.array(out, dtype=int)


def brute_force_opt(inst: Instance, parts: np.ndarray | None = None):
    n = inst.n
    parts = set_partitions(n) if parts is None else parts
    iu = np.triu_indices(n, k=1)
    same = (parts[:, iu[0]] == parts[:, iu[1]]).astype(float)
    costs = same @ inst.w_minus[iu] + (1 - same) @ inst.w_plus[iu]
    b = int(np.argmin(costs))
    return float(costs[b]), parts[b]


def _cluster_cost(inst, C_mask, S_mask, members, rest):
    Wm, Wp = inst.w_minus, inst.w_plus
    inside = sum(Wm[a, b] for x, a in enumerate(members) for b in members[x + 1:])
    across = sum(Wp[a, b] for a in members for b in rest)
    return inside + across


def expected_kwikcluster(inst: Instance) -> float:
    n = inst.n
    assert n <= 20
    plus = inst.majority()
    nbr = [sum(1 << j for j in range(n) if plus[v, j]) for v in range(n)]

    @lru_cache(maxsize=None)
    def E(S: int) -> float:
        if S == 0:
            return 0.0
        verts = [v for v in range(n) if S >> v & 1]
        tot = 0.0
        for k in verts:
            C = (nbr[k] & S) | (1 << k)
            members = [v for v in verts if C >> v & 1]
            rest = [v for v in verts if not C >> v & 1]
            tot += _cluster_cost(inst, C, S, members, rest) + E(S & ~C)
        return tot / len(verts)

    return E((1 << n) - 1)


def expected_lp_kwikcluster(inst: Instance, dist: np.ndarray) -> float:
    n = inst.n
    assert n <= 10
    q = 1.0 - np.clip(dist, 0, 1)

    @lru_cache(maxsize=None)
    def E(S: int) -> float:
        if S == 0:
            return 0.0
        verts = [v for v in range(n) if S >> v & 1]
        tot = 0.0
        for k in verts:
            others = [v for v in verts if v != k]
            m = len(others)
            for t in range(1 << m):
                p = 1.0
                members, rest = [k], []
                for b, v in enumerate(others):
                    if t >> b & 1:
                        p *= q[k, v]
                        members.append(v)
                    else:
                        p *= 1 - q[k, v]
                        rest.append(v)
                if p == 0.0:
                    continue
                C = sum(1 << v for v in members)
                tot += p * (_cluster_cost(inst, C, S, members, rest) + E(S & ~C))
        return tot / len(verts)

    return E((1 << n) - 1)
