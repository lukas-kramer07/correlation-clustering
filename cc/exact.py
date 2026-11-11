"""Exact tools for tiny instances: brute-force optimum and exact expected costs.

These avoid Monte Carlo noise when checking the approximation guarantees.
  * brute_force_opt: enumerate all set partitions (Bell(n) of them), n <= 10.
  * expected_kwikcluster: E[cost] of KwikCluster via a DP over vertex subsets (2^n states).
  * expected_lp_kwikcluster: E[cost] of LP-KwikCluster; enumerates the random cluster
    formed around each pivot, O(n 3^n) work, n <= 10.
  * graph_classes: every graph on n vertices up to relabelling, n <= 7.
"""
from __future__ import annotations

import itertools
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


def _canonical_codes(X: np.ndarray, pair_perms: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Smallest integer code of each graph in X (one 0/1 column per pair) over all relabellings."""
    w = 2.0 ** np.arange(X.shape[1])              # exact in float64 for up to 53 pairs
    X = X.astype(float)
    return np.concatenate([(X[i:i + chunk][:, pair_perms] @ w).min(axis=1)
                           for i in range(0, len(X), chunk)])


def graph_classes(n: int) -> np.ndarray:
    """One adjacency matrix per isomorphism class of graphs on n vertices, shape (classes, n, n).

    Built vertex by vertex: every graph on k vertices is a graph on k - 1 vertices plus one
    vertex with some neighbour set, so extending one representative per class in all 2^(k-1)
    ways reaches every class. Duplicates are removed by a canonical code (minimum over all
    k! relabellings). The class counts are 1, 2, 4, 11, 34, 156, 1044 for n = 1..7.
    """
    assert 1 <= n <= 7
    reps = np.zeros((1, 1, 1), dtype=bool)
    for k in range(2, n + 1):
        subsets = np.array(list(itertools.product([False, True], repeat=k - 1)))
        cand = np.zeros((len(reps), len(subsets), k, k), dtype=bool)
        cand[:, :, :k - 1, :k - 1] = reps[:, None]
        cand[:, :, k - 1, :k - 1] = subsets[None]
        cand[:, :, :k - 1, k - 1] = subsets[None]
        cand = cand.reshape(-1, k, k)
        iu = np.triu_indices(k, 1)
        pair_id = np.zeros((k, k), dtype=int)
        pair_id[iu] = np.arange(len(iu[0]))
        pair_id += pair_id.T
        perms = np.array(list(itertools.permutations(range(k))))
        pair_perms = pair_id[perms[:, iu[0]], perms[:, iu[1]]]
        _, first = np.unique(_canonical_codes(cand[:, iu[0], iu[1]], pair_perms), return_index=True)
        reps = cand[np.sort(first)]
    return reps
