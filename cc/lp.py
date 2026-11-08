"""LP relaxation and exact ILP for correlation clustering (Charikar, Guruswami, Wirth 2005).

Variables d_ij in [0, 1] for i < j (d_ij = x-_ij = "i and j are separated").

    minimise   sum_{i<j} w+_ij d_ij + w-_ij (1 - d_ij)
    subject to d_ik <= d_ij + d_jk     for all distinct i, j, k      (triangle inequality)
               0 <= d_ij <= 1

There are 3 * C(n, 3) triangle constraints, so both solvers add them lazily: solve with
the current subset, find violated triangles, add the most violated ones, repeat.  The
final LP solution violates no triangle, so it is optimal for the full LP.  The ILP loop
is the same with integrality; an integral d with no violated triangle is a partition.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix

from .instance import Instance


def _pair_index(n: int):
    iu = np.triu_indices(n, k=1)
    P = -np.ones((n, n), dtype=np.int64)
    P[iu] = np.arange(len(iu[0]))
    P[(iu[1], iu[0])] = np.arange(len(iu[0]))
    return iu, P


def _to_matrix(n, iu, x):
    D = np.zeros((n, n))
    D[iu] = x
    D[(iu[1], iu[0])] = x
    return D


def violated_triangles(D: np.ndarray, tol: float = 1e-7, limit: int | None = None):
    """Triples (i, j, k), i < j, with D[i,j] > D[i,k] + D[k,j] + tol, most violated first."""
    n = D.shape[0]
    found_t, found_v = [], []
    iu0, iu1 = np.triu_indices(n, k=1)
    for k in range(n):
        V = D[iu0, iu1] - D[iu0, k] - D[k, iu1]
        mask = (V > tol) & (iu0 != k) & (iu1 != k)
        if mask.any():
            idx = np.nonzero(mask)[0]
            found_t.append(np.stack([iu0[idx], iu1[idx], np.full(len(idx), k)], axis=1))
            found_v.append(V[idx])
    if not found_t:
        return np.empty((0, 3), dtype=np.int64)
    T, V = np.concatenate(found_t), np.concatenate(found_v)
    order = np.argsort(-V)
    if limit is not None:
        order = order[:limit]
    return T[order]


def _rows(P, triples):
    """Constraint rows  d_ij - d_ik - d_kj <= 0."""
    m = len(triples)
    i, j, k = triples[:, 0], triples[:, 1], triples[:, 2]
    rows = np.repeat(np.arange(m), 3)
    cols = np.stack([P[i, j], P[i, k], P[k, j]], axis=1).ravel()
    vals = np.tile([1.0, -1.0, -1.0], m)
    return rows, cols, vals


def _objective(inst, iu):
    c = inst.w_plus[iu] - inst.w_minus[iu]
    const = float(inst.w_minus[iu].sum())
    return c, const


def solve_lp(inst: Instance, max_rounds: int = 500, per_round: int | None = None,
             tol: float = 1e-7, verbose: bool = False):
    """Return (lp_value, D, info). D[i, j] = x^-_ij (separation); x^+_ij = 1 - D[i, j]."""
    n = inst.n
    t0 = time.perf_counter()
    if n < 3:
        D = (inst.w_plus <= inst.w_minus).astype(float)
        np.fill_diagonal(D, 0)
        return inst.cost(np.arange(n)) if n else 0.0, D, {"rounds": 0, "cuts": 0, "time": 0.0}
    iu, P = _pair_index(n)
    c, const = _objective(inst, iu)
    per_round = per_round or max(200, 10 * n)
    cuts = np.empty((0, 3), dtype=np.int64)
    x = (c < 0).astype(float)  # unconstrained optimum
    for rnd in range(max_rounds):
        D = _to_matrix(n, iu, x)
        new = violated_triangles(D, tol, per_round)
        if len(new) == 0:
            break
        cuts = np.concatenate([cuts, new])
        r, cc_, v = _rows(P, cuts)
        A = csr_matrix((v, (r, cc_)), shape=(len(cuts), len(c)))
        res = linprog(c, A_ub=A, b_ub=np.zeros(len(cuts)), bounds=(0, 1), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        x = np.clip(res.x, 0, 1)
        if verbose:
            print(f"  LP round {rnd}: {len(cuts)} cuts, value {res.fun + const:.4f}")
    else:
        raise RuntimeError("LP cutting-plane loop did not converge")
    D = _to_matrix(n, iu, x)
    val = float(c @ x + const)
    return val, D, {"rounds": rnd, "cuts": int(len(cuts)), "time": time.perf_counter() - t0,
                    "_cuts": cuts}


def _labels_from_integral(D):
    n = D.shape[0]
    labels = -np.ones(n, dtype=int)
    c = 0
    for v in range(n):
        if labels[v] < 0:
            labels[(D[v] < 0.5) & (labels < 0)] = c
            labels[v] = c
            c += 1
    return labels


def solve_ilp(inst: Instance, time_limit: float = 120.0, init_cuts=None,
              max_rounds: int = 200, tol: float = 1e-6):
    """Exact optimum via lazy-constraint MILP (HiGHS).

    Returns (opt_value or None, labels or None, info). None means the time limit was hit.
    """
    n = inst.n
    t0 = time.perf_counter()
    if n < 3:
        return solve_lp(inst)[0], np.arange(n) if n else np.array([], int), {"time": 0.0}
    iu, P = _pair_index(n)
    c, const = _objective(inst, iu)
    if init_cuts is None:
        init_cuts = solve_lp(inst)[2]["_cuts"]
    cuts = np.asarray(init_cuts, dtype=np.int64).reshape(-1, 3)
    for rnd in range(max_rounds):
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            return None, None, {"time": time.perf_counter() - t0, "status": "timeout"}
        r, cc_, v = _rows(P, cuts)
        A = csr_matrix((v, (r, cc_)), shape=(len(cuts), len(c)))
        cons = [LinearConstraint(A, -np.inf, 0.0)] if len(cuts) else []
        res = milp(c, constraints=cons, integrality=np.ones(len(c)), bounds=Bounds(0, 1),
                   options={"time_limit": remaining, "disp": False})
        if res.status != 0 or res.x is None:
            return None, None, {"time": time.perf_counter() - t0, "status": res.message}
        x = np.round(res.x)
        D = _to_matrix(n, iu, x)
        new = violated_triangles(D, tol)
        if len(new) == 0:
            labels = _labels_from_integral(D)
            val = inst.cost(labels)
            assert abs(val - (c @ x + const)) < 1e-6
            return val, labels, {"time": time.perf_counter() - t0, "rounds": rnd,
                                 "cuts": int(len(cuts)), "status": "optimal"}
        cuts = np.concatenate([cuts, new])
    return None, None, {"time": time.perf_counter() - t0, "status": "max_rounds"}
