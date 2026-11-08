"""Weighted correlation clustering instances on the complete graph.

An instance is a pair of symmetric non-negative matrices (w_plus, w_minus) with zero
diagonal.  A clustering is an integer label vector.  The cost of a clustering is

    sum_{i<j, same cluster}  w_minus[i, j]  +  sum_{i<j, different clusters} w_plus[i, j].

The unweighted problem is the special case w_plus = A, w_minus = 1 - A (off-diagonal),
where A is the 0/1 adjacency matrix of the "+" edges.
"""
from __future__ import annotations

import numpy as np


class Instance:
    def __init__(self, w_plus: np.ndarray, w_minus: np.ndarray, name: str = ""):
        w_plus = np.asarray(w_plus, dtype=float).copy()
        w_minus = np.asarray(w_minus, dtype=float).copy()
        if w_plus.shape != w_minus.shape or w_plus.ndim != 2 or w_plus.shape[0] != w_plus.shape[1]:
            raise ValueError("w_plus and w_minus must be square matrices of equal shape")
        if not (np.allclose(w_plus, w_plus.T) and np.allclose(w_minus, w_minus.T)):
            raise ValueError("weights must be symmetric")
        if (w_plus < 0).any() or (w_minus < 0).any():
            raise ValueError("weights must be non-negative")
        np.fill_diagonal(w_plus, 0.0)
        np.fill_diagonal(w_minus, 0.0)
        self.w_plus, self.w_minus, self.name = w_plus, w_minus, name

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_signs(cls, plus: np.ndarray, name: str = "") -> "Instance":
        """Unweighted instance from a symmetric boolean matrix of '+' edges."""
        plus = np.asarray(plus, dtype=bool)
        plus = plus | plus.T
        np.fill_diagonal(plus, False)
        minus = ~plus
        np.fill_diagonal(minus, False)
        return cls(plus.astype(float), minus.astype(float), name)

    # ------------------------------------------------------------------ properties
    @property
    def n(self) -> int:
        return self.w_plus.shape[0]

    def is_unweighted(self) -> bool:
        off = ~np.eye(self.n, dtype=bool)
        s = self.w_plus + self.w_minus
        return bool(np.isin(self.w_plus[off], (0.0, 1.0)).all() and np.allclose(s[off], 1.0))

    def satisfies_probability(self, tol: float = 1e-9) -> bool:
        off = ~np.eye(self.n, dtype=bool)
        return bool(np.abs((self.w_plus + self.w_minus)[off] - 1.0).max(initial=0.0) <= tol)

    def satisfies_triangle(self, tol: float = 1e-9) -> bool:
        """Triangle inequality on w_minus: w-_ik <= w-_ij + w-_jk (ACN constraint (ii))."""
        W = self.w_minus
        for k in range(self.n):
            V = W - W[:, [k]] - W[[k], :]
            V[k, :] = V[:, k] = -np.inf
            if V.max() > tol:
                return False
        return True

    def majority(self) -> np.ndarray:
        """Boolean '+' matrix of the unweighted majority instance (ties go to '-')."""
        plus = self.w_plus > self.w_minus
        np.fill_diagonal(plus, False)
        return plus

    # ------------------------------------------------------------------ objective
    def cost(self, labels) -> float:
        labels = np.asarray(labels)
        same = labels[:, None] == labels[None, :]
        iu = np.triu_indices(self.n, k=1)
        return float(np.where(same, self.w_minus, self.w_plus)[iu].sum())

    def trivial_lower_bound(self) -> float:
        """sum_{i<j} min(w+, w-): every pair pays at least its smaller weight."""
        iu = np.triu_indices(self.n, k=1)
        return float(np.minimum(self.w_plus, self.w_minus)[iu].sum())

    def __repr__(self) -> str:
        return f"Instance(name={self.name!r}, n={self.n})"


def canonical(labels) -> np.ndarray:
    """Relabel clusters 0,1,2,... in order of first appearance."""
    labels = np.asarray(labels)
    out = np.empty(len(labels), dtype=int)
    seen: dict = {}
    for i, l in enumerate(labels):
        out[i] = seen.setdefault(l, len(seen))
    return out
