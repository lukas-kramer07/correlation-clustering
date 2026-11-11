"""Instances built from real data sets bundled with scikit-learn (no download needed).

similarity_instance : unweighted.  Standardise features; pair (i, j) is '+' iff the
                      Euclidean distance is below the q-quantile of all pairwise distances,
                      with q chosen equal to the fraction of same-class pairs.  This is the
                      'noisy pairwise classifier' view used in deduplication / entity
                      resolution.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from .instance import Instance

LOADERS = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
}


def load(name: str, n: int | None = None, rng=None):
    rng = np.random.default_rng(rng)
    d = LOADERS[name]()
    X, y = d.data.astype(float), d.target
    if n is not None and n < len(y):
        idx = np.sort(rng.choice(len(y), size=n, replace=False))
        X, y = X[idx], y[idx]
    return StandardScaler().fit_transform(X), y


def similarity_instance(name: str, n: int | None = None, rng=None):
    X, y = load(name, n, rng)
    D = squareform(pdist(X))
    iu = np.triu_indices(len(y), k=1)
    same_frac = float((y[iu[0]] == y[iu[1]]).mean())
    thr = np.quantile(D[iu], same_frac)
    return Instance.from_signs(D <= thr, f"{name}-sim(n={len(y)})"), y

