"""Instances built from real data sets bundled with scikit-learn (no download needed).

similarity_instance : unweighted.  Standardise features; pair (i, j) is '+' iff the
                      Euclidean distance is below the q-quantile of all pairwise distances,
                      with q chosen equal to the fraction of same-class pairs.  This is the
                      'noisy pairwise classifier' view used in deduplication / entity
                      resolution.
consensus_instance  : weighted (ACN constraint (iii), aggregation).  Run m k-means
                      clusterings with random k and random feature subsets; w+_ij is the
                      fraction of clusterings that put i and j together, w-_ij = 1 - w+_ij.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .instance import Instance

LOADERS = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
    "digits": datasets.load_digits,
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


def consensus_instance(name: str, n: int | None = None, m: int = 20, rng=None):
    rng = np.random.default_rng(rng)
    X, y = load(name, n, rng)
    clusterings = []
    for _ in range(m):
        k = int(rng.integers(2, 7))
        feats = rng.choice(X.shape[1], size=max(2, X.shape[1] // 2), replace=False)
        km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(1 << 31)))
        clusterings.append(km.fit_predict(X[:, feats]))
    C = np.array(clusterings)
    Wp = (C[:, :, None] == C[:, None, :]).mean(axis=0)
    Wm = 1 - Wp
    inst = Instance(Wp, Wm, f"{name}-consensus(n={len(y)},m={m})")
    return inst, y, clusterings
