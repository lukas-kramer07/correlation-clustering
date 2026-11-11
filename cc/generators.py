"""Synthetic instance families."""
from __future__ import annotations

import numpy as np

from .instance import Instance


def _sym_bernoulli(n, p, rng):
    U = rng.random((n, n))
    A = np.triu(U < p, k=1)
    return A | A.T


def random_signed(n: int, p: float = 0.5, rng=None) -> Instance:
    """Each pair is '+' independently with probability p (no planted structure)."""
    rng = np.random.default_rng(rng)
    return Instance.from_signs(_sym_bernoulli(n, p, rng), f"random(n={n},p={p})")


def planted(sizes, eps: float, rng=None):
    """Planted partition; each pair's sign is flipped with probability eps.

    eps = 0 is the best case (a perfectly consistent instance, OPT = 0).  This also models
    entity resolution: a noisy pairwise 'same entity?' classifier with error rate eps.
    """
    rng = np.random.default_rng(rng)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    n = len(truth)
    plus = truth[:, None] == truth[None, :]
    flip = _sym_bernoulli(n, eps, rng)
    return Instance.from_signs(plus ^ flip, f"planted({list(sizes)},eps={eps})"), truth


def complete_minus_edge(n: int) -> Instance:
    """K_n with all '+' except one '-' edge (0, 1).  OPT = 1 and E[KwikCluster] = 3 - 6/n."""
    plus = np.ones((n, n), dtype=bool)
    plus[0, 1] = plus[1, 0] = False
    return Instance.from_signs(plus, f"K{n}-e")


def star(leaves: int) -> Instance:
    """Centre 0 joined by '+' to every leaf; leaves pairwise '-'.

    OPT = leaves - 1, LP = leaves / 2 (d = 1/2 on spokes), so the LP gap tends to 2
    (Charikar, Guruswami, Wirth 2005).
    """
    n = leaves + 1
    plus = np.zeros((n, n), dtype=bool)
    plus[0, 1:] = plus[1:, 0] = True
    return Instance.from_signs(plus, f"star({leaves})")

