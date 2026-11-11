import itertools

import numpy as np
import pytest

from cc import Instance, kwikcluster, lp_kwikcluster, solve_ilp, solve_lp
from cc.exact import brute_force_opt, expected_kwikcluster, expected_lp_kwikcluster, set_partitions
from cc.generators import complete_minus_edge, planted, random_signed, star


def test_cost_counts_disagreements():
    plus = np.zeros((3, 3), bool)
    plus[0, 1] = plus[1, 2] = True        # bad triangle: + + -
    inst = Instance.from_signs(plus)
    assert inst.cost([0, 0, 0]) == 1      # the '-' edge (0,2) inside
    assert inst.cost([0, 1, 2]) == 2      # both '+' edges cut
    assert inst.cost([0, 0, 1]) == 1


def test_bell_numbers():
    assert [len(set_partitions(n)) for n in range(1, 8)] == [1, 2, 5, 15, 52, 203, 877]


@pytest.mark.parametrize("seed", range(8))
def test_lp_le_ilp_eq_bruteforce(seed):
    rng = np.random.default_rng(seed)
    inst = random_signed(7, 0.5, rng)
    lp, D, _ = solve_lp(inst)
    opt, lab, _ = solve_ilp(inst)
    bf, _ = brute_force_opt(inst)
    assert lp <= opt + 1e-7
    assert abs(opt - bf) < 1e-7
    assert abs(inst.cost(lab) - opt) < 1e-7
    for i, j, k in itertools.permutations(range(7), 3):
        assert D[i, k] <= D[i, j] + D[j, k] + 1e-6


def test_exact_expectation_matches_simulation():
    inst = random_signed(7, 0.5, 3)
    _, D, _ = solve_lp(inst)
    rng = np.random.default_rng(0)
    R = 20000
    mc_p = np.mean([inst.cost(kwikcluster(inst, rng)) for _ in range(R)])
    mc_l = np.mean([inst.cost(lp_kwikcluster(inst, D, rng)) for _ in range(R)])
    assert abs(mc_p - expected_kwikcluster(inst)) < 0.05
    assert abs(mc_l - expected_lp_kwikcluster(inst, D)) < 0.05


@pytest.mark.parametrize("n", [4, 6, 9])
def test_kn_minus_edge_closed_form(n):
    inst = complete_minus_edge(n)
    assert abs(expected_kwikcluster(inst) - (3 - 6 / n)) < 1e-9
    assert brute_force_opt(inst)[0] == 1


def test_star_lp_gap():
    inst = star(6)
    lp, D, _ = solve_lp(inst)
    assert abs(lp - 3.0) < 1e-7
    assert brute_force_opt(inst)[0] == 5


def test_best_case_recovered_exactly():
    inst, truth = planted([5, 7, 9], 0.0, 1)
    for s in range(5):
        lab = kwikcluster(inst, s)
        assert inst.cost(lab) == 0
    lp, D, _ = solve_lp(inst)
    assert lp == pytest.approx(0)
    assert inst.cost(lp_kwikcluster(inst, D, 0)) == 0


def test_presentation_aliases():
    from cc import lp_rounding, pivot
    assert pivot is kwikcluster and lp_rounding is lp_kwikcluster
