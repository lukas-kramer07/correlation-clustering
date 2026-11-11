# Correlation clustering: Pivot vs. LP rounding

Implementation and numerical tests of the two pivot algorithms of
Ailon, Charikar and Newman, *Aggregating Inconsistent Information: Ranking and Clustering*,
J. ACM 55(5), 2008, on unweighted complete graphs:

| Algorithm | Rule after choosing a random pivot `k` | Guarantee |
|---|---|---|
| **Pivot** (`pivot`, ACN's KwikCluster) | put every unclustered `+` neighbour of `k` in its cluster | E[cost] ≤ 3·OPT |
| **LP rounding** (`lp_rounding`, ACN's LP-KwikCluster) | put each unclustered `j` in `k`'s cluster independently with probability `x+_kj = 1 − x−_kj`, where `x` is an optimal solution of the Charikar–Guruswami–Wirth LP | E[cost] ≤ 2.5·LP ≤ 2.5·OPT |

## Layout

```
cc/instance.py     instances and the cost function
cc/algorithms.py   pivot (kwikcluster) and lp_rounding (lp_kwikcluster)
cc/lp.py           LP relaxation and exact ILP, both with lazy triangle constraints (HiGHS)
cc/exact.py        brute-force OPT, exact E[cost] of both algorithms, all graphs up to relabelling (tiny n)
cc/generators.py   random, planted (best case), K_n minus an edge, star (worst cases)
cc/datasets.py     similarity graphs from scikit-learn data sets (Iris, Wine, Breast Cancer)
experiments/       run_experiments.py (all results), exhaustive_search.py, summarize.py
results/           CSV output of the experiments
tests/             pytest suite
```

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q tests
cd experiments
python run_experiments.py          # ~10-15 min, writes results/*.csv
python exhaustive_search.py        # ~15 s, exact check of every sign pattern on up to 7 vertices
```

## Headline results

* `K_n` minus one edge: E[Pivot] = (3 - 6/n)·OPT, so the 3 bound is tight; LP rounding pays OPT.
* Star with m leaves: E[LP rounding] / LP -> 2.5, so the 2.5 bound (relative to LP) is tight.
* An exhaustive exact check of every sign pattern on up to 7 vertices (1,044 graphs up to relabelling at n = 7) finds nothing worse.
  For each n from 4 to 7 the maximum of E[Pivot] / OPT is exactly 3 - 6/n, the value of `K_n` minus one edge, and the unique
  maximiser of E[LP rounding] / LP is the star. The maximum of E[LP rounding] / OPT is 1.64 (for the LP optimum the solver returns).
* On random, planted and real instances: mean cost / LP of at most 2.54 (Pivot) and 1.97 (LP rounding).
* Planted clusters with 5% noise are the hardest case for Pivot (about 2.4-2.5·OPT); the LP is integral
  there, so LP rounding is optimal in every run. At 20-40% noise LP rounding gains nothing over Pivot.

Seeds are fixed. The LP optimum is not always unique; a different HiGHS version can return
a different optimal vertex, which can shift LP-rounding numbers slightly.

## Notes on the implementation

* The LP has `C(n,2)` variables and `3·C(n,3)` triangle constraints. `solve_lp` starts with
  none, adds the most violated triangles, and re-solves until no triangle is violated; the
  final point is then optimal for the full LP. `n = 100` takes up to about 2.5 minutes.
* `solve_ilp` uses the same lazy loop with integrality, seeded with the LP's cuts. It is
  practical up to roughly `n = 20` on unstructured random graphs and `n = 40` on planted ones.
* `cc/exact.py` computes expected costs exactly (DP over vertex subsets), so tiny-instance
  ratios carry no sampling noise.

## License

MIT, see [LICENSE](LICENSE).
