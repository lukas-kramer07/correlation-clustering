# Correlation clustering: Pivot vs. LP rounding

Implementation and empirical study of the two pivot algorithms of
Ailon, Charikar and Newman, *Aggregating Inconsistent Information: Ranking and Clustering*,
J. ACM 55(5), 2008:

| Algorithm | Rule after choosing a random pivot `k` | Guarantee (unweighted) |
|---|---|---|
| **Pivot** (`pivot`, ACN's KwikCluster) | put every unclustered `+` neighbour of `k` in its cluster | E[cost] ≤ 3·OPT |
| **LP rounding** (`lp_rounding`, ACN's LP-KwikCluster) | put each unclustered `j` in `k`'s cluster independently with probability `x+_kj = 1 − x−_kj`, where `x` is an optimal solution of the Charikar–Guruswami–Wirth LP | E[cost] ≤ 2.5·LP ≤ 2.5·OPT |

Weighted instances with probability constraints (`w+ + w− = 1`): 5 and 2.5.
With probability constraints plus the triangle inequality: 2 and 2.

## Layout

```
cc/instance.py     instances, cost function, constraint checks, majority instance
cc/algorithms.py   kwikcluster, lp_kwikcluster, pick_a_cluster, local_search
cc/lp.py           LP relaxation and exact ILP, both with lazy triangle constraints (HiGHS)
cc/exact.py        brute-force OPT, exact E[cost] of both algorithms for tiny n
cc/generators.py   random, planted (best case), K_n minus an edge, star (worst cases), weighted
cc/datasets.py     real instances from scikit-learn data sets (similarity and consensus)
experiments/       run_experiments.py (all results), adversarial_search.py, summarize.py
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
python adversarial_search.py 7     # exact hill-climbing search on 7-vertex instances
```

## Headline results

* `K_n` minus one edge: E[Pivot] = (3 - 6/n)·OPT, so the 3 bound is tight; LP rounding pays OPT.
* Star with m leaves: E[LP rounding] / LP -> 2.5, so the 2.5 bound (relative to LP) is tight.
* An exact adversarial search on 7 vertices finds nothing worse: its Pivot maximum is 15/7, the value of `K_7` minus one edge (attained on `K_7` minus three disjoint edges), and its LP rounding maximum is the 6-leaf star.
* On random, planted, weighted and real instances: mean ratios to LP of 1.0-2.5 (Pivot) and 1.0-1.96 (LP rounding);
  LP rounding is optimal whenever the LP is integral (low noise, consensus data).

Seeds are fixed. The LP optimum is not always unique; a different HiGHS version can return
a different optimal vertex, which can shift LP-rounding numbers slightly.

## Notes on the implementation

* The LP has `C(n,2)` variables and `3·C(n,3)` triangle constraints. `solve_lp` starts with
  none, adds the most violated triangles, and re-solves until no triangle is violated; the
  final point is then optimal for the full LP. `n = 100` takes from seconds
  (consensus) to about two minutes (similarity graphs).
* `solve_ilp` uses the same lazy loop with integrality, seeded with the LP's cuts. It is
  practical up to roughly `n = 20` on unstructured random graphs and `n = 40` on planted ones.
* `cc/exact.py` computes expected costs exactly (DP over vertex subsets), so tiny-instance
  ratios carry no sampling noise.

## License

MIT, see [LICENSE](LICENSE).
