# Soft routing + goal+need blend (vs goal-identification bottleneck)

layer=15 k=6 queries=45 (uncentered goal tree)

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR | route_acc/coverage |
|---|---|---|---|---|---|
| oracle_greedy | 0.689 | 0.8 | 0.867 (0.756, 0.956) | 0.754 | 1.0 |
| gen_greedy | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.333 |
| gen_greedy_blend | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.333 |
| gen_soft_goalonly | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.533 |
| gen_soft_blend | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.533 |
| gen_soft_blend_m6 | 0.267 | 0.333 | 0.333 (0.2, 0.467) | 0.293 | 1.0 |
| oracle_soft_blend | 0.689 | 0.8 | 0.844 (0.733, 0.933) | 0.739 | 1.0 |
