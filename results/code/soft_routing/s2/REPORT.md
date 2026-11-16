# Soft routing + goal+need blend (vs goal-identification bottleneck)

layer=15 k=6 queries=45 (uncentered goal tree)

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR | route_acc/coverage |
|---|---|---|---|---|---|
| oracle_greedy | 0.689 | 0.8 | 0.867 (0.756, 0.956) | 0.754 | 1.0 |
| gen_greedy | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.333 |
| gen_soft_blend | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 | 0.533 |
| gen_soft_blend_m6 | 0.267 | 0.333 | 0.333 (0.2, 0.467) | 0.293 | 1.0 |
| gen_greedy_centroid | 0.511 | 0.6 | 0.6 (0.444, 0.733) | 0.552 | 0.667 |
| gen_soft_centroid | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | 0.933 |
| gen_soft_centroid_m6 | 0.578 | 0.6 | 0.667 (0.511, 0.8) | 0.596 | 1.0 |
| gen_soft_centroid_goalonly | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | 0.933 |
| oracle_soft_centroid | 0.2 | 0.378 | 0.489 (0.333, 0.622) | 0.305 | 0.889 |
