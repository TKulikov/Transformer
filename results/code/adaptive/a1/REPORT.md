# Confidence-adaptive routing (goal when confident, flat fallback)

layer=15 k=6 queries=45 (centroid, uncentered)

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR | goal_mode frac | goal_mode acc |
|---|---|---|---|---|---|---|
| flat_baseline | 0.578 | 0.711 | 0.733 (0.6, 0.844) | 0.649 | 0.0 | None |
| gen_soft_centroid | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | None | None |
| adaptive_t0.3 | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | 1.0 | 0.667 |
| adaptive_t0.5 | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | 1.0 | 0.667 |
| adaptive_t0.7 | 0.578 | 0.667 | 0.667 (0.511, 0.8) | 0.607 | 1.0 | 0.667 |
| adaptive_t0.9 | 0.578 | 0.711 | 0.733 (0.6, 0.844) | 0.649 | 0.0 | None |
| oracle_greedy | 0.689 | 0.8 | 0.867 (0.756, 0.956) | 0.754 | None | None |
