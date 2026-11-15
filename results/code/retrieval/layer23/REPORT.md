# Retrieval ablation (taskq, decomposed + bootstrap)

layer=23 k=6 corpus=79 queries=45

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] | route_acc |
|---|---|---|---|---|---|
| flat_plain | 0.422 | 0.6 | 0.644 (0.511, 0.778) | 0.507 (0.378, 0.637) | - |
| flat_diverse | 0.422 | 0.644 | 0.644 (0.511, 0.778) | 0.522 (0.389, 0.648) | - |
| flat_centered | 0.578 | 0.622 | 0.622 (0.489, 0.756) | 0.6 (0.456, 0.733) | - |
| flat_centered_diverse | 0.578 | 0.578 | 0.622 (0.489, 0.756) | 0.585 (0.441, 0.719) | - |
| goal_uncentered | 0.467 | 0.711 | 0.867 (0.756, 0.956) | 0.601 (0.48, 0.71) | - |
| goal_centered | 0.6 | 0.756 | 0.867 (0.756, 0.956) | 0.69 (0.567, 0.804) | - |
| goal_prefetch | 0.6 | 0.756 | 0.867 (0.756, 0.956) | 0.69 (0.567, 0.804) | - |
| needroute | 0.156 | 0.156 | 0.156 (0.044, 0.267) | 0.156 (0.044, 0.267) | - |

## Paired contrasts (treatment vs baseline, per query)

| contrast | better | worse | equal |
|---|---|---|---|
| diversification_flat (flat_diverse vs flat_plain) | 4 | 2 | 39 |
| centering_flat (flat_centered vs flat_plain) | 9 | 5 | 31 |
| routing (goal_centered vs flat_centered_diverse) | 13 | 0 | 32 |
| centering_goal (goal_centered vs goal_uncentered) | 8 | 5 | 32 |
| prefetch (goal_prefetch vs goal_centered) | 0 | 0 | 45 |
| oracle_vs_needroute (needroute vs goal_centered) | 0 | 32 | 13 |
