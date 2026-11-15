# Retrieval ablation (taskq, decomposed + bootstrap)

layer=15 k=6 corpus=79 queries=45

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] | route_acc |
|---|---|---|---|---|---|
| flat_plain | 0.578 | 0.733 | 0.733 (0.6, 0.844) | 0.652 (0.522, 0.774) | - |
| flat_diverse | 0.578 | 0.711 | 0.733 (0.6, 0.844) | 0.649 (0.52, 0.771) | - |
| flat_centered | 0.578 | 0.644 | 0.644 (0.489, 0.778) | 0.611 (0.467, 0.744) | - |
| flat_centered_diverse | 0.578 | 0.6 | 0.644 (0.489, 0.778) | 0.596 (0.456, 0.733) | - |
| goal_uncentered | 0.689 | 0.8 | 0.867 (0.756, 0.956) | 0.754 (0.632, 0.858) | - |
| goal_centered | 0.733 | 0.756 | 0.8 (0.667, 0.911) | 0.753 (0.62, 0.871) | - |
| goal_prefetch | 0.733 | 0.756 | 0.8 (0.667, 0.911) | 0.753 (0.62, 0.871) | - |
| needroute | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 (0.133, 0.4) | - |
| goal_generated | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 (0.133, 0.4) | 0.333 |

## Paired contrasts (treatment vs baseline, per query)

| contrast | better | worse | equal |
|---|---|---|---|
| diversification_flat (flat_diverse vs flat_plain) | 1 | 1 | 43 |
| centering_flat (flat_centered vs flat_plain) | 7 | 8 | 30 |
| routing (goal_centered vs flat_centered_diverse) | 9 | 0 | 36 |
| centering_goal (goal_centered vs goal_uncentered) | 4 | 3 | 38 |
| prefetch (goal_prefetch vs goal_centered) | 0 | 0 | 45 |
| oracle_vs_needroute (needroute vs goal_centered) | 0 | 24 | 21 |
| oracle_vs_generated (goal_generated vs goal_centered) | 0 | 24 | 21 |
