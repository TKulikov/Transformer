# Retrieval ablation (taskq, decomposed + bootstrap)

layer=19 k=6 corpus=79 queries=45

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] | route_acc |
|---|---|---|---|---|---|
| flat_plain | 0.4 | 0.467 | 0.622 (0.489, 0.756) | 0.459 (0.326, 0.583) | - |
| flat_diverse | 0.4 | 0.578 | 0.622 (0.489, 0.756) | 0.478 (0.35, 0.598) | - |
| flat_centered | 0.422 | 0.556 | 0.556 (0.4, 0.689) | 0.474 (0.337, 0.607) | - |
| flat_centered_diverse | 0.422 | 0.511 | 0.556 (0.4, 0.689) | 0.476 (0.338, 0.609) | - |
| goal_uncentered | 0.533 | 0.733 | 0.8 (0.667, 0.911) | 0.629 (0.497, 0.75) | - |
| goal_centered | 0.578 | 0.711 | 0.778 (0.644, 0.889) | 0.649 (0.513, 0.77) | - |
| goal_prefetch | 0.578 | 0.711 | 0.778 (0.644, 0.889) | 0.649 (0.513, 0.77) | - |
| needroute | 0.2 | 0.2 | 0.2 (0.089, 0.311) | 0.2 (0.089, 0.311) | - |

## Paired contrasts (treatment vs baseline, per query)

| contrast | better | worse | equal |
|---|---|---|---|
| diversification_flat (flat_diverse vs flat_plain) | 7 | 0 | 38 |
| centering_flat (flat_centered vs flat_plain) | 5 | 8 | 32 |
| routing (goal_centered vs flat_centered_diverse) | 15 | 2 | 28 |
| centering_goal (goal_centered vs goal_uncentered) | 5 | 4 | 36 |
| prefetch (goal_prefetch vs goal_centered) | 0 | 0 | 45 |
| oracle_vs_needroute (needroute vs goal_centered) | 0 | 26 | 19 |
