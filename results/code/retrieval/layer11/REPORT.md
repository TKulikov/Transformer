# Retrieval ablation (taskq, decomposed + bootstrap)

layer=11 k=6 corpus=79 queries=45

| condition | hit@1 | hit@3 | hit@6 [95% CI] | MRR [95% CI] | route_acc |
|---|---|---|---|---|---|
| flat_plain | 0.689 | 0.733 | 0.733 (0.6, 0.844) | 0.711 (0.578, 0.833) | - |
| flat_diverse | 0.689 | 0.733 | 0.733 (0.6, 0.844) | 0.711 (0.578, 0.833) | - |
| flat_centered | 0.711 | 0.733 | 0.733 (0.6, 0.844) | 0.722 (0.578, 0.844) | - |
| flat_centered_diverse | 0.711 | 0.733 | 0.733 (0.6, 0.844) | 0.722 (0.578, 0.844) | - |
| goal_uncentered | 0.756 | 0.8 | 0.911 (0.822, 0.978) | 0.799 (0.686, 0.898) | - |
| goal_centered | 0.822 | 0.844 | 0.867 (0.756, 0.956) | 0.837 (0.726, 0.933) | - |
| goal_prefetch | 0.822 | 0.844 | 0.867 (0.756, 0.956) | 0.837 (0.726, 0.933) | - |
| needroute | 0.267 | 0.267 | 0.267 (0.133, 0.4) | 0.267 (0.133, 0.4) | - |

## Paired contrasts (treatment vs baseline, per query)

| contrast | better | worse | equal |
|---|---|---|---|
| diversification_flat (flat_diverse vs flat_plain) | 0 | 0 | 45 |
| centering_flat (flat_centered vs flat_plain) | 3 | 2 | 40 |
| routing (goal_centered vs flat_centered_diverse) | 6 | 0 | 39 |
| centering_goal (goal_centered vs goal_uncentered) | 3 | 2 | 40 |
| prefetch (goal_prefetch vs goal_centered) | 0 | 0 | 45 |
| oracle_vs_needroute (needroute vs goal_centered) | 0 | 27 | 18 |
