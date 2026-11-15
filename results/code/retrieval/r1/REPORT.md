# Retrieval ablation (taskq)

layer=15 k=6 corpus=79 functions tasks=15

| condition | hit@1 | hit@3 | hit@6 | MRR |
|---|---|---|---|---|
| flat_plain | 0.533 | 0.667 | 0.667 | 0.6 |
| flat_centered | 0.533 | 0.6 | 0.6 | 0.567 |
| goal_routed | 0.733 | 0.8 | 0.8 | 0.767 |
| goal_prefetch | 0.733 | 0.8 | 0.8 | 0.767 |
