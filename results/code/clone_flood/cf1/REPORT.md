# Clone-flooding: MMR lambda sweep (utils, goal-routed uncentered)

layer=15 k=10

| target | λ=0.0 | λ=0.3 | λ=0.5 | λ=0.7 | λ=1.0 |
|---|---|---|---|---|---|
| utils.backoff_delay | 1 | 1 | 1 | 1 | 1 |
| utils.retry | 6 | 10 | 10 | 6 | 3 |
| utils.parse_duration | 6 | 3 | 2 | 2 | 2 |
| utils.dedupe | 4 | 3 | 3 | 3 | 3 |
| utils.clamp | 1 | 1 | 1 | 1 | 1 |
| utils.chunked | 2 | 2 | 2 | 2 | 2 |
| **hit@6** | 1.0 | 0.833 | 0.833 | 1.0 | 1.0 |
