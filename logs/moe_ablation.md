MoE ablation: nano, TinyStories head (2500 docs), 300 steps, B16 T128, seeds [0, 1]

| arm | active params/token | val s0 | val s1 | mean val loss | Δ vs first |
|---|---|---|---|---|---|
| A base(n_group=2) | 343,808 | 3.8578 | 3.8448 | **3.8513** | +0.0000 |
| B wide-shared | 344,320 | 3.8548 | 3.8509 | **3.8529** | +0.0015 |
| C base+n_group=1 | 343,808 | 3.8324 | 3.8345 | **3.8335** | -0.0179 |
| D wide+n_group=1 | 344,320 | 3.8317 | 3.8329 | **3.8323** | -0.0190 |
| E D+scalar-gate | 344,324 | 3.8393 | 3.8317 | **3.8355** | -0.0158 |
| F D+token-gate | 344,576 | 3.8543 | 3.8268 | **3.8405** | -0.0108 |
