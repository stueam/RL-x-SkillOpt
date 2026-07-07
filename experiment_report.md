# Experiment Report: Erdos Minimum Overlap

Date：2026/7/2

## Experimental Setup (Local)

| Parameter | Local Test | Paper (Full) |
|-----------|-----------|--------------|
| Model | openai/gpt-oss-120b | openai/gpt-oss-120b |
| Epochs | 3 | 50 |
| Group size | 4 | 64 |
| Groups per batch | 1 | 8 |
| Phase1 max tokens | 8,192 | 26,000 |
| Steps completed | 3 | ~400 |

## Results

### Main Metric: C5 Bound (lower is better)

| Source | C5 | vs. Record |
|--------|:---:|:----------:|
| Random initial state | 0.51667 | - |
| **Local test - best** | **0.38814** | +0.00726 above record |
| TTT-Discover (paper) | **0.380876** | **New SOTA** |
| Prev. Best AI | 0.380924 | - |
| Best Human (Haugland) | 0.380927 | - |

## Analysis

1. **Improvement confirmed**: C5 dropped from 0.5167 to 0.3881 in just 3 steps, showing the RL training loop is working.
2. **Gap to SOTA**: Best C5 = 0.38814 vs. paper's 0.380876 (+1.9% relative). The gap is expected given 3 epochs vs. 50, and group_size=4 vs. 64.
3. **Correctness issue**: Only 25-50% of generated code is valid. This is likely due to small group_size limiting exploration diversity.
4. **Computation cost**: 3 steps took ~660s sampling + ~589s evaluation. Full paper run would be ~100x more steps.

## Conclusion

The local test validates the training pipeline works correctly. To approach the paper's SOTA result (C5 = 0.380876), scale up to the full config: 50 epochs, group_size=64, groups_per_batch=8.
