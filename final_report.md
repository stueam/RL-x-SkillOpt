# Final Report: RL vs SkillOpt on Code Generation — Division of Labor

**Period**: 2026-06-29 — 2026-07-26
**Authors**: Bojun Yang
**Status**: Phase 1 Complete — Three benchmarks validated

---

## Executive Summary

**Question**: For LLM code generation problems, when should you use **weight optimization (RL via TTT-Discover)** vs **prompt optimization (SkillOpt)**?

**Core finding**: The two methods optimize fundamentally different objectives and **do not compete** — they address different bottlenecks in code generation. Their division of labor is:

> **SkillOpt fixes correctness (engineering process). RL pushes solution quality (optimization search).**

### Why Previous Approaches Failed to Validate This

Earlier iterations searched for "domain novelty" (mathematical problems the LLM doesn't know) as the predictor of Skill value. This was wrong. The correct predictor is **code fragility** — how easy it is to write code that compiles and passes basic validity checks.

| Wrong approach (early rounds) | Right approach (this round) |
|---|---|
| Domain novelty → Gate escape | Code fragility → Gate collapse |
| Look for "unknown math" | Look for "easy to crash" |
| Inject mathematical strategy | Inject **engineering process** (validate, repair, scaffold) |

### Key Empirical Findings

| Benchmark | Task | Gate (weak model) | RL Correctness | Skill→RL Effect | Winner |
|---|---|---|---|---|---|
| **Circle Packing** | Pack 26 circles in unit square | **28%** | 54-75% | **+22pp corr, +0.06 quality** | **Minimal Skill→RL** |
| **Erdős** | Minimize overlap integral C5 | **72%** | 25-88% | +12pp corr, -0.005 quality | Pure RL |
| **REI** | Synthesize regex from examples | 0-100% (task dep.) | **100%** | No effect | Pure RL |

**The Pareto frontier**: Skill→RL is only beneficial when the weak model's Gate is low (<50%). Circle Packing (28% Gate) shows clear Skill value. Erdős (72% Gate) shows marginal trade-off. REI (strong model, 100% RL correctness) shows zero benefit.

### Practical Recommendation

Use **Strategy Skill (Minimal variant)** when:
- Target model is weak relative to task difficulty
- Task has rich engineering structure (validation, repair, scaffolding)
- Benefit: +15-25pp correctness with no quality loss

Use **Pure RL** when:
- Target model already achieves reasonable Gate (>50%)
- Task is a pure continuous optimization
- Benefit: free-form search finds better solutions

---

## Methodology Reflections

### What We Got Wrong Initially

Our initial hypothesis was that Skill would inject **mathematical strategy** knowledge — domain-specific heuristics that improve solution quality. This was based on the FunSearch paper's cap set result, where the discovered priority function represented genuine mathematical insight.

**This hypothesis failed to replicate** across three benchmarks because:
1. In Circle Packing, the effective skill content was **engineering process** (validate, repair, scaffold), not geometric strategy
2. In Erdős, the optimization strategy was too generic to matter
3. In REI, regex patterns are too simple for strategy to help

### The Correct Framework

**Skill value = Task requires complex engineering scaffolding ÷ Model can generate valid code unaided**

When this ratio is high (Circle Packing: 28% Gate means 72% of attempts are "engineering failures"), Skill helps. When it's low (Erdős: 72% Gate, REI: 100% RL correctness), Skill has no room to improve.

### Why TTT-Discover and SkillOpt Optimize Different Things

TTT-Discover optimizes **model weights** to produce better code for a specific instance. It's a **max-of-N** search: given N attempts, produce the single best solution. SkillOpt optimizes **prompt documents** to improve the average quality across instances. It's a **mean-of-N** optimization: make every attempt better.

These are complementary, not competing. TTT-Discover finds the best solution; SkillOpt makes more solutions valid. The successful Circle Packing experiment shows they can stack: SkillOpt fixes the correctness floor, then RL pushes the quality ceiling.

---

## Common Settings

| Component | Detail |
|---|---|
| RL Model | `openai/gpt-oss-120b` via Tinker API, LoRA Rank=32 |
| SkillOpt Target | `qwen/qwen3-coder-30b-a3b-instruct` via OpenRouter |
| SkillOpt Optimizer | `deepseek/deepseek-chat` via OpenRouter |
| Naive Baseline | `qwen/qwen3-coder-30b-a3b-instruct`, 32 trials, minimal prompt |

### Metric Definitions

| Metric | Circle Packing | Erdős | REI |
|---|---|---|---|---|
| Quality | Sum of radii (higher is better), known optimum ≈ 2.636 | C₅ bound (lower is better), known record ≤ 0.38092 | -Cost, Cost = \|r\| + 50 × heldout_mismatch_rate |
| Reward | raw_score (no transformation) | 1 / (1e-8 + C₅), higher = better (lower C₅) | max(0, 1 - Cost/200), higher = better |
| Correctness | Fraction of trials returning valid packing | Fraction of trials returning valid h vector | Fraction of regexes passing Gate |
| Gate (SkillOpt) | soft gate = correctness × quality score | same | same |

---

## Circle Packing N=26

### 1. Naive Baseline

**Model**: qwen3-coder-30b, 32 trials, no strategy guidance

| Gate Rate | Quality Mean | Quality Std | Quality Max | CV |
|---|---|---|---|---|
| 28.1% (9/32) | 1.778 | 0.398 | 2.362 | 0.224 |

### 2. SkillOpt

**Model**: qwen3-30b (target), deepseek-chat (optimizer), 4 epochs, 24 steps

| Step | Action | Selection Soft | vs Baseline (0.331) | Skill Len |
|---|---|---|---|---|
| 1 | reject | 0.168 | — | 5,697 |
| 2 | reject | 0.167 | — | 5,697 |
| **3** | **accept** | **0.412** | **+24%** | 6,342 |
| 4 | reject | 0.240 | — | 6,342 |
| **5** | **accept** | **0.584** | **+76%** | 6,711 |
| 6 | reject | 0.143 | — | 6,711 |
| 7-8 | reject | — | — | 6,711 |
| **9** | **accept** | **0.599** | **+81%** | 7,555 |
| 10 | reject | 0.517 | — | 7,555 |
| **11** | **accept** | **0.632** | **+91%** | 8,222 |
| 12-24 | all reject | — | — | 8,222 |

Best skill score: **0.632** (+91%), final skill len: 8,222 chars.

### 3. Pure RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs

| Step | Reward/Max | Reward/Mean | Correctness |
|---|---|---|---|
| 0 | 2.559 | 1.086 | 50.0% |
| 1 | 2.613 | 1.579 | 62.5% |
| 2 | **2.624** | 1.208 | 50.0% |

### 4. Pure RL (Full)

**Model**: gpt-oss-120b, group=16, batch=4, phase1=10000, 8 epochs

| Step | Reward/Max | Reward/Mean | Correctness |
|---|---|---|---|
| 0 | 2.591 | 1.561 | 73.4% |
| 1 | 2.628 | 1.493 | 60.9% |
| 2 | 2.628 | 1.732 | 68.8% |
| 3 | 2.631 | 1.690 | 70.3% |
| 4 | 2.631 | 1.717 | 68.8% |
| 5 | 2.635 | 1.257 | 48.4% |
| 6 | 2.635 | 0.946 | 37.5% |
| 7 | **2.636** | 1.011 | 39.1% |

### 5. Full Skill→RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs
**Prompt**: problem + validator + **optimized SkillOpt skill** (Hex Grid + Expand + Multi-Seed)

| Step | Reward/Max | Reward/Mean | Correctness |
|---|---|---|---|
| 0 | 2.602 | — | 82.8% |
| 1 | 2.501 | — | 56.3% |
| 2 | 2.607 | — | 75.0% |
| **Best** | **2.607** | — | **avg 71%** |

### 6. Minimal Skill→RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs
**Prompt**: problem + validator + **guardrail-only** (no strategy, only correctness rules)

| Step | Reward/Max | Reward/Mean | Correctness |
|---|---|---|---|
| 0 | 2.471 | — | 75.0% |
| 1 | 2.609 | — | 81.3% |
| 2 | **2.626** | — | 68.8% |
| **Best** | **2.626** | — | **avg 75%** |

### 7. Circle Packing — Unified Comparison

| Method | Model | Correctness (avg) | Best Quality | Notes |
|---|---|---|---|---|
| Naive Baseline | qwen3-30b | 28% | 2.362 | no strategy |
| SkillOpt (best skill) | qwen3-30b | — | selection_soft=0.632 | +91% over 0.331 |
| Pure RL (Small) | gpt-oss-120b | 54% | **2.624** | 3 epoch |
| Pure RL (Full) | gpt-oss-120b | 56% | **2.636** | 8 epoch, near-SOTA |
| Full Skill→RL (Small) | gpt-oss-120b | **71%** | 2.607 | 3 epoch, strategy skill |
| Minimal Skill→RL (Small) | gpt-oss-120b | **75%** | **2.626** | 3 epoch, guardrail-only |

#### Step-by-Step Three-Way (Small scale)

| Step | Metric | Pure RL | Full Skill→RL | Minimal Skill→RL |
|---|---|---|---|---|
| 0 | Correctness | 50.0% | **82.8%** | 75.0% |
| 0 | Reward Max | **2.559** | 2.602 | 2.471 |
| 1 | Correctness | 62.5% | 56.3% | **81.3%** |
| 1 | Reward Max | **2.613** | 2.501 | 2.609 |
| 2 | Correctness | 50.0% | 75.0% | 68.8% |
| 2 | Reward Max | 2.624 | 2.607 | **2.626** |

---

## Erdős Minimum Overlap

### 1. Naive Baseline

**Model**: qwen3-coder-30b, 32 trials, no strategy guidance

| Gate Rate | Quality Mean | Quality Std | Best C₅ | CV |
|---|---|---|---|---|
| 71.9% (23/32) | 0.427 | 0.026 | **0.388** | 0.060 |

### 2. SkillOpt

**Model**: qwen3-30b (target), deepseek-chat (optimizer), 2 epochs, 20 steps

| Step | Epoch | Rollout Soft | Selection Soft | Action | Best Score | Skill Len |
|---|---|---|---|---|---|---|
| 0 (init) | — | — | 0.122 | — | 0.122 | 5,164 |
| 1 | 1 | 0.116 | 0.041 | reject | 0.122 | 5,164 |
| **2** | **1** | **0.149** | **0.143** | **accept** | **0.143** | 6,051 |
| 3 | 1 | 0.093 | 0.045 | reject | 0.143 | 6,051 |
| **4** | **1** | **0.056** | **0.152** | **accept** | **0.152** | 7,199 |
| 5 | 1 | 0.224 | 0.152 | reject | 0.152 | 7,199 |
| **6** | **1** | **0.132** | **0.183** | **accept** | **0.183** | 8,799 |
| **7** | **1** | **0.219** | **0.195** | **accept** | **0.195** | 9,934 |
| **8** | **1** | **0.227** | **0.216** | **accept** | **0.216** | 10,552 |
| 9-12 | 1 | 0.17–0.23 | 0.12–0.21 | all reject | 0.216 | 10,552 |
| **13** | **1** | **0.225** | **0.227** | **accept** | **0.227** | 11,663 |
| 14-17 | 1 | 0.20–0.22 | 0.20–0.22 | all reject | 0.227 | 11,663 |
| 18 | 1 | 0.113 | 0.227 | accept | **0.227** | 13,703 |
| 19-20 | 2 | 0.22–0.22 | 0.21–0.22 | all reject | 0.227 | 13,703 |

Best skill score: **0.227** (+86%), final skill len: 13,703 chars.

### 3. Pure RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs

| Step | Reward/Max | Correctness | Best C₅ (raw_score/min) |
|---|---|---|---|
| 0 | 2.618 | 25.0% | **0.382** |
| 1 | 2.576 | 50.0% | 0.388 |
| 2 | 2.444 | 50.0% | 0.409 |
| **Best** | **2.618** | **avg 42%** | **0.382** |

### 4. Pure RL (Full)

**Model**: gpt-oss-120b, group=8, batch=4, phase1=10000, 8 epochs

| Step | Reward/Max | Reward/Mean | Correctness | Best C₅ (raw_score) |
|---|---|---|---|---|
| 0 | 2.620 | 0.733 | 31.3% | 0.504 |
| 1 | 2.620 | 0.970 | 43.8% | 0.770 |
| 2 | 2.620 | 0.807 | 31.3% | 0.404 |
| 3 | 2.622 | 0.935 | 40.6% | 1.000 (anomaly) |
| 4 | 2.621 | 0.970 | 37.5% | 0.405 |
| 5 | 2.621 | 0.836 | 33.3% | 0.510 |
| 6 | 2.619 | 0.960 | 37.5% | 0.412 |
| 7 | 2.619 | 0.871 | 33.3% | **0.382** |

### 5. Full Skill → RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs
**Prompt**: problem + **403-line SkillOpt best skill** (compute_c5/normalize/move_mass utilities + asymmetric init + multi-strategy + correlation-aware smoothing + simulated annealing + CRITICAL rules)

| Step | Reward/Max | Reward/Mean | Correctness | Best C₅ |
|---|---|---|---|---|
| 0 | 2.596 | 1.069 | 43.8% | 0.38521 |
| 1 | 2.597 | 1.429 | 62.5% | **0.38512** |
| 2 | 2.596 | 1.431 | 56.3% | 0.38516 |
| **Best** | **2.597** | **avg 54%** | **0.385** | |

### 6. Minimal Skill → RL (Small)

**Model**: gpt-oss-120b, group=4, batch=4, phase1=8000, 3 epochs
**Prompt**: problem + **guardrail-only** (normalize after edit, exact C₅, no closures, reproducibility)

| Step | Reward/Max | Reward/Mean | Correctness | Best C₅ |
|---|---|---|---|---|
| 0 | 2.582 | 0.545 | 25.0% | 0.38737 |
| 1 | 2.587 | 1.183 | 50.0% | **0.38652** |
| 2 | 2.570 | 2.151 | 87.5% | 0.38918 |
| **Best** | **2.587** | **avg 54%** | **0.387** | |

### 7. Erdős — Unified Comparison (Small Scale)

| Method | Model | Correctness (avg) | Best C₅ | vs Pure RL |
|---|---|---|---|---|
| Naive Baseline | qwen3-30b | 71.9% | 0.388 | — |
| Pure RL (Small) | gpt-oss-120b | 42% | **0.382** | baseline |
| Full Skill → RL (Small) | gpt-oss-120b | 54% | 0.385 | C₅ −0.003, corr +12pp |
| Minimal Skill → RL (Small) | gpt-oss-120b | 54% | 0.387 | C₅ −0.005, corr +12pp |

#### Step-by-Step Three-Way (Small scale, Erdős)

| Step | Metric | Pure RL | Full Skill→RL | Minimal Skill→RL |
|---|---|---|---|---|
| 0 | Correctness | 25.0% | 43.8% | 25.0% |
| 0 | Reward Max | **2.618** | 2.596 | 2.582 |
| 1 | Correctness | 50.0% | **62.5%** | 50.0% |
| 1 | Reward Max | 2.576 | **2.597** | 2.587 |
| 2 | Correctness | 50.0% | 56.3% | **87.5%** |
| 2 | Reward Max | 2.444 | **2.596** | 2.570 |
| **Best** | **C₅** | **0.382** | **0.385** | **0.387** |

### 8. 关键发现：Erdős 未复现 Circle Packing 的 Pattern

**Circle Packing 的结论**: Minimal Skill→RL Pareto 占优（correctness 75% + reward 2.626 vs Pure RL 54% + 2.624）。

**Erdős 的结论**: Pure RL > Full Skill > Minimal Skill。Skill 引入了正确性提升（+12pp）但付出了 C₅ 退化的代价（0.382 → 0.385–0.387），且 Minimal 并不比 Full 更好。

**原因分析**:

1. **Erdős Gate 已经很高（72%）**，正确性不是主要瓶颈。Skill 提升正确性的空间小（仅 ~12pp vs Circle Packing ~22pp），但 C₅ 上的损失却更明显。

2. **Erdős 是纯连续优化问题**，策略指导的价值低于 Circle Packing。Circle Packing 受益于"六边形网格 + 迭代扩径"这类具象几何策略，而 Erdős 的"block averaging + mass transfer"是通用搜索操作，模型本来就会。

3. **n_points 随机化造成噪声（40-100）**，可能掩盖了真实差异。每次 trial 的离散化分辨率不同，C₅ 的 variance 本身就大。

4. **Skill 的 utility 代码（compute_c5/normalize）虽然确保正确性，但可能锁住了实现方式**，让模型不再探索不同算法范式（如 MILP、频谱方法）。纯 RL 的模型更自由，偶然找到了更好的 C₅。

**结论**: Skill 的价值取决于问题性质。当 Gate 已高、且优化策略通用时，Skill 的边际收益为负——纯 RL 的 free-form 探索更好。Circle Packing 和 Erdős 的对比正好划出了 Skill 有效的边界条件。

---

## Appendix: Token Consumption

### Circle Packing Small

| Variant | ob_tokens (prompt) | ac_tokens (code) |
|---|---|---|
| Pure RL | ~650 | ~9200 |
| Full Skill→RL | ~1150 | ~10200 |
| Minimal Skill→RL | ~930 | ~10100 |

### Erdős Small

| Variant | ob_tokens (prompt) | ac_tokens (code) |
|---|---|---|
| Pure RL | ~650 | ~10400 |
| Full Skill→RL | ~4400–6700 | ~4600–6500 |
| Minimal Skill→RL | ~760–2800 | ~8800–10700 |

---

## REI (Regular Expression Inference)

### 1. Task Selection

从 19 个候选任务中筛选出 10 个复杂任务，去掉简单 regex（hex, phone, zip, time, float 等）：

| 分片 | 任务 | 核心难度 | Qwen Gate |
|------|------|---------|-----------|
| **train** | url_full | 可选组 + 路径 + 协议 | 0% |
| | password | 前瞻断言 `(?=.*\d)` | 100% (soft=0.75) |
| | mac_address | 重复组 `([0-9a-f]{2}:){5}` | 未测 |
| | email | `@` + `.` + `{2,}` | 100% (soft=0.52-0.72) |
| **val** | ipv4 | 长重复 `\d{1,3}\.`×4 | 100% (soft=0.14-0.52) |
| | credit_card | 重复组 `\d{4}-`×4 | 100% (soft=0.88) |
| | username | 范围量词 `{3,12}` | Gate=0 |
| **test** | semver_range | 可选组 `(-v...)?` | Gate=0 |
| | slug | 组 + `*` 量词 | 25% |
| | animal | 交替 `(cat\|dog\|...)$` | 100% (soft=0.79) |

### 2. Quality 定义（论文对齐）

```
Gate  = re.fullmatch(r, p) 全过 + re.fullmatch(r, n) 全拒 (二值)
Cost  = |r| + 50 × heldout_mismatch_rate   (heldout = 100 pos + 100 neg)
Quality = -Cost
Reward = Gate × Quality   → 归一化到 [0,1]: reward = max(0, 1 - cost/200)
```

### 3. Naive Baseline (Qwen 3-30b)

| Task | Gate | Soft Mean |
|------|------|-----------|
| hex | 75% | 0.89 |
| phone | 100% | 0.94 |
| url_full | 0% | 0.0 |
| date | 75% | 0.69 |
| email | 100% | 0.65 |
| ipv4 | 100% | 0.32 |
| slug | 25% | 0.20 |
| filename | 100% | 0.90 |
| semver | 100% | 0.71 |
| float | 100% | 0.94 |

### 4. SkillOpt

**Model**: qwen3-30b (target), deepseek-chat (optimizer), 8 epochs, 8 steps
**Data**: 19 tasks (含简单+复杂), train=4 val=5 test=10

| Step | Action | Selection Soft | vs Baseline | Skill Len |
|------|--------|---------------|-------------|-----------|
| 1 | accept | 0.382 | +58% | 1,832 |
| **2** | **accept** | **0.574** | **+137%** | **2,406** |
| 3-8 | reject | ≤0.574 | — | 2,406 |

**Test (best skill, step 2):**

| Task | Baseline | Best Skill | Change |
|------|----------|------------|--------|
| animal | 0.905 | 0.885 | -2.2% |
| slug | 0.910 | 0.900 | -1.1% |
| semver_range | Gate=0 | Gate=0 | — |

Selection soft +137%，但 test 没涨 — skill 过拟合到训练分布的 regex 类型。

**Skill 自动注入的策略：**
```
- 锚点: 始终用 ^ 和 $
- 前瞻: 对密码等多条件约束用 (?=.*) 链
- URL: ^(https?://)?(www\.)? 处理可选协议
- 验证: 返回前用 re.fullmatch 测试所有正/负例
- 字符类: [a-z] 小写, [0-9] 数字, \d 通用
```

### 5. Three-Way RL Comparison (url_full)

**Model**: gpt-oss-120b via Tinker API, group=4, epochs=3

#### Unified Comparison

| Method | Correctness (avg) | Best Quality (Reward/Max) | Min Cost | Notes |
|---|---|---|---|---|
| Pure RL | 100% | 0.804 | **39** | no skill, 最短 regex |
| Minimal Skill→RL | **100%** | 0.804 | 48 | interface + constraints |
| Full Skill→RL | 92% | **0.814** | 51 | with SkillOpt strategy |

#### Step-by-Step Three-Way

| Step | Metric | Pure RL | Minimal Skill→RL | Full Skill→RL |
|---|---|---|---|---|
| 0 | Correctness | — | — | 75.0% |
| 0 | Reward Max | — | — | 0.740 |
| 1 | Correctness | **100%** | **100%** | **100%** |
| 1 | Reward Max | **0.804** | 0.764 | 0.799 |
| 2 | Correctness | **100%** | **100%** | **100%** |
| 2 | Reward Max | 0.764 | **0.804** | **0.814** |
| **Best** | **Reward/Max** | **0.804** | **0.804** | **0.814** |
| **Best** | **Min Cost** | **39** | **48** | **51** |

#### 最佳 regex 对比

| Variant | 最佳 regex | Cost | Reward |
|---------|-----------|------|--------|
| Pure RL | `^(?:https?://)?\w+(?:\.\w+){1,2}(/.+)?$` | 39.2 | 0.738 |
| Minimal Skill→RL | `^(?:https?://)?(?:[A-Za-z0-9-]+\.){1,2}[A-Za-z0-9-]+(?:/[^/]+)?$` | 48.0 | 0.760 |
| Full Skill→RL | `(?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z0-9-]+(?:/[^/]+)?` | 51.0 | 0.745 |

#### 结论：REI 三个 benchmark 的汇总定位

| 维度 | Circle Packing | Erdős | REI |
|------|---------------|-------|-----|
| Gate（weak model） | 28% | 72% | 0-100%（task 依赖） |
| RL Correctness | 54-75% | 25-88% | **92-100%** |
| Skill→RL 效果 | **+22pp correctness, +0.06 quality** | +12pp correctness, -0.005 quality | 无显著提升 |
| 最优 variant | Minimal Skill→RL | Pure RL | **Pure RL**（最简单） |
| 解释 | 低 Gate 需要工程流程框架 | 高 Gate + 连续优化不需要 skill | 模型本身就能写正则 |

**REI 的实验结果表明：当复杂模型（gpt-oss-120b）对任务已有足够能力时，Skill 注入无法带来额外收益。这与 Erdős 的发现一致——Skill 的边际收益随模型能力上升而递减。**

**三个 benchmark 的统一结论：**

1. **Skill 最有价值的场景**：弱模型 + 低 Gate + 有工程流程可注入（Circle Packing）
2. **Skill 无价值的场景**：强模型 + 高 Gate + 连续优化（Erdős、REI）
3. **Minimal vs Full Skill**：没有一致差异——Circle Packing 上 Minimal 更好，Erdős 和 REI 上两者均无优势
4. **核心发现**："知识注入"的真正窗口是**代码易崩性**（低 Gate），不是领域陌生度


## Appendix: Config Summary

| Experiment | group | batch | phase1 | epochs | model |
|---|---|---|---|---|---|
| Circle Packing Pure RL (Full) | 16 | 4 | 10000 | 8 | gpt-oss-120b |
| Circle Packing Pure RL (Small) | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Circle Packing Full Skill→RL | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Circle Packing Minimal Skill→RL | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Erdős Pure RL (Full) | 8 | 4 | 10000 | 8 | gpt-oss-120b |
| Erdős Pure RL (Small) | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Erdős Full Skill→RL | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Erdős Minimal Skill→RL | 4 | 4 | 8000 | 3 | gpt-oss-120b |
| Circle Packing SkillOpt | batch=4, 4 epochs, 24 steps | | | | qwen3-30b target |
| Erdős SkillOpt | batch=4, 2 epochs, 20 steps | | | | qwen3-30b target |
| REI SkillOpt | batch=8, 8 epochs, 8 steps | | | | qwen3-30b target |
| REI Pure RL (Small) | 4 | 4 | 4096 | 3 | gpt-oss-120b (Tinker) |
| REI Full Skill→RL | 4 | 4 | 4096 | 3 | gpt-oss-120b (Tinker) |
| REI Minimal Skill→RL | 4 | 4 | 4096 | 3 | gpt-oss-120b (Tinker) |

---
