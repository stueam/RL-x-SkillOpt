# TTT-Discover + SkillOpt: Code Generation Agents Research

Investigating the division of labor between **weight optimization (RL via TTT-Discover)** and **prompt optimization (SkillOpt)** for LLM code generation.

**Core finding**: SkillOpt fixes correctness (engineering process); RL pushes solution quality (optimization search). They address different bottlenecks and **do not compete**.

---

## Project Structure

```
root/
├── ttt_discover/               # RL framework (TTT-Discover, weights optimization)
├── skillopt/                    # Skill optimization framework (prompt optimization)
│   ├── engine/                  # Training loop
│   ├── model/                   # LLM backends (OpenAI, DeepSeek, Qwen)
│   ├── gradient/                # Analyst reflection & merge
│   ├── prompts/                 # Generic prompt templates
│   ├── datasets/                # Data loading
│   └── envs/                    # Built-in env base classes
├── benchmarks/                  # One directory per problem
│   ├── cap_set/                 # Cap set in F_3^n (FunSearch-style)
│   │   ├── problem.py           # Shared domain logic
│   │   ├── rl_env.py            # TTT-Discover RL environment
│   │   ├── skillopt_env/        # SkillOpt environment
│   │   │   ├── evaluator.py     # Gate × Quality evaluator
│   │   │   ├── adapter.py       # SkillOpt adapter
│   │   │   ├── rollout.py       # Rollout logic
│   │   │   ├── dataloader.py    # Data loader
│   │   │   ├── prompts/         # Analyst prompts
│   │   │   └── skills/          # Skill documents (initial.md, best.md)
│   │   ├── config/              # YAML configs
│   │   └── data/                # Train/val/test splits
│   ├── circle_packing/          # Circle packing (26 circles in unit square)
│   ├── erdos_min_overlap/       # Erdős minimum overlap problem
│   └── rei/                     # Regular Expression Inference
│       ├── problem.py           # validate_regex, score_generalization
│       ├── rl_env.py            # RL environment (3 skill variants: none/minimal/best)
│       ├── skillopt_env/
│       ├── config/
│       └── data/                # 10 complex tasks, 100 heldout each
├── configs/                     # Shared config files
│   └── _base_/                  # Base config defaults
├── scripts/                     # Training entry points
├── outputs/                     # Experiment outputs (SkillOpt)
├── tinker_log/                  # RL training logs (TTT-Discover)
├── SkillOpt-main/               # Legacy SkillOpt code (backup)
├── run_*.py                     # Experiment entry points (see below)
├── final_report.md              # Comprehensive experiment report
├── .env                         # API keys
└── requirements/
```

## Running Experiments

### RL (TTT-Discover — requires WSL)

```bash
cd /mnt/c/Users/1/Desktop/TTT-Discover
source venv/bin/activate

# Circle Packing
python run_circle_packing.py

# Erdős
python run_erdos.py
python run_erdos_rl_small.py

# REI (three variants)
python run_rei.py                      # Pure RL
python run_rei_skill_rl_minimal.py     # Minimal Skill → RL
python run_rei_skill_rl.py             # Full Skill → RL

# Cap Set
python run_cap_set.py
```

### SkillOpt Training

```bash
python run_cap_set_skillopt.py
python run_circle_packing_skillopt.py
python run_erdos_skillopt.py
python run_rei_skillopt.py
```

### Naive Baseline

```bash
python run_cap_set_baseline.py
python run_rei_baseline.py
```

## Benchmark Comparison

| Benchmark | Task | Gate (weak) | RL Correctness | Skill→RL Effect | Winner |
|---|---|---|---|---|---|
| **Circle Packing** | Pack 26 circles in unit square | **28%** | 54-75% | **+22pp corr, +0.06 quality** | **Minimal Skill→RL** |
| **Erdős** | Minimize overlap integral C5 | **72%** | 25-88% | +12pp corr, -0.005 quality | Pure RL |
| **REI** | Synthesize regex from examples | 0-100% | **100%** | No effect | Pure RL |
| **Cap Set** | Max subset F_3ⁿ no 3-term AP | 100% | — | SkillOpt baseline only | N/A (too easy) |

### Key Insight

**Skill value = Task requires complex engineering scaffolding / Model can generate valid code unaided**

| Condition | Circle Packing | Erdős | REI |
|---|---|---|---|
| Engineering complexity | High (validate, repair, scaffold) | Medium (normalize) | Low (just compile + match) |
| Weak model Gate | 28% (72% crash) | 72% | 0-100% |
| Skill effect | **+22pp** | +12pp, -quality | **0** |

## Three-Way RL Comparison

For each benchmark, we compare:
1. **Pure RL**: No skill injection, model free-form
2. **Minimal Skill→RL**: Interface + constraints only (no strategy)
3. **Full Skill→RL**: SkillOpt-optimized strategy injected

| Metric | Circle Packing | Erdős | REI |
|---|---|---|---|
| Best variant | Minimal Skill→RL | Pure RL | Pure RL |
| Correctness | 75% Minimal vs 54% Pure | 54% Minimal vs 42% Pure | 100% all variants |
| Quality | 2.626 Minimal vs 2.624 Pure | 0.385 Full vs **0.382** Pure | 0.814 Full vs 0.804 Pure |

## API Keys (`.env`)

```
AZURE_OPENAI_API_KEY=sk-or-v1-...     # OpenRouter key
AZURE_OPENAI_ENDPOINT=https://openrouter.ai/api/v1
AZURE_OPENAI_AUTH_MODE=openai_compatible
DEEPSEEK_API_KEY=sk-f5566...          # DeepSeek key
TINKER_API_KEY=tml-...                # Tinker API key (for RL)
```

For DeepSeek target (override in code):
```python
configure_azure_openai(target_endpoint="https://api.deepseek.com", target_auth_mode="openai_compatible", target_api_key=DEEPSEEK_API_KEY)
set_target_deployment("deepseek-v4-flash")
```

## Dependencies

Install WSL + Ubuntu 22.04, then in WSL:

```bash
cd /mnt/c/Users/1/Desktop/TTT-Discover
python -m venv venv
source venv/bin/activate
pip install -r requirements/requirements.txt
```

## Experiment Logs

- **RL logs**: `tinker_log/<experiment_name>/`
- **SkillOpt logs**: `SkillOpt-main/outputs/<experiment_name>/`
- **Consolidated report**: `final_report.md`

## Key Files for Each Benchmark

| File | Purpose |
|------|---------|
| `benchmarks/<name>/problem.py` | Shared domain logic |
| `benchmarks/<name>/rl_env.py` | RL environment (TTT-Discover) |
| `benchmarks/<name>/skillopt_env/evaluator.py` | Gate × Quality evaluator |
| `benchmarks/<name>/skillopt_env/adapter.py` | SkillOpt adapter |
| `benchmarks/<name>/skillopt_env/rollout.py` | Rollout execution |
| `benchmarks/<name>/skillopt_env/prompts/` | Analyst prompts (error/success) |
| `benchmarks/<name>/skillopt_env/skills/` | Skill documents |
| `benchmarks/<name>/config/default.yaml` | SkillOpt config |
| `benchmarks/<name>/data/` | Train/val/test splits |
| `run_<name>.py` | RL entry point |
| `run_<name>_skillopt.py` | SkillOpt entry point |
| `run_<name>_baseline.py` | Baseline evaluation |
