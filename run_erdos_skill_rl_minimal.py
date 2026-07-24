"""Minimal Skill → RL for Erdős Min Overlap: Guardrails only, no strategy prescription.

Compares against pure RL baseline (log.md):
  Pure RL (8-epoch): C₅ = 0.38180, correctness ~37% avg

Usage:
    cd /mnt/c/Users/1/Desktop/TTT-Discover
    source venv/bin/activate
    python run_erdos_skill_rl_minimal.py
"""

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["DISABLE_WANDB"] = "1"

print(f"TINKER_API_KEY: {'SET' if os.environ.get('TINKER_API_KEY') else 'MISSING'}")

from examples.erdos_min_overlap.env import (
    ErdosMinOverlapEnv,
    ErdosMinOverlapRewardEvaluator,
    verify_c5_solution,
)
from ttt_discover import DiscoverConfig, discover, State, Environment


class ErdosMinOverlapEnvMinimal(Environment):
    """Erdős with minimal skill: only correctness guardrails."""

    reward_function = ErdosMinOverlapRewardEvaluator
    state_type = State
    max_construction_len = 1000

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return ErdosMinOverlapEnv.create_initial_state(problem_type)

    def is_maximize(self) -> bool:
        return False

    def get_question(self) -> str:
        state = self.initial_state
        state_ctx = state.to_prompt(0.3808, metric_name="C₅ bound", maximize=False)

        construction_section = ""
        if hasattr(state, "construction") and state.construction is not None and len(state.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(state.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        if state.code and state.code.strip():
            code_section = """Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc.
Unless you make a meaningful improvement, you will not be rewarded."""
        else:
            code_section = """Write code to optimize this construction."""

        return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: 1000s for your code to run
- **CPUs**: 2 available

## Correctness Guardrails (violations cause zero reward)

- After EVERY modification to h, normalize to maintain sum(h) = n_points/2 and clip to [0,1].
  Normalization skips → h out of [0,1] → instant failure. This is the #1 failure mode.
- Report the EXACT C₅ from np.correlate. Do NOT round, adjust, or fabricate. C₅ mismatch = failure.
- All helper functions at top level (no nested closures). NEVER use global/nonlocal.
- Use `np.random.RandomState(seed)` for reproducibility.
- validate internally before returning: check h in [0,1] and sum(h) ≈ n_points/2.

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy, math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- evaluate_erdos_solution() and initial_h_values (if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found
- Minimize C₅. Current record: C₅ ≤ 0.38092. Goal: C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
"""


config = DiscoverConfig(
    env_type=ErdosMinOverlapEnvMinimal,
    problem_type="",
    num_cpus_per_task=1,
    eval_timeout=600,
    experiment_name="erdos-minimal-skill-rl",
    wandb_project="",
    # Small scale for fast comparison (matches Circle Packing Small config)
    num_epochs=3,
    group_size=4,
    groups_per_batch=4,
    phase1_max_tokens=8000,
)

discover(config)
