from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover

_OPTIMAL = {(12, 7): 792, (15, 10): 3003, (21, 15): 43596, (24, 17): 237984}


class AdmissibleSetRewardEvaluator(SandboxRewardEvaluator):

    def __init__(self, dimension=15, weight=10):
        super().__init__()
        self.dimension = dimension
        self.weight = weight

    def get_program_entrypoint(self) -> str:
        return "run_admissible_set"

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)

        try:
            achieved = int(output)
        except (ValueError, TypeError) as e:
            return self._get_failure_entry(f"Bad return format: {e}")

        optimal = _OPTIMAL.get((self.dimension, self.weight), 0)
        if optimal > 0:
            normalized = achieved / optimal
        else:
            normalized = achieved / max(achieved, 1)

        return {
            "reward": normalized,
            "correctness": 1.0 if normalized >= 1.0 else 0.0,
            "raw_score": achieved,
            "msg": f"Achieved: {achieved}, Optimal: {optimal}, Ratio: {normalized:.4f}",
            "result_construction": [],
            "stdout": getattr(self, '_last_stdout', ''),
        }


class AdmissibleSetEnv(Environment):
    reward_function = AdmissibleSetRewardEvaluator
    state_type = State

    dimension = 12
    weight = 7

    def is_maximize(self) -> bool:
        return True

    def get_question(self) -> str:
        state_ctx = self.initial_state.to_prompt(
            3000.0, metric_name="set size", maximize=True
        )
        optimal = _OPTIMAL.get((self.dimension, self.weight), "unknown")

        return f"""You are an expert in combinatorial optimization.

Your task is to write a complete Python program that constructs a maximum-cardinality symmetric constant-weight admissible set I({self.dimension}, {self.weight}).

## Problem

A symmetric constant-weight admissible set I(n, w) is a collection of vectors in {{0,1,2}}ⁿ such that:
- Each vector has exactly w non-zero entries (weight w)
- No vector is coordinate-wise dominated by another (in the weight sense: 0→0, 1→1, 2→2 via mapping [0,1,1,2,2,3,3])
- No triple of vectors forms a "bad triple" pattern

## Required Interface

Define a function:
```python
def run_admissible_set(dimension: int, weight: int) -> int:
    \"\"\"Returns the size of the admissible set constructed.\"\"\"
```

Your function must:
1. Generate all valid candidate vectors in {{0,1,2}}^dimension with exactly weight non-zero entries
2. Use a greedy algorithm: iteratively pick the best candidate, remove dominated/invalid ones
3. Return the achieved set size

Known optimal size: {optimal}
Reward = achieved / optimal (higher = better).

## Rules
- Use numpy, itertools, math as needed
- No filesystem or network IO
- Must be deterministic

{state_ctx}
"""


def discover_admissible_set():
    config = DiscoverConfig(
        env_type=AdmissibleSetEnv,
        problem_type="",
        num_cpus_per_task=1,
        eval_timeout=60,
        experiment_name="admissible-set-rl",
        wandb_project="",
        num_epochs=3,
        group_size=4,
        groups_per_batch=4,
        phase1_max_tokens=8000,
    )
    discover(config)


if __name__ == "__main__":
    discover_admissible_set()
