from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover

_OPTIMAL = {1: 2, 2: 4, 3: 9, 4: 20, 5: 45, 6: 112, 7: 236, 8: 496}


class CapSetRewardEvaluator(SandboxRewardEvaluator):

    def __init__(self, n=5):
        super().__init__()
        self.n = n

    def get_program_entrypoint(self) -> str:
        return "construct"

    def get_reward(self, code: str, state: State) -> float:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)
        try:
            m = int(output)
        except (ValueError, TypeError) as e:
            return self._get_failure_entry(f"Bad return format: {e}")

        optimal = _OPTIMAL.get(self.n, 0)
        normalized = m / optimal if optimal > 0 else m / max(m, 1)
        return {
            "reward": normalized,
            "correctness": 1.0 if normalized >= 1.0 else 0.0,
            "raw_score": m,
            "msg": f"Size: {m}, Optimal: {optimal}, Ratio: {normalized:.4f}",
            "result_construction": [],
            "stdout": getattr(self, '_last_stdout', ''),
        }


class CapSetEnv(Environment):
    reward_function = CapSetRewardEvaluator
    state_type = State

    n = 5

    def is_maximize(self) -> bool:
        return True

    def get_question(self) -> str:
        state_ctx = self.initial_state.to_prompt(
            3000.0, metric_name="cap set size", maximize=True
        )
        optimal = _OPTIMAL.get(self.n, "unknown")

        return f"""You are an expert in extremal combinatorics.

Construct a large cap set in F_3^{self.n}.

## Definition
A cap set in F_3^n is a subset of {{0,1,2}}^n with no three distinct elements summing to 0 (mod 3) coordinatewise.
For any pair (a,b), the vector z = (-a-b) mod 3 must not be in the set (unless z=a or z=b).

## Required Interface
```python
def construct() -> int:
    # Build the largest cap set you can for n={self.n}
    # Return the SIZE (m) as an int
```

## Strategy
- Generate all 3^{self.n} = {3**self.n} candidate vectors
- Use np.random.RandomState(seed) to shuffle candidates differently
- Greedily select vectors that don't violate the cap property
- Try multiple seeds and return the best size found

Known optimal: {optimal}
Reward = size / optimal (higher = better).

## Rules
- Use numpy and itertools
- Return only the integer size
- Code between ```python and ```

{state_ctx}
"""


def discover_cap_set():
    config = DiscoverConfig(
        env_type=CapSetEnv,
        problem_type="",
        num_cpus_per_task=1,
        eval_timeout=60,
        experiment_name="cap-set-rl",
        wandb_project="",
        num_epochs=3,
        group_size=4,
        groups_per_batch=4,
        phase1_max_tokens=8000,
    )
    discover(config)


if __name__ == "__main__":
    discover_cap_set()
