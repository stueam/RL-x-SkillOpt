"""TTT-Discover RL environment for REI (aligned with CirclePacking pattern)."""
import json, os, re

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover

# Load instances from data files
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_INSTANCES: dict[str, dict] = {}
for split in ["train", "val", "test"]:
    path = os.path.join(_DATA_DIR, split, "items.json")
    if os.path.exists(path):
        for item in json.load(open(path)):
            _INSTANCES[item["id"]] = item


class ReiRewardEvaluator(SandboxRewardEvaluator):
    """Minimal evaluator — matches CirclePacking pattern (no custom __init__)."""

    def get_program_entrypoint(self) -> str:
        return "run_rei"

    def get_reward(self, code: str, state: State) -> dict:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)

        try:
            pattern = str(output).strip()
        except (ValueError, TypeError) as e:
            return self._get_failure_entry(f"Bad return: {e}")

        if not pattern:
            return self._get_failure_entry("empty_pattern")

        # Look up instance data by problem_type (set by framework from DiscoverConfig.problem_type)
        instance_id = str(self.problem_type)
        data = _INSTANCES.get(instance_id)
        if data is None:
            return self._get_failure_entry(f"unknown_instance: {instance_id}")

        # Gate: valid regex
        try:
            re.compile(pattern)
        except re.error:
            return self._get_failure_entry("invalid_regex")

        # Gate: match all positives
        for p in data["positives"]:
            if not re.fullmatch(pattern, p):
                return self._get_failure_entry("missed_positive")

        # Gate: reject all negatives
        for n in data["negatives"]:
            if re.fullmatch(pattern, n):
                return self._get_failure_entry("accepted_negative")

        # Quality = -cost (paper: cost = |r| + 50 * heldout_mismatch_rate)
        mismatches = 0
        total = 0
        for p in data.get("heldout_positives", []):
            if not re.fullmatch(pattern, p):
                mismatches += 1
            total += 1
        for n in data.get("heldout_negatives", []):
            if re.fullmatch(pattern, n):
                mismatches += 1
            total += 1
        heldout_mismatch_rate = mismatches / total if total > 0 else 0.0
        cost = len(pattern) + 50.0 * heldout_mismatch_rate
        normalized = max(0.0, 1.0 - cost / 200.0)

        return {
            "reward": normalized,
            "correctness": 1.0,
            "raw_score": int(cost),
            "msg": f"Pattern: {pattern} (cost={cost:.1f})",
            "result_construction": [],
            "stdout": getattr(self, '_last_stdout', ''),
        }


class ReiEnv(Environment):
    reward_function = ReiRewardEvaluator
    state_type = State
    skill_variant = "best"  # "none" | "minimal" | "best"

    def is_maximize(self) -> bool:
        return True

    def get_question(self) -> str:
        data = _INSTANCES.get(str(self.problem_type))
        if data is None:
            return "Error: unknown instance"

        pos = json.dumps(data["positives"], indent=2)
        neg = json.dumps(data["negatives"], indent=2)
        state_ctx = self.initial_state.to_prompt(1.0, metric_name="regex quality")

        # Load skill from SkillOpt training (Skill → RL injection)
        skill_map = {"none": None, "minimal": "minimal.md", "best": "best.md"}
        skill_file = skill_map.get(self.skill_variant, None)
        skill_section = ""
        if skill_file:
            skill_path = os.path.join(os.path.dirname(__file__), "skillopt_env", "skills", skill_file)
            if os.path.exists(skill_path):
                skill_section = open(skill_path).read()

        strategy_section = ""
        if skill_section:
            strategy_section = f"\n## Strategy\n{skill_section}\n"

        return f"""You are an expert in regular expression synthesis.

Infer a regex pattern from example strings.

## Definition
- Positives: strings that MUST match the regex
- Negatives: strings that MUST NOT match

## Required Interface
```python
def run_rei() -> str:
    # Return a regex pattern string
```

## Validation
1. Pattern must compile as valid Python regex
2. Must match ALL positives via re.fullmatch
3. Must reject ALL negatives
{strategy_section}
## Positive examples
{pos}

## Negative examples
{neg}

Shorter, more accurate patterns score higher. Code between ```python and ```.

{state_ctx}
"""


def discover_rei(instance_id: str = "hex", skill_variant: str = "none"):
    ReiEnv.skill_variant = skill_variant
    variant_tag = "" if skill_variant == "none" else "-" + skill_variant
    config = DiscoverConfig(
        env_type=ReiEnv,
        problem_type=instance_id,
        num_cpus_per_task=1,
        eval_timeout=15,
        experiment_name="rei-rl-" + instance_id + variant_tag,
        wandb_project="",
        num_epochs=3,
        group_size=4,
        groups_per_batch=4,
        phase1_max_tokens=4096,
    )
    discover(config)


if __name__ == "__main__":
    discover_rei()
