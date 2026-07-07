import inspect
import json
from typing import Any, Dict, List

from datasets import load_dataset

from ttt_discover import Environment, SandboxRewardEvaluator, State, DiscoverConfig, discover


LIVE_MATH_DATASET = "LiveMathematicianBench/LiveMathematicianBench"


def load_live_math_data(split: str = "train") -> List[Dict[str, Any]]:
    """Load LiveMathematicianBench dataset from HuggingFace."""
    ds = load_dataset(LIVE_MATH_DATASET, split=split)
    return [item for item in ds]


def format_mcq_question(mcq: Dict[str, Any]) -> str:
    """Format MCQ question with choices for the prompt."""
    question = mcq["question"]
    choices = mcq["choices"]
    choice_text = "\n".join([f"{c['label']}. {c['text']}" for c in choices])
    return f"{question}\n\nChoices:\n{choice_text}"


def extract_answer_letter(response: str) -> str:
    """Extract answer choice letter (A-E) from model response."""
    import re
    # Look for patterns like "Answer: A" or "Answer: A." or just "A" at end
    patterns = [
        r"(?i)answer\s*[:\-]\s*([A-E])",
        r"(?i)final answer\s*[:\-]\s*([A-E])",
        r"(?i)the answer is\s*([A-E])",
        r"\b([A-E])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


class LiveMathRewardEvaluator(SandboxRewardEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_data = None
        self.correct_answer = None

    def get_program_entrypoint(self) -> str:
        return "solve"

    def get_reward(self, code: str, state: State) -> Dict[str, Any]:
        output, error_msg = self.execute_code(code, state)
        if error_msg:
            return self._get_failure_entry(error_msg)

        # Extract answer from output
        answer_letter = extract_answer_letter(str(output))
        if not answer_letter:
            return self._get_failure_entry("No valid answer letter found in output")

        is_correct = (answer_letter == self.correct_answer)
        reward = 1.0 if is_correct else 0.0

        return {
            "reward": reward,
            "correctness": 1.0 if is_correct else 0.0,
            "raw_score": reward,
            "msg": f"Predicted: {answer_letter}, Correct: {self.correct_answer}, {'Correct' if is_correct else 'Incorrect'}",
            "result_construction": [],
            "stdout": getattr(self, '_last_stdout', ''),
        }


class LiveMathEnv(Environment):
    reward_function = LiveMathRewardEvaluator
    state_type = State

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load problem data for this instance
        self._load_problem()

    def _load_problem(self):
        """Load the specific problem for this environment instance."""
        data_list = load_live_math_data("train")
        # Use problem_type as index or hash to select problem
        if self.problem_type.isdigit():
            idx = int(self.problem_type) % len(data_list)
        else:
            # Hash the problem_type to get consistent index
            idx = hash(self.problem_type) % len(data_list)
        self.problem_data = data_list[idx]
        mcq = self.problem_data["mcq"]
        self.correct_answer = mcq["correct_choice"]["label"]
        # Set correct answer on reward evaluator
        self.reward_function.correct_answer = self.correct_answer
        self.reward_function.problem_data = self.problem_data

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        # Load problem to get context for initial state
        data_list = load_live_math_data("train")
        if problem_type.isdigit():
            idx = int(problem_type) % len(data_list)
        else:
            idx = hash(problem_type) % len(data_list)
        problem = data_list[idx]
        mcq = problem["mcq"]
        theorem = problem.get("theorem", "")
        sketch = problem.get("sketch", "")
        theorem_type = problem.get("theorem_type", [])

        # Build initial context
        context = f"""Theorem: {theorem[:2000]}...

Proof Sketch: {sketch[:1000]}...

Theorem Type: {', '.join(theorem_type)}

Question: {mcq['question']}

Choices:
"""
        for choice in mcq["choices"]:
            context += f"  {choice['label']}. {choice['text']}\n"

        return State(
            timestep=-1,
            code="",
            value=0.0,
            construction=[context],
        )

    def is_maximize(self) -> bool:
        return True

    def get_question(self) -> str:
        state = self.initial_state
        problem = self.problem_data
        mcq = problem["mcq"]
        theorem = problem.get("theorem", "")
        sketch = problem.get("sketch", "")
        theorem_type = problem.get("theorem_type", [])

        question_text = format_mcq_question(mcq)

        state_ctx = state.to_prompt(1.0, metric_name="accuracy", maximize=True)

        return f"""You are an expert mathematician specializing in advanced mathematics including algebraic geometry, differential geometry, topology, analysis, and mathematical physics.

Your task is to answer a multiple-choice question grounded in a recent research theorem. You will be given:
1. The theorem statement
2. A proof sketch / key ideas
3. The multiple-choice question testing understanding of the theorem

## Theorem
{theorem[:3000]}

## Proof Sketch
{sketch[:1500]}

## Theorem Type
{', '.join(theorem_type)}

## Question
{question_text}

{state_ctx}

Rules:
- You must define a `solve()` function that returns the answer choice letter (A, B, C, D, or E) as a string.
- You can use Python to reason, compute, or verify your thinking.
- Print your reasoning and final answer.
- The function should return just the letter, e.g., "A"

Think step by step. First provide your reasoning between <reasoning> and </reasoning> tags, then return the final program between ```python and ```.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""

    def _get_code_languages(self) -> list[str]:
        return ["python"]


def discover_livemath(problem_type: str = "0"):
    config = DiscoverConfig(
        env_type=LiveMathEnv,
        problem_type=problem_type,
        num_cpus_per_task=1,
        eval_timeout=300,
        experiment_name=f"test-livemath-{problem_type}-run",
        wandb_project="",
        num_epochs=10,
        group_size=8,
        groups_per_batch=2,
        phase1_max_tokens=8000,
        learning_rate=4e-5,
        temperature=1.0,
        kl_penalty_coef=0.1,
    )
    discover(config)


if __name__ == "__main__":
    discover_livemath("0")