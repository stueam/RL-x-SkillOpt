"""Naive baseline for REI: deepseek-v4-flash.
Loads all tasks from generated data files and runs 4 trials each."""
import sys, os, time, json
from dotenv import load_dotenv
load_dotenv()

from skillopt.model import chat_target
from skillopt.model.azure_openai import configure_azure_openai, set_target_deployment
from benchmarks.rei.skillopt_env.evaluator import evaluate_rollout

dpsk_key = os.environ.get("DEEPSEEK_API_KEY", "")
configure_azure_openai(
    target_endpoint="https://api.deepseek.com",
    target_auth_mode="openai_compatible",
    target_api_key=dpsk_key,
)
set_target_deployment("deepseek-v4-flash")

# Load all tasks from data files
TASKS = []
for split in ["train", "val", "test"]:
    path = f"benchmarks/rei/data/{split}/items.json"
    if os.path.exists(path):
        TASKS.extend(json.load(open(path)))

print(f"Loaded {len(TASKS)} tasks from data files\n")

SYSTEM = """You are an expert in regular expression synthesis.
Write `solve(positives, negatives)` that returns a regex pattern.

```python
def solve(positives, negatives):
    # Return a regex pattern string
```

The pattern must compile, match ALL positives, reject ALL negatives.
Output code between ```python and ```.
"""

LOG_FILE = "baseline_rei_log.jsonl"
total_gate = 0
total_trials = 0

for task in TASKS:
    for t in range(4):
        print(f"--- Trial {t+1}/4 | {task['id']} ---", flush=True)
        user = (
            f"## Positives\n{json.dumps(task['positives'], indent=2)}\n\n"
            f"## Negatives\n{json.dumps(task['negatives'], indent=2)}\n\n"
            f"Write solve(positives, negatives). Code between ```python and ```."
        )
        t0 = time.time()
        resp, _ = chat_target(system=SYSTEM, user=user, max_completion_tokens=4096, retries=3, stage="baseline")
        api = time.time() - t0
        t1 = time.time()
        res = evaluate_rollout(
            resp,
            positives=task["positives"],
            negatives=task["negatives"],
            heldout_positives=task["heldout_positives"],
            heldout_negatives=task["heldout_negatives"],
            eval_timeout=30,
        )
        ev = time.time() - t1
        total_gate += res["hard"]
        total_trials += 1
        fail = f" | fail={res['fail_reason']}" if res['fail_reason'] else ""
        print(f"  API={api:.1f}s | eval={ev:.1f}s | Gate={res['hard']} | soft={res['soft']:.3f}{fail}")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "task": task["id"], "trial": t + 1,
                "gate": res["hard"], "soft": res["soft"],
                "fail_reason": res["fail_reason"],
                "api_time_s": round(api, 1), "eval_time_s": round(ev, 1),
            }, ensure_ascii=False) + "\n")

print(f"\n=== Summary: Gate={total_gate}/{total_trials} ({100*total_gate/total_trials:.1f}%) ===")
print(f"Log saved to {LOG_FILE}")
