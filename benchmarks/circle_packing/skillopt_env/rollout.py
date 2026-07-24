"""Circle packing rollout -- code generation + execution + scoring.

Given a skill document and a batch of items (each describing an N=26
circle-packing instance with a different random seed), the target LLM
generates Python code that solves the packing problem.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .evaluator import evaluate_rollout
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="circlepacking").format(skill_section=skill_section)


def _build_user(item: dict) -> str:
    num_circles = item.get("num_circles", 26)
    seed = item.get("seed", 0)
    target = item.get("target", 2.636)
    return (
        f"## Problem\n"
        f"Pack {num_circles} circles in a unit square [0,1]×[0,1] "
        f"to maximize the sum of radii.\n\n"
        f"## Parameters\n"
        f"- Number of circles: {num_circles}\n"
        f"- Random seed for initialization: {seed}\n"
        f"- Known good solution sum: ~{target}\n\n"
        f"## Requirements\n"
        f"Define a function `run_packing()` that returns:\n"
        f"  `centers: np.ndarray` shape ({num_circles}, 2)\n"
        f"  `radii: np.ndarray` shape ({num_circles},)\n"
        f"  `sum_radii: float`\n\n"
        f"Constraints:\n"
        f"- Circles must be inside [0,1]²\n"
        f"- No overlaps allowed\n"
        f"- All radii must be non-negative\n"
        f"- Use numpy, scipy, math as needed\n"
        f"- No file or network I/O\n\n"
        f"Think step by step, then output the code between ```python and ```."
    )


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_completion_tokens: int = 8000,
    exec_timeout: int = 120,
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "task_type": "circle_packing",
        "hard": 0,
        "soft": 0.0,
        "raw_score": 0.0,
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        system = _build_system(skill_content)
        user = _build_user(item)

        resp_text, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )

        result["response"] = resp_text
        result["agent_ok"] = True

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(resp_text)

        eval_result = evaluate_rollout(resp_text, eval_timeout=exec_timeout)
        result["hard"] = eval_result.get("hard", 0)
        result["soft"] = eval_result.get("soft", 0.0)
        result["raw_score"] = eval_result.get("raw_score", 0.0)
        result["fail_reason"] = eval_result.get("fail_reason", "")

        if "centers" in eval_result:
            result["centers"] = eval_result["centers"]
        if "radii" in eval_result:
            result["radii"] = eval_result["radii"]

        eval_detail = (
            f"[EVALUATION RESULT]\n"
            f"Item: {item_id}\n"
            f"Hard: {result['hard']}\n"
            f"Soft: {result['soft']:.4f}\n"
            f"Raw Score: {result['raw_score']:.6f}\n"
            f"Fail reason: {result['fail_reason']}"
        )
        conversation = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": resp_text},
            {"role": "system", "content": eval_detail},
        ]
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

    except Exception as e:
        result["fail_reason"] = f"error: {e}"

    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    exec_timeout: int = 120,
    workers: int = 8,
    max_completion_tokens: int = 8000,
    task_timeout: int = 300,
) -> list[dict]:
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids = set()
    existing = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))

    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)
    started_at = {}

    def _run_one(it):
        started_at[str(it["id"])] = time.time()
        return process_one(
            it, out_root, skill_content,
            exec_timeout=exec_timeout,
            max_completion_tokens=max_completion_tokens,
        )

    def _timeout_result(it):
        return {
            "id": str(it["id"]),
            "task_type": "circle_packing",
            "hard": 0, "soft": 0.0, "raw_score": 0.0,
            "response": "", "fail_reason": "task-timeout",
            "agent_ok": False,
        }

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if task_timeout
                    and str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = _timeout_result(item)
                        res["fail_reason"] = f"error: {type(e).__name__}: {e}"
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct_count += 1
                    avg = correct_count / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(hard={avg:.3f}) id={res['id']} "
                        f"score={res.get('raw_score', 0):.4f} "
                        f"fail={res.get('fail_reason', '')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    completed += 1
                    print(f"    [rollout] {completed}/{total} id={res['id']} TIMEOUT", flush=True)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return results


