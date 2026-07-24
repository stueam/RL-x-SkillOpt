from __future__ import annotations
import json, os, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.admissible_set.evaluator import evaluate_rollout
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content):
    section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="admissible_set").format(skill_section=section)


def _build_user(item):
    dimension = item.get("dimension", 15)
    weight = item.get("weight", 10)
    optimal = item.get("optimal", "?")
    return (
        f"## Problem\n"
        f"Write a complete Python program to construct a maximum-cardinality "
        f"symmetric constant-weight admissible set I({dimension}, {weight}).\n\n"
        f"## Parameters\n"
        f"- Dimension n = {dimension}\n"
        f"- Weight w = {weight}\n"
        f"- Known optimal: {optimal}\n\n"
        f"Define `run_admissible_set(dimension, weight) -> int` between ```python and ```."
    )


def process_one(item, out_root, skill_content, *, max_completion_tokens=8000, exec_timeout=60):
    item_id = str(item["id"])
    result = {"id": item_id, "task_type": "admissible_set", "hard": 0, "soft": 0.0,
              "raw_score": 0, "response": "", "fail_reason": "", "agent_ok": False}
    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)
        system = _build_system(skill_content)
        user = _build_user(item)
        resp_text, _ = chat_target(
            system=system, user=user,
            max_completion_tokens=max_completion_tokens, retries=5,
            stage="rollout", timeout=exec_timeout,
        )
        result["response"] = resp_text
        result["agent_ok"] = True
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "response.txt"), "w", encoding="utf-8") as f:
            f.write(resp_text)
        dim = item.get("dimension", 15)
        wt = item.get("weight", 10)
        er = evaluate_rollout(resp_text, dimension=dim, weight=wt, eval_timeout=exec_timeout)
        result.update({k: er.get(k) for k in ("hard", "soft", "raw_score", "fail_reason")})
        if "achieved" in er:
            result["achieved"] = er["achieved"]
        detail = (
            f"[EVALUATION RESULT]\nItem: {item_id}\n"
            f"Hard: {result['hard']}\nSoft: {result['soft']:.4f}\n"
            f"Achieved: {result.get('achieved', 0)}\nFail: {result['fail_reason']}"
        )
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump([
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp_text},
                {"role": "system", "content": detail},
            ], f, ensure_ascii=False, indent=2)
    except Exception as e:
        result["fail_reason"] = f"error: {e}"
    return result


def run_batch(items, out_root, skill_content, *, exec_timeout=60, workers=8,
              max_completion_tokens=8000, task_timeout=120):
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)
    done_ids, existing = set(), []
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
    completed, correct_count = len(existing), sum(1 for r in existing if r.get("hard"))
    results = list(existing)

    def _run_one(it):
        return process_one(it, out_root, skill_content, exec_timeout=exec_timeout,
                           max_completion_tokens=max_completion_tokens)

    def _timeout_result(it):
        return {"id": str(it["id"]), "task_type": "admissible_set", "hard": 0, "soft": 0.0,
                "raw_score": 0, "response": "", "fail_reason": "task-timeout", "agent_ok": False}

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                for fut in done:
                    pending_futs.remove(fut)
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = _timeout_result(futs[fut])
                        res["fail_reason"] = f"error: {e}"
                    results.append(res)
                    completed += 1
                    correct_count += 1 if res.get("hard") else 0
                    print(f"    [rollout] {completed}/{total} (hard={correct_count/max(completed,1):.3f}) "
                          f"id={res['id']} achieved={res.get('raw_score', 0)} fail={res.get('fail_reason','')}",
                          flush=True)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return results
