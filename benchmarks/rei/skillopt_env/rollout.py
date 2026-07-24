from __future__ import annotations
import json, os, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .evaluator import evaluate_rollout
from skillopt.model import chat_target
from skillopt.prompts import load_prompt


def _build_system(skill_content):
    section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return load_prompt("rollout_system", env="rei").format(skill_section=section)


def _build_user(item):
    pos = item.get("positives", [])
    neg = item.get("negatives", [])
    desc = item.get("description", "")
    return (
        f"## Problem\n"
        f"Regular Expression Inference: synthesize a regex pattern that matches all positive strings and rejects all negative strings.\n\n"
        f"## Description\n{desc}\n\n"
        f"## Positive examples ({len(pos)})\n{json.dumps(pos, indent=2)}\n\n"
        f"## Negative examples ({len(neg)})\n{json.dumps(neg, indent=2)}\n\n"
        f"Define `solve(positives, negatives) -> str` returning a regex pattern between ```python and ```."
    )


def process_one(item, out_root, skill_content, *, max_completion_tokens=4096, exec_timeout=30):
    item_id = str(item["id"])
    result = {"id": item_id, "task_type": "rei", "hard": 0, "soft": 0.0,
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
        er = evaluate_rollout(
            resp_text,
            positives=list(item.get("positives", [])),
            negatives=list(item.get("negatives", [])),
            heldout_positives=list(item.get("heldout_positives", [])),
            heldout_negatives=list(item.get("heldout_negatives", [])),
            eval_timeout=exec_timeout,
        )
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


def run_batch(items, out_root, skill_content, *, exec_timeout=30, workers=4,
              max_completion_tokens=4096, task_timeout=90):
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
        return {"id": str(it["id"]), "task_type": "rei", "hard": 0, "soft": 0.0,
                "raw_score": 0, "response": "", "fail_reason": "task-timeout", "agent_ok": False}

    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
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
                          f"id={res['id']} soft={res.get('soft', 0):.3f} fail={res.get('fail_reason','')}",
                          flush=True)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return results
