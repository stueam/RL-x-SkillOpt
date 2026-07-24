"""REI evaluator: Gate × Quality.

Gate = valid regex + matches all positives + rejects all negatives.
Quality = held-out generalization accuracy (0-1).
"""
from __future__ import annotations
import json, re, sys, textwrap

from benchmarks.rei.problem import validate_regex, score_generalization


def extract_code(text: str) -> str | None:
    if not text or not isinstance(text, str):
        return None
    matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def extract_imports(code: str) -> str:
    """Extract import lines from the code for the runner template."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            lines.append(line)
    return "\n".join(lines)


RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback
import re

{user_code}

# ── Auto-validation ───────────────────────────────────────────────────────────
try:
    pattern = solve({positives!r}, {negatives!r})
    if not isinstance(pattern, str):
        print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "not_a_string"}}) + "__END__")
        sys.exit(0)
    # Gate check
    try:
        re.compile(pattern)
    except re.error as e:
        print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": f"invalid_regex: {{e}}"}}) + "__END__")
        sys.exit(0)
    for p in {positives!r}:
        if not re.fullmatch(pattern, p):
            print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": f"missed_positive: {{p!r}}"}}) + "__END__")
            sys.exit(0)
    for n in {negatives!r}:
        if re.fullmatch(pattern, n):
            print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": f"accepted_negative: {{n!r}}"}}) + "__END__")
            sys.exit(0)
    # Quality = -cost  (paper: cost = alpha*|r| + beta*complexity(r) + gamma*heldout_mismatch_rate)
    # alpha=1, beta=0 (skip complexity for now), gamma=50
    r_len = len(pattern)
    # Held-out mismatch rate
    hits = 0
    total = 0
    for p in {heldout_positives!r}:
        if re.fullmatch(pattern, p):
            hits += 1
        total += 1
    for n in {heldout_negatives!r}:
        if not re.fullmatch(pattern, n):
            hits += 1
        total += 1
    heldout_acc = hits / total if total > 0 else 1.0
    mismatch_rate = 1.0 - heldout_acc
    cost = r_len + 50.0 * mismatch_rate
    quality = -cost
    print("__RESULT__" + json.dumps({{"gate": 1, "quality": quality, "cost": cost, "r_len": r_len, "heldout_acc": heldout_acc, "pattern": pattern}}) + "__END__")
except Exception:
    traceback.print_exc()
    print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "exec_error"}}) + "__END__", flush=True)
    sys.exit(1)
""")


def evaluate_rollout(response: str, positives: list, negatives: list,
                     heldout_positives: list, heldout_negatives: list,
                     eval_timeout: int = 30) -> dict:
    code = extract_code(response)
    if not code:
        return {"hard": 0, "soft": 0.0, "raw_score": 0, "fail_reason": "no_code", "response": response}

    script = RUNNER_TEMPLATE.format(
        user_code=code,
        positives=positives,
        negatives=negatives,
        heldout_positives=heldout_positives,
        heldout_negatives=heldout_negatives,
    )

    import subprocess, tempfile
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    try:
        tmp.write(script)
        tmp.close()
        proc = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True, text=True, timeout=eval_timeout,
        )
        stdout = proc.stdout or ""

        result_match = re.search(r"__RESULT__(.*?)__END__", stdout, re.DOTALL)
        if result_match:
            data = json.loads(result_match.group(1))
            if data.get("gate") == 1:
                cost = float(data.get("cost", 0))
                # Normalize cost to 0-1 soft: cost=0 → 1.0, cost=200 → 0.0
                soft = max(0.0, 1.0 - cost / 200.0)
                return {
                    "hard": 1, "soft": soft, "raw_score": int(data.get("r_len", 0)),
                    "achieved": soft,
                    "fail_reason": "", "response": response,
                }
            else:
                err = data.get("error", "gate_fail")
                return {"hard": 0, "soft": 0.0, "raw_score": 0,
                        "fail_reason": err, "response": response}
        else:
            stderr = (proc.stderr or "")[:500]
            reason = f"no_result|stderr={stderr}" if stderr else "no_result"
            return {"hard": 0, "soft": 0.0, "raw_score": 0,
                    "fail_reason": reason, "response": response}
    except subprocess.TimeoutExpired:
        return {"hard": 0, "soft": 0.0, "raw_score": 0,
                "fail_reason": "timeout", "response": response}
    except Exception as e:
        return {"hard": 0, "soft": 0.0, "raw_score": 0,
                "fail_reason": f"error: {type(e).__name__}: {e}", "response": response}
    finally:
        import os
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
