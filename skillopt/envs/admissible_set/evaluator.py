from __future__ import annotations
import json, re, sys, textwrap

_INVALID_SCORE = 0.001

RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback

{user_code}

# ── Auto-run harness ─────────────────────────────────────────────────────────
try:
    achieved = run_admissible_set({dimension}, {weight})
    if not isinstance(achieved, int):
        achieved = int(achieved)
    print("__RESULT__" + json.dumps({{
        "achieved": achieved,
        "success": True
    }}) + "__END__")
except Exception:
    traceback.print_exc()
    print("__RESULT__" + json.dumps({{
        "success": False,
        "error": traceback.format_exc()
    }}) + "__END__", flush=True)
    sys.exit(1)
""")


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


def normalize_score(achieved: int, optimal: int) -> float:
    if optimal > 0:
        return min(1.0, achieved / optimal)
    return min(1.0, achieved / max(achieved, 1))


def evaluate_rollout(response: str, dimension: int = 12, weight: int = 7, eval_timeout: int = 60) -> dict:
    code = extract_code(response)
    if not code:
        return {"hard": 0, "soft": 0.0, "raw_score": 0, "fail_reason": "no_code", "response": response}

    script = RUNNER_TEMPLATE.format(
        user_code=code,
        dimension=dimension,
        weight=weight,
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
            if data.get("success"):
                achieved = int(data["achieved"])
                soft = normalize_score(achieved, 1)
                return {
                    "hard": 1, "soft": soft, "raw_score": achieved,
                    "achieved": achieved,
                    "fail_reason": "", "response": response,
                }
            else:
                return {"hard": 0, "soft": 0.0, "raw_score": 0,
                        "fail_reason": "exec_error", "response": response}
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
