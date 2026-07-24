"""Cap Set evaluator: Gate × Quality。

Gate 是真硬门——构造必须形状对、值域对、无重复、满足 cap 性质。
Quality = 集合大小 m，越大越好。
"""
from __future__ import annotations
import json, re, sys, textwrap

import numpy as np

_OPTIMAL = {1: 2, 2: 4, 3: 9, 4: 20, 5: 45, 6: 112, 7: 236, 8: 496}


def is_cap_set(A: np.ndarray) -> bool:
    """检查 A (shape (m, n), 元素 in {0,1,2}) 是否是合法 cap set。"""
    m = A.shape[0]
    if m < 3:
        return True
    Aset = set(map(tuple, A))
    for i in range(m):
        ai = A[i]
        for j in range(i + 1, m):
            z = tuple(np.mod(-(ai + A[j]), 3))
            if z in Aset and z != tuple(ai) and z != tuple(A[j]):
                return False
    return True


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


RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback
import numpy as np
from itertools import product

{user_code}

# ── Auto-validation ───────────────────────────────────────────────────────────
try:
    A = construct({n}, {seed})
    A = np.asarray(A, dtype=int)
    if A.ndim != 2 or A.shape[1] != {n} or A.shape[0] < 1:
        print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "shape"}}) + "__END__")
        sys.exit(0)
    if not np.all((A >= 0) & (A <= 2)):
        print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "values"}}) + "__END__")
        sys.exit(0)
    if len(set(map(tuple, A))) != A.shape[0]:
        print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "dup"}}) + "__END__")
        sys.exit(0)
    Aset = set(map(tuple, A))
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            z = tuple(np.mod(-(A[i] + A[j]), 3))
            if z in Aset and z != tuple(A[i]) and z != tuple(A[j]):
                print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "not_cap"}}) + "__END__")
                sys.exit(0)
    m = int(A.shape[0])
    print("__RESULT__" + json.dumps({{"gate": 1, "quality": m}}) + "__END__")
except Exception:
    traceback.print_exc()
    print("__RESULT__" + json.dumps({{"gate": 0, "quality": 0, "error": "exec_error"}}) + "__END__", flush=True)
    sys.exit(1)
""")


def evaluate_rollout(response: str, n: int = 4, optimal: int = 20, seed: int = 42, eval_timeout: int = 60) -> dict:
    code = extract_code(response)
    if not code:
        return {"hard": 0, "soft": 0.0, "raw_score": 0, "fail_reason": "no_code", "response": response}

    script = RUNNER_TEMPLATE.format(user_code=code, n=n, seed=seed)

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
                quality = int(data["quality"])
                soft = min(1.0, quality / optimal) if optimal > 0 else 0.0
                return {
                    "hard": 1, "soft": soft, "raw_score": quality,
                    "achieved": quality,
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
