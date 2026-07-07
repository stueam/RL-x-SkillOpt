"""Circle packing validation and scoring.
"""
from __future__ import annotations

import json
import math
import re
import sys
import textwrap

import numpy as np

# Upper bound for normalizing sum of radii (N=26, theoretical max ~2.7)
_UPPER_BOUND = 3.0
# Tiny score awarded when code runs but packing is invalid,
# so the gate sees >0 signal and can accept skill updates.
_INVALID_PACKING_SCORE = 0.001


def validate_packing(centers, radii):
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values detected"
    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False, f"Circle {i} at ({x},{y}) r={r} is outside unit square"
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"
    return True, ""


def check_packing_correctness(centers, radii, num_circles):
    shape_valid = centers.shape == (num_circles, 2) and radii.shape == (num_circles,)
    if not shape_valid:
        return False
    valid, _ = validate_packing(centers, radii)
    return valid


def normalize_score(sum_radii: float) -> float:
    return min(sum_radii / _UPPER_BOUND, 1.0)


RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback
import numpy as np

{user_code}

try:
    centers, radii, sum_r = run_packing()
    if isinstance(centers, np.ndarray):
        centers = centers.tolist()
    if isinstance(radii, np.ndarray):
        radii = radii.tolist()
    print("__RESULT__" + json.dumps({{
        "centers": centers,
        "radii": radii,
        "sum_radii": float(sum_r),
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
    # 1) ```python\n...```
    matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # 2) ```\n...```
    matches = re.findall(r"```\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # 3) Some models output "python\ncode..." without backticks
    matches = re.findall(r"(?:^|\n)python\n(.+?)(?=\n\n|$)", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # 4) "```python\n...```" with possible whitespace variations
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def evaluate_rollout(response: str, eval_timeout: int = 120) -> dict:
    code = extract_code(response)
    if not code:
        return {"hard": 0, "soft": 0.0, "raw_score": 0.0, "fail_reason": "no_code", "response": response}

    script = RUNNER_TEMPLATE.format(user_code=code)

    import subprocess
    import tempfile
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
                centers = np.array(data["centers"])
                radii = np.array(data["radii"])
                sum_r = data["sum_radii"]
                valid = check_packing_correctness(centers, radii, len(radii))
                if valid:
                    soft = normalize_score(sum_r)
                    return {
                        "hard": 1,
                        "soft": soft,
                        "raw_score": sum_r,
                        "centers": data["centers"],
                        "radii": data["radii"],
                        "fail_reason": "",
                        "response": response,
                    }
                else:
                    # Code ran but packing invalid -> hard=1 (code worked), soft=tiny (gate signal)
                    return {"hard": 1, "soft": _INVALID_PACKING_SCORE, "raw_score": 0.0,
                            "fail_reason": "invalid_packing", "response": response}
            else:
                return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                        "fail_reason": f"exec_error", "response": response}
        else:
            stderr = (proc.stderr or "")[:500]
            reason = f"no_result|stderr={stderr}" if stderr else "no_result"
            return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                    "fail_reason": reason, "response": response}
    except subprocess.TimeoutExpired:
        return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                "fail_reason": "timeout", "response": response}
    except subprocess.CalledProcessError as e:
        return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                "fail_reason": f"called_process_error|{e.stderr[:300] if e.stderr else str(e)}", "response": response}
    except FileNotFoundError as e:
        return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                "fail_reason": f"file_not_found: {e}", "response": response}
    except Exception as e:
        return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                "fail_reason": f"error: {type(e).__name__}: {e}", "response": response}
    finally:
        import os
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
