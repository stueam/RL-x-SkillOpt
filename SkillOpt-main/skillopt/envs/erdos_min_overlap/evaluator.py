"""Erdos minimum overlap problem validation and scoring.
"""
from __future__ import annotations

import json
import math
import re
import sys
import textwrap

import numpy as np

# The theoretical lower bound for C5 is ~0.3808.
# We use 0.5 as the upper bound for normalization since
# any reasonable solution should be below this.
_UPPER_BOUND = 0.5
# Tiny score awarded when code runs but solution is invalid
_INVALID_SOLUTION_SCORE = 0.001


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    TOL = 1e-9
    if np.any(h_values < -TOL) or np.any(h_values > 1 + TOL):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")
    h_values = np.clip(h_values, 0.0, 1.0)

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if not np.isclose(current_sum, target_sum, rtol=1e-10):
        h_values = h_values * (target_sum / current_sum)
        h_values = np.clip(h_values, 0.0, 1.0)

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def normalize_score(c5_bound: float) -> float:
    return max(0.0, min(1.0, 1.0 - c5_bound / _UPPER_BOUND))


RUNNER_TEMPLATE = textwrap.dedent("""\
import json, sys, traceback
import numpy as np

{user_code}

try:
    h_values, c5_bound, n_points = run()
    if isinstance(h_values, np.ndarray):
        h_values = h_values.tolist()
    print("__RESULT__" + json.dumps({{
        "h_values": h_values,
        "c5_bound": float(c5_bound),
        "n_points": int(n_points),
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
    matches = re.findall(r"(?:^|\n)python\n(.+?)(?=\n\n|$)", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def evaluate_rollout(response: str, eval_timeout: int = 600) -> dict:
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
                h_values = np.array(data["h_values"])
                c5_bound = float(data["c5_bound"])
                n_points = int(data["n_points"])

                try:
                    c5_bound = verify_c5_solution(h_values, c5_bound, n_points)
                except (ValueError, AssertionError) as e:
                    return {"hard": 1, "soft": _INVALID_SOLUTION_SCORE, "raw_score": 0.0,
                            "fail_reason": f"invalid_solution: {e}", "response": response}

                if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
                    return {"hard": 1, "soft": _INVALID_SOLUTION_SCORE, "raw_score": 0.0,
                            "fail_reason": "invalid_c5_bound", "response": response}

                soft = normalize_score(c5_bound)
                return {
                    "hard": 1,
                    "soft": soft,
                    "raw_score": c5_bound,
                    "h_values": data["h_values"],
                    "c5_bound": c5_bound,
                    "n_points": n_points,
                    "fail_reason": "",
                    "response": response,
                }
            else:
                return {"hard": 0, "soft": 0.0, "raw_score": 0.0,
                        "fail_reason": "exec_error", "response": response}
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
