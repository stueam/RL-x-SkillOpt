"""Rollout and local scoring for lightweight TTT-Discover math tasks."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np

from skillopt.model import chat_target


ALLOWED_IMPORT_ROOTS = {"numpy", "math", "random", "itertools", "time"}
BANNED_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
    "input",
    "breakpoint",
    "help",
    "dir",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}
BANNED_IMPORT_ROOTS = {
    "asyncio",
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "threading",
}


def _extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


def _validate_candidate_ast(code: str) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"from-import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute not allowed: {node.attr}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                root = func.value
                if isinstance(root, ast.Name) and root.id in BANNED_IMPORT_ROOTS:
                    errors.append(f"call through banned module: {root.id}.{func.attr}")
    return sorted(set(errors))


def validate_packing(centers: np.ndarray, radii: np.ndarray, n: int) -> tuple[bool, str, float]:
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers, dtype=float)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii, dtype=float)
    if centers.shape != (n, 2):
        return False, f"centers must have shape ({n}, 2), got {centers.shape}", 0.0
    if radii.shape != (n,):
        return False, f"radii must have shape ({n},), got {radii.shape}", 0.0
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False, "centers/radii contain NaN or inf", 0.0
    if np.any(radii < 0):
        return False, "negative radius", 0.0
    if np.any(centers - radii[:, None] < -1e-10) or np.any(centers + radii[:, None] > 1.0 + 1e-10):
        return False, "circle outside unit square", 0.0

    for i in range(n):
        deltas = centers[i + 1 :] - centers[i]
        if len(deltas) == 0:
            continue
        dists = np.sqrt(np.sum(deltas * deltas, axis=1))
        min_allowed = radii[i] + radii[i + 1 :] - 1e-10
        if np.any(dists < min_allowed):
            j = int(np.where(dists < min_allowed)[0][0]) + i + 1
            return False, f"circles {i} and {j} overlap", 0.0
    raw = float(np.sum(radii))
    return True, f"valid packing; sum_radii={raw:.8f}", raw


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int) -> float:
    h_values = np.array(h_values, dtype=np.float64)
    if h_values.ndim != 1:
        raise ValueError(f"h_values must be 1D, got shape {h_values.shape}")
    if h_values.shape[0] != int(n_points):
        raise ValueError(f"expected {n_points} points, got {h_values.shape[0]}")
    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf")
    if np.any(h_values < -1e-10) or np.any(h_values > 1 + 1e-10):
        raise ValueError(f"h_values outside [0, 1]: [{h_values.min()}, {h_values.max()}]")

    target_sum = int(n_points) / 2.0
    current_sum = float(np.sum(h_values))
    if abs(current_sum - target_sum) > 1e-8:
        if abs(current_sum) < 1e-12:
            raise ValueError("h_values sum is zero")
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < -1e-10) or np.any(h_values > 1 + 1e-10):
            raise ValueError("normalization pushes h_values outside [0, 1]")

    dx = 2.0 / int(n_points)
    computed_c5 = float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))
    if not np.isfinite(computed_c5):
        raise ValueError("computed C5 is not finite")
    if not np.isclose(float(c5_achieved), computed_c5, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
    return computed_c5


def _circle_prompt(item: dict, skill_content: str, n: int, target: float) -> tuple[str, str]:
    system = (
        "You are a mathematical discovery agent. Write Python code that searches for dense circle packings. "
        "Search aggressively using multiple trials, local improvement, and best-valid tracking. "
        "The skill provides guidance; use it to write better searchers.\n\n"
        f"## Skill\n{skill_content.strip() if skill_content.strip() else '(none)'}"
    )
    user = f"""Pack {n} equal or variable circles in the unit square [0,1]x[0,1] to maximize sum(radii).

Goal: beat sum_radii = {target}. Known-good packings reach >2.6.
This attempt seed: {item['attempt_seed']}.

## Requirements
- Define `def run_packing():` → `(centers, radii, sum_radii)`
- centers: ({n}, 2) float ndarray in [0,1]
- radii: ({n},) nonnegative float ndarray
- No overlaps: dist(c_i, c_j) >= r_i + r_j - 1e-10
- All circles strictly inside unit square
- Use only numpy, math, random, itertools, time
- Runtime < 55 seconds

## Search Strategy (implement ALL of these)
1. **Multi-trial**: Run 10-100 independent trials with different seeds
2. **Local improvement**: Perturb → repair overlaps → expand radii
3. **Best-valid tracking**: Keep the best valid (centers, radii, sum) across all trials
4. **Layered approach**: Try several packing strategies and pick the best valid

## Good techniques to try
- Greedy: place circles one at a time, each at the position allowing the largest radius
- Perturb + expand: start from a dense grid, jitter centers, shrink to remove overlaps, then slowly expand
- Boundary packing: large circles at corners/edges, smaller ones fill interior gaps
- Hexagonal lattice: 26 circles in a hexagonal arrangement, then perturb+expand
- Force-directed: apply repulsion between overlapping circles, move apart gradually

Output ONLY the final code block. No explanations, no markdown beyond the code fence.
"""
    return system, user


def _erdos_prompt(item: dict, skill_content: str, target: float, n_points: int) -> tuple[str, str]:
    system = (
        "You are a mathematical discovery agent. Generate executable Python code for the current "
        "continuous-reward benchmark. Follow the skill when useful, but prioritize a valid construction.\n\n"
        f"## Skill\n{skill_content.strip() if skill_content.strip() else '(none)'}"
    )
    user = f"""Find a numerical construction for the Erdos minimum overlap problem.

Represent h as `n_points` samples on [0, 2], with 0 <= h[i] <= 1 and sum(h) = n_points / 2.
The evaluator computes:
    C5 = max(np.correlate(h, 1 - h, mode="full") * (2.0 / n_points))

Goal: minimize C5. Reference target for normalization: C5 ~= {target}.

Rules:
- Return one final Python code block.
- Define exactly this entrypoint: `def run(seed=42, budget_s=20, **kwargs):`
- `run` must return `(h_values, c5_bound, n_points)`.
- Use n_points <= {n_points}; smaller is fine if it improves speed.
- Use only numpy, math, random, itertools, and time.
- No filesystem, network, subprocesses, multiprocessing, or dynamic imports.
- This attempt seed is {item['attempt_seed']}.
- A global `initial_h_values` may be available; use it if useful, but avoid local traps.

Output only the final code block.
"""
    return system, user


def _runner_source(task: str, n: int, erdos_n: int, seed: int) -> str:
    validators = inspect_validators_source()
    if task.startswith("circle_packing"):
        call = f"""
import candidate
out = candidate.run_packing()
centers, radii, reported = out
ok, msg, raw = validate_packing(centers, radii, {n})
_result_payload = {{"ok": ok, "msg": msg, "raw_score": raw, "reported": float(reported) if isinstance(reported, (int, float, np.floating)) else None}}
"""
    else:
        initial = _initial_erdos_values(erdos_n, seed)
        call = f"""
import candidate
candidate.initial_h_values = np.array({initial!r}, dtype=float)
out = candidate.run(seed={seed}, budget_s=20)
h_values, c5_bound, n_points = out
raw = verify_c5_solution(h_values, c5_bound, int(n_points))
_result_payload = {{"ok": True, "msg": f"valid erdos construction; C5={{raw:.8f}}", "raw_score": raw, "n_points": int(n_points)}}
"""
    return f"""
import contextlib
import io
import json
import numpy as np

{validators}

buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
{textwrap.indent(call.strip(), '        ')}
except Exception as exc:
    _result_payload = {{"ok": False, "msg": f"{{type(exc).__name__}}: {{exc}}", "raw_score": None}}
_result_payload["stdout"] = buf.getvalue()[-4000:]
print(json.dumps(_result_payload))
"""


def inspect_validators_source() -> str:
    return textwrap.dedent(
        """
        import numpy as np

        def validate_packing(centers, radii, n):
            if not isinstance(centers, np.ndarray):
                centers = np.array(centers, dtype=float)
            if not isinstance(radii, np.ndarray):
                radii = np.array(radii, dtype=float)
            if centers.shape != (n, 2):
                return False, f"centers must have shape ({n}, 2), got {centers.shape}", 0.0
            if radii.shape != (n,):
                return False, f"radii must have shape ({n},), got {radii.shape}", 0.0
            if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
                return False, "centers/radii contain NaN or inf", 0.0
            if np.any(radii < 0):
                return False, "negative radius", 0.0
            if np.any(centers - radii[:, None] < -1e-10) or np.any(centers + radii[:, None] > 1.0 + 1e-10):
                return False, "circle outside unit square", 0.0
            for i in range(n):
                deltas = centers[i + 1:] - centers[i]
                if len(deltas) == 0:
                    continue
                dists = np.sqrt(np.sum(deltas * deltas, axis=1))
                min_allowed = radii[i] + radii[i + 1:] - 1e-10
                if np.any(dists < min_allowed):
                    j = int(np.where(dists < min_allowed)[0][0]) + i + 1
                    return False, f"circles {i} and {j} overlap", 0.0
            return True, f"valid packing; sum_radii={float(np.sum(radii)):.8f}", float(np.sum(radii))

        def verify_c5_solution(h_values, c5_achieved, n_points):
            h_values = np.array(h_values, dtype=np.float64)
            if h_values.ndim != 1:
                raise ValueError(f"h_values must be 1D, got shape {h_values.shape}")
            if h_values.shape[0] != int(n_points):
                raise ValueError(f"expected {n_points} points, got {h_values.shape[0]}")
            if not np.all(np.isfinite(h_values)):
                raise ValueError("h_values contain NaN or inf")
            if np.any(h_values < -1e-10) or np.any(h_values > 1 + 1e-10):
                raise ValueError(f"h_values outside [0, 1]: [{h_values.min()}, {h_values.max()}]")
            target_sum = int(n_points) / 2.0
            current_sum = float(np.sum(h_values))
            if abs(current_sum - target_sum) > 1e-8:
                if abs(current_sum) < 1e-12:
                    raise ValueError("h_values sum is zero")
                h_values = h_values * (target_sum / current_sum)
                if np.any(h_values < -1e-10) or np.any(h_values > 1 + 1e-10):
                    raise ValueError("normalization pushes h_values outside [0, 1]")
            dx = 2.0 / int(n_points)
            computed_c5 = float(np.max(np.correlate(h_values, 1.0 - h_values, mode="full") * dx))
            if not np.isfinite(computed_c5):
                raise ValueError("computed C5 is not finite")
            if not np.isclose(float(c5_achieved), computed_c5, atol=1e-4):
                raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")
            return computed_c5
        """
    )


def _initial_erdos_values(n_points: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.2, 0.2, n_points)
    perturbation -= np.mean(perturbation)
    values = np.clip(values + perturbation, 1e-6, 1 - 1e-6)
    values *= (n_points / 2.0) / float(np.sum(values))
    return [float(x) for x in values]


def _preexec_limits(timeout_s: int):
    def _set_limits() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (max(timeout_s + 2, 3), max(timeout_s + 4, 5)))
            resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))
            resource.setrlimit(resource.RLIMIT_AS, (2_000_000_000, 2_000_000_000))
        except Exception:
            pass

    return _set_limits


def _run_candidate_subprocess(
    *,
    code: str,
    pred_dir: str,
    task: str,
    num_circles: int,
    erdos_n_points: int,
    seed: int,
    timeout_s: int,
) -> dict[str, Any]:
    ast_errors = _validate_candidate_ast(code)
    if ast_errors:
        return {"ok": False, "msg": "; ".join(ast_errors), "raw_score": None}

    sandbox = Path(pred_dir) / "sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / "candidate.py").write_text(code, encoding="utf-8")
    (sandbox / "runner.py").write_text(
        _runner_source(task, num_circles, erdos_n_points, seed),
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            [sys.executable, "runner.py"],
            cwd=str(sandbox),
            text=True,
            capture_output=True,
            timeout=max(timeout_s, 3),
            preexec_fn=_preexec_limits(timeout_s) if os.name == "posix" else None,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONNOUSERSITE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": f"timeout after {timeout_s}s", "raw_score": None}
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0 and not stdout:
        return {"ok": False, "msg": f"process failed rc={proc.returncode}: {stderr[-1000:]}", "raw_score": None}
    last_line = stdout.splitlines()[-1] if stdout else ""
    try:
        data = json.loads(last_line)
    except Exception:
        return {
            "ok": False,
            "msg": f"runner did not return JSON; rc={proc.returncode}; stderr={stderr[-1000:]}",
            "raw_score": None,
            "stdout": stdout[-1000:],
        }
    data.setdefault("stdout", stdout[-1000:])
    data.setdefault("stderr", stderr[-1000:])
    return data


def _score(task: str, ok: bool, raw_score: float | None, circle_target: float, erdos_target: float) -> float:
    if not ok or raw_score is None:
        return 0.0
    raw = float(raw_score)
    if task.startswith("circle_packing"):
        return max(0.0, min(raw / circle_target, 1.25))
    return max(0.0, min(erdos_target / max(raw, 1e-9), 1.25))


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    num_circles: int = 26,
    circle_target: float = 2.636,
    erdos_target: float = 0.38092,
    erdos_n_points: int = 80,
    exec_timeout: int = 60,
    max_completion_tokens: int = 8192,
) -> dict:
    item_id = str(item["id"])
    task = str(item.get("problem_type") or "circle_packing_26")
    pred_dir = os.path.join(out_root, "predictions", item_id.replace("/", "_"))
    os.makedirs(pred_dir, exist_ok=True)

    if task.startswith("circle_packing"):
        system, user = _circle_prompt(item, skill_content, num_circles, circle_target)
    elif task.startswith("erdos_min_overlap"):
        system, user = _erdos_prompt(item, skill_content, erdos_target, erdos_n_points)
    else:
        raise ValueError(f"Unsupported TTTDiscover math problem_type={task!r}")

    result = {
        "id": item_id,
        "task_type": task,
        "task_description": item.get("task_description", task),
        "question": user,
        "hard": 0.0,
        "soft": 0.0,
        "raw_score": None,
        "normalized_score": 0.0,
        "valid": False,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 1,
    }

    try:
        response, _ = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rollout",
            timeout=exec_timeout,
        )
        code = _extract_code(response)
        (Path(pred_dir) / "candidate.py").write_text(code, encoding="utf-8")
        eval_result = _run_candidate_subprocess(
            code=code,
            pred_dir=pred_dir,
            task=task,
            num_circles=num_circles,
            erdos_n_points=erdos_n_points,
            seed=int(item.get("attempt_seed", 42)),
            timeout_s=exec_timeout,
        )
        ok = bool(eval_result.get("ok"))
        raw_score = eval_result.get("raw_score")
        normalized = _score(task, ok, raw_score, circle_target, erdos_target)
        result.update(
            {
                "hard": normalized,
                "soft": normalized,
                "raw_score": raw_score,
                "normalized_score": normalized,
                "valid": ok,
                "predicted_answer": code[:4000],
                "response": response,
                "agent_ok": True,
                "fail_reason": "" if ok else str(eval_result.get("msg") or "invalid solution"),
            }
        )

        conversation = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
            {
                "role": "system",
                "content": (
                    f"Evaluation: valid={ok}; raw_score={raw_score}; "
                    f"normalized_reward={normalized:.6f}; message={eval_result.get('msg', '')}"
                ),
            },
        ]
        (Path(pred_dir) / "target_system_prompt.txt").write_text(system, encoding="utf-8")
        (Path(pred_dir) / "target_user_prompt.txt").write_text(user, encoding="utf-8")
        (Path(pred_dir) / "evaluation.json").write_text(json.dumps(eval_result, indent=2), encoding="utf-8")
        (Path(pred_dir) / "conversation.json").write_text(json.dumps(conversation, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        result["fail_reason"] = f"error: {type(exc).__name__}: {exc}"

    return result


def run_batch(
    *,
    items: list[dict],
    out_root: str,
    skill_content: str,
    workers: int = 4,
    num_circles: int = 26,
    circle_target: float = 2.636,
    erdos_target: float = 0.38092,
    erdos_n_points: int = 80,
    exec_timeout: int = 60,
    max_completion_tokens: int = 8192,
) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")
    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                done_ids.add(str(row["id"]))
                existing.append(row)
    pending = [item for item in items if str(item["id"]) not in done_ids]
    if not pending:
        return existing

    results = list(existing)
    total = len(existing) + len(pending)
    completed = len(existing)
    score_sum = sum(float(row.get("hard", 0.0) or 0.0) for row in existing)
    best_raw = None
    for row in existing:
        raw = row.get("raw_score")
        if raw is not None:
            if best_raw is None:
                best_raw = raw
            elif str(row.get("task_type", "")).startswith("circle_packing"):
                best_raw = max(best_raw, raw)
            else:
                best_raw = min(best_raw, raw)

    with open(results_path, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(
                    process_one,
                    item,
                    out_root,
                    skill_content,
                    num_circles=num_circles,
                    circle_target=circle_target,
                    erdos_target=erdos_target,
                    erdos_n_points=erdos_n_points,
                    exec_timeout=exec_timeout,
                    max_completion_tokens=max_completion_tokens,
                ): item
                for item in pending
            }
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending_futs.remove(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        item = futs[fut]
                        row = {
                            "id": str(item["id"]),
                            "task_type": item.get("problem_type", ""),
                            "hard": 0.0,
                            "soft": 0.0,
                            "raw_score": None,
                            "valid": False,
                            "fail_reason": f"error: {type(exc).__name__}: {exc}",
                            "agent_ok": False,
                        }
                    results.append(row)
                    completed += 1
                    score_sum += float(row.get("hard", 0.0) or 0.0)
                    raw = row.get("raw_score")
                    task = str(row.get("task_type", ""))
                    if raw is not None:
                        if best_raw is None:
                            best_raw = raw
                        elif task.startswith("circle_packing"):
                            best_raw = max(best_raw, raw)
                        else:
                            best_raw = min(best_raw, raw)
                    mean_score = score_sum / max(completed, 1)
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"reward={mean_score:.4f} raw={raw} best_raw={best_raw} "
                        f"valid={row.get('valid')}",
                        flush=True,
                    )
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()

    return results
