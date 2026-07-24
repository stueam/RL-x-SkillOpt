"""SkillOpt adapter for lightweight TTT-Discover math discovery tasks."""

from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.tttdiscover_math.dataloader import TTTDiscoverMathDataLoader
from skillopt.envs.tttdiscover_math.rollout import run_batch


class TTTDiscoverMathAdapter(EnvAdapter):
    """Adapter for single-problem continuous-reward math discovery."""

    def __init__(
        self,
        problem_type: str = "circle_packing_26",
        discovery_train_size: int = 24,
        discovery_val_size: int = 8,
        discovery_test_size: int = 16,
        max_turns: int = 1,
        exec_timeout: int = 60,
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        seed: int = 42,
        max_completion_tokens: int = 8192,
        num_circles: int = 26,
        circle_target: float = 2.636,
        erdos_target: float = 0.38092,
        erdos_n_points: int = 80,
        reflection_success_threshold: float = 0.85,
        reflection_raw_success_threshold: float = 2.4,
        reflection_low_score_as_failure: bool = True,
        reflection_include_cur_best: bool = True,
        **kwargs,
    ) -> None:
        self.problem_type = str(problem_type or "circle_packing_26")
        self.max_turns = int(max_turns or 1)
        self.exec_timeout = int(exec_timeout or 60)
        self.workers = int(workers or 4)
        self.analyst_workers = int(analyst_workers or 4)
        self.failure_only = bool(failure_only)
        self.minibatch_size = int(minibatch_size or 4)
        self.edit_budget = int(edit_budget or 4)
        self.max_completion_tokens = int(max_completion_tokens or 8192)
        self.num_circles = int(num_circles or 26)
        self.circle_target = float(circle_target or 2.636)
        self.erdos_target = float(erdos_target or 0.38092)
        self.erdos_n_points = int(erdos_n_points or 80)
        self.reflection_success_threshold = float(reflection_success_threshold or 0.85)
        self.reflection_raw_success_threshold = float(
            reflection_raw_success_threshold or 2.4
        )
        if (
            self.problem_type.startswith("erdos_min_overlap")
            and self.reflection_raw_success_threshold >= 1.0
        ):
            self.reflection_raw_success_threshold = self.erdos_target * 1.10
        self.reflection_low_score_as_failure = bool(reflection_low_score_as_failure)
        self.reflection_include_cur_best = bool(reflection_include_cur_best)
        self.dataloader = TTTDiscoverMathDataLoader(
            problem_type=self.problem_type,
            train_size=discovery_train_size,
            val_size=discovery_val_size,
            test_size=discovery_test_size,
            seed=seed,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        self.reflection_success_threshold = float(
            cfg.get("reflection_success_threshold", self.reflection_success_threshold)
            or self.reflection_success_threshold
        )
        self.reflection_raw_success_threshold = float(
            cfg.get(
                "reflection_raw_success_threshold",
                self.reflection_raw_success_threshold,
            )
            or self.reflection_raw_success_threshold
        )
        if (
            self.problem_type.startswith("erdos_min_overlap")
            and self.reflection_raw_success_threshold >= 1.0
        ):
            self.reflection_raw_success_threshold = self.erdos_target * 1.10
        self.reflection_low_score_as_failure = _as_bool(
            cfg.get("reflection_low_score_as_failure", self.reflection_low_score_as_failure),
            default=self.reflection_low_score_as_failure,
        )
        self.reflection_include_cur_best = _as_bool(
            cfg.get("reflection_include_cur_best", self.reflection_include_cur_best),
            default=self.reflection_include_cur_best,
        )

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        return run_batch(
            items=list(env_manager or []),
            out_root=out_dir,
            skill_content=skill_content,
            workers=self.workers,
            num_circles=self.num_circles,
            circle_target=self.circle_target,
            erdos_target=self.erdos_target,
            erdos_n_points=self.erdos_n_points,
            exec_timeout=self.exec_timeout,
            max_completion_tokens=self.max_completion_tokens,
        )

    def _prepare_reflection_results(self, results: list[dict]) -> list[dict]:
        """Separate validity from optimization quality for reflection.

        Selection/evaluation should optimize the continuous normalized reward.
        Reflection needs a different signal:
        - invalid programs are hard failures;
        - valid but low-scoring programs are optimization failures;
        - only valid high-scoring programs or the current best valid example are
          success examples.

        Without this split, a safe fallback packing with reward around 0.79 can
        be treated as a success and reinforced, which is not what we want.
        """
        prepared: list[dict] = []
        best_raw = None
        best_index = None
        for idx, row in enumerate(results):
            if not bool(row.get("valid")):
                continue
            raw = row.get("raw_score")
            if raw is None:
                continue
            try:
                raw_val = float(raw)
            except (TypeError, ValueError):
                continue
            if best_raw is None or self._is_better_raw(raw_val, best_raw):
                best_raw = raw_val
                best_index = idx

        for row in results:
            item = dict(row)
            valid = bool(item.get("valid"))
            score = float(item.get("normalized_score", item.get("hard", 0.0)) or 0.0)
            raw = item.get("raw_score")
            try:
                raw_value = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                raw_value = None
            task = str(item.get("task_type") or self.problem_type)
            is_cur_best = (
                self.reflection_include_cur_best
                and best_index is not None
                and item.get("id") == results[best_index].get("id")
            )
            is_high_score = raw_value is not None and self._meets_raw_success(
                task,
                raw_value,
            )

            item["reflection_score"] = score
            item["reflection_success_threshold"] = self.reflection_success_threshold
            item["reflection_raw_success_threshold"] = self.reflection_raw_success_threshold
            item["reflection_is_cur_best"] = bool(is_cur_best)
            item["reflection_is_high_score"] = bool(is_high_score)

            if not valid:
                item["hard"] = 0.0
                item["soft"] = score
                if not item.get("fail_reason"):
                    item["fail_reason"] = "invalid construction or execution failure"
            elif is_high_score or is_cur_best or score >= self.reflection_success_threshold:
                item["hard"] = 1.0
                item["soft"] = score
                item["fail_reason"] = ""
            elif self.reflection_low_score_as_failure:
                item["hard"] = 0.0
                item["soft"] = score
                item["fail_reason"] = (
                    f"valid but low-scoring construction: raw_score={raw}, "
                    f"normalized_reward={score:.4f}, required_success_threshold="
                    f"{self.reflection_success_threshold:.4f}, raw_success_threshold="
                    f"{self.reflection_raw_success_threshold:.4f}. Treat validity as the floor; "
                    "learn how to improve the search strategy from higher-scoring valid examples "
                    "and the current best valid candidate."
                )
            else:
                item["hard"] = 1.0
                item["soft"] = score
                item["fail_reason"] = ""

            if task.startswith("circle_packing") and valid and raw is not None:
                item["task_description"] = (
                    f"{item.get('task_description', task)} "
                    f"(circle packing raw_score={float(raw):.6f}; target={self.circle_target:.6f})"
                )
            if is_cur_best and raw_value is not None:
                item["task_description"] = (
                    f"{item.get('task_description', task)} "
                    f"(cur_best raw_score={raw_value:.6f}; raw_success_threshold="
                    f"{self.reflection_raw_success_threshold:.6f})"
                )
            prepared.append(item)
        return prepared

    def _is_better_raw(self, candidate: float, incumbent: float) -> bool:
        if self.problem_type.startswith("erdos_min_overlap"):
            return candidate < incumbent
        return candidate > incumbent

    def _meets_raw_success(self, task: str, raw_value: float) -> bool:
        if task.startswith("erdos_min_overlap"):
            return raw_value <= self.reflection_raw_success_threshold
        return raw_value >= self.reflection_raw_success_threshold

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        from skillopt.gradient.reflect import run_minibatch_reflect

        reflection_results = self._prepare_reflection_results(results)

        # ── Inject high-score buffer items as success examples ─────────
        high_score_buffer: list[dict] = kwargs.pop("high_score_buffer", [])
        if high_score_buffer:
            for buf_item in high_score_buffer:
                if not bool(buf_item.get("valid")):
                    continue
                injected = dict(buf_item)
                normalized = float(injected.get("normalized_score", 0) or 0)
                injected["hard"] = 1.0
                injected["soft"] = normalized
                injected["reflection_score"] = normalized
                injected["reflection_success_threshold"] = self.reflection_success_threshold
                injected["reflection_raw_success_threshold"] = self.reflection_raw_success_threshold
                injected["reflection_is_cur_best"] = False
                injected["reflection_is_high_score"] = True
                injected["fail_reason"] = ""
                injected["_from_buffer"] = True
                # Tag the trajectory description so the analyst knows
                task = str(injected.get("task_type") or self.problem_type)
                raw = injected.get("raw_score")
                if raw is not None:
                    injected["task_description"] = (
                        f"{injected.get('task_description', task)} "
                        f"(HIGH-SCORE BUFFER: raw_score={float(raw):.6f}; "
                        f"replay of best example across all prior steps)"
                    )
                reflection_results.append(injected)

        return run_minibatch_reflect(
            results=reflection_results,
            skill_content=skill_content,
            prediction_dir=kwargs.get(
                "prediction_dir", os.path.join(out_dir, "predictions")
            ),
            patches_dir=kwargs.get(
                "patches_dir", os.path.join(out_dir, "patches")
            ),
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get("random_seed"),
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            meta_skill_context=kwargs.get("meta_skill_context", ""),
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        return [self.problem_type]


def _as_bool(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
