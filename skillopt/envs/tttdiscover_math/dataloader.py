"""Synthetic single-problem dataloader for TTT-Discover style math tasks."""

from __future__ import annotations

import random

from skillopt.datasets.base import BaseDataLoader, BatchSpec


class TTTDiscoverMathDataLoader(BaseDataLoader):
    """Generate repeated attempts for one continuous-reward discovery problem."""

    def __init__(
        self,
        problem_type: str = "circle_packing_26",
        train_size: int = 24,
        val_size: int = 8,
        test_size: int = 16,
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.problem_type = str(problem_type or "circle_packing_26")
        self.train_size = int(train_size or 24)
        self.val_size = int(val_size or 8)
        self.test_size = int(test_size or 16)
        self.seed = int(seed or 42)

    def setup(self, cfg: dict) -> None:
        self.problem_type = str(cfg.get("problem_type", self.problem_type) or self.problem_type)
        self.train_size = int(cfg.get("discovery_train_size", self.train_size) or self.train_size)
        self.val_size = int(cfg.get("discovery_val_size", self.val_size) or self.val_size)
        self.test_size = int(cfg.get("discovery_test_size", self.test_size) or self.test_size)
        self.seed = int(cfg.get("seed", self.seed) or self.seed)
        print(
            f"  [TTTDiscoverMathDataLoader] problem={self.problem_type} "
            f"train={self.train_size} val={self.val_size} test={self.test_size}"
        )

    def get_train_size(self) -> int:
        return self.train_size

    def _make_item(self, split: str, idx: int, seed: int) -> dict:
        return {
            "id": f"{self.problem_type}:{split}:{idx:04d}",
            "problem_type": self.problem_type,
            "attempt_seed": int(seed),
            "task_type": self.problem_type,
            "task_description": f"{self.problem_type} discovery attempt",
        }

    def _items(self, split: str, count: int, seed_offset: int) -> list[dict]:
        return [
            self._make_item(split, idx, self.seed + seed_offset + idx)
            for idx in range(count)
        ]

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        items = self._items("train", self.train_size, seed_offset=0)
        rng = random.Random(seed + epoch * 1000)
        rng.shuffle(items)

        total_batches = steps_per_epoch * accumulation
        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_seed = seed + epoch * 1000 + batch_idx + 1
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)
            if not batch_items and items:
                batch_items = list(items)
                random.Random(batch_seed).shuffle(batch_items)
                batch_items = batch_items[:batch_size]
            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=batch_seed,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )
        return batches

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        items = self._items("train", self.train_size, seed_offset=0)
        random.Random(seed).shuffle(items)
        items = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        canonical = "val" if split in {"valid_seen", "selection", "val"} else split
        if canonical == "train":
            count = self.train_size
            offset = 0
        elif canonical == "test":
            count = self.test_size
            offset = 20_000
        else:
            canonical = "val"
            count = self.val_size
            offset = 10_000
        if env_num and env_num > 0:
            count = min(count, int(env_num))
        items = self._items(canonical, count, seed_offset=offset)
        return BatchSpec(
            phase="eval",
            split=canonical,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )
