"""Circle packing dataloader — loads items from split dir.

Each item represents the same N=26 circle packing problem with a
different random seed, generating diverse training trajectories.
"""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class CirclePackingDataLoader(SplitDataLoader):
    """Circle packing dataloader with seed-based item diversity."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def get_task_types(self) -> list[str]:
        return ["circle_packing"]
