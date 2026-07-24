from __future__ import annotations
from skillopt.datasets.base import SplitDataLoader


class ReiDataLoader(SplitDataLoader):
    def get_task_types(self) -> list[str]:
        return ["rei"]
