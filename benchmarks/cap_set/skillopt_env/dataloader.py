from __future__ import annotations
from skillopt.datasets.base import SplitDataLoader


class CapSetDataLoader(SplitDataLoader):
    def get_task_types(self) -> list[str]:
        return ["cap_set"]
