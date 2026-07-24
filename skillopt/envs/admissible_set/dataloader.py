from __future__ import annotations
from skillopt.datasets.base import SplitDataLoader


class AdmissibleSetDataLoader(SplitDataLoader):
    def get_task_types(self) -> list[str]:
        return ["admissible_set"]
