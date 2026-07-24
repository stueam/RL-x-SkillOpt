from __future__ import annotations
from skillopt.envs.base import EnvAdapter
from .dataloader import ReiDataLoader
from .rollout import run_batch


class ReiAdapter(EnvAdapter):
    def __init__(
        self, split_dir="", data_path="", split_mode="split_dir",
        split_ratio="2:1:7", split_seed=42, split_output_dir="",
        exec_timeout=30, workers=4, analyst_workers=2,
        failure_only=False, minibatch_size=4, edit_budget=4,
        seed=42, limit=0, max_completion_tokens=4096,
    ) -> None:
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = ReiDataLoader(
            split_dir=split_dir, data_path=data_path, split_mode=split_mode,
            split_ratio=split_ratio, split_seed=split_seed,
            split_output_dir=split_output_dir, seed=seed, limit=limit,
        )

    def setup(self, cfg):
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch, **kw):
        return list(batch.payload or [])

    def build_train_env(self, batch_size, seed, **kw):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kw)
        return self.build_env_from_batch(batch, **kw)

    def build_eval_env(self, env_num, split, seed, **kw):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kw)
        return self.build_env_from_batch(batch, **kw)

    def rollout(self, env_manager, skill_content, out_dir, **kw):
        return run_batch(
            items=env_manager, out_root=out_dir, skill_content=skill_content,
            exec_timeout=self.exec_timeout, workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
            task_timeout=self.exec_timeout + 60,
        )

    def get_task_types(self):
        return self.dataloader.get_task_types()
