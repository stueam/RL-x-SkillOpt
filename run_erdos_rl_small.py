"""Pure RL Small for Erdős Min Overlap — baseline for Skill→RL comparison.

Runs with the original prompt (no skill guardrails), small-scale config for fast comparison.

Usage:
    cd /mnt/c/Users/1/Desktop/TTT-Discover
    source venv/bin/activate
    python run_erdos_rl_small.py
"""

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["DISABLE_WANDB"] = "1"

print(f"TINKER_API_KEY: {'SET' if os.environ.get('TINKER_API_KEY') else 'MISSING'}")

from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
from ttt_discover import DiscoverConfig, discover

config = DiscoverConfig(
    env_type=ErdosMinOverlapEnv,
    problem_type="",
    num_cpus_per_task=1,
    eval_timeout=600,
    experiment_name="erdos-rl-small",
    wandb_project="",
    # Match Circle Packing Small config
    num_epochs=3,
    group_size=4,
    groups_per_batch=4,
    phase1_max_tokens=8000,
)

discover(config)
