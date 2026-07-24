"""Full Skill → RL: REI with best skill from SkillOpt injected."""
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from benchmarks.rei.rl_env import discover_rei
discover_rei("url_full", skill_variant="best")
