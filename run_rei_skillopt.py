"""Convenience wrapper: run SkillOpt training for REI."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

os.chdir(os.path.join(os.path.dirname(__file__), "SkillOpt-main"))
os.system(f"{sys.executable} scripts/train.py --config ../benchmarks/rei/config/default.yaml")
