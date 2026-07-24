import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
print(f"TINKER_API_KEY set: {bool(os.environ.get('TINKER_API_KEY'))}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

# 缩小实验规模后再运行
from examples.erdos_min_overlap.env import discover_erdos_min_overlap
discover_erdos_min_overlap()