import os
from dotenv import load_dotenv
load_dotenv()
# 国内镜像
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
print(f"TINKER_API_KEY set: {bool(os.environ.get('TINKER_API_KEY'))}")
print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

from examples.circle_packing.env import discover_circle_packing

discover_circle_packing("26")