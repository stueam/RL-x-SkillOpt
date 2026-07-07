import os
from dotenv import load_dotenv

load_dotenv()  # 自动读取 .env 文件到 os.environ

# 可选：打印检查环境变量是否生效
print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
print(f"TINKER_API_KEY set: {bool(os.environ.get('TINKER_API_KEY'))}")

# 缩小实验规模后再运行
from examples.erdos_min_overlap.env import discover_erdos_min_overlap
discover_erdos_min_overlap()