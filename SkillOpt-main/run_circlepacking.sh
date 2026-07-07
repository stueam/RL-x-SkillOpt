#!/usr/bin/env bash
# Circle Packing × SkillOpt — 一键运行脚本 (Linux/Mac)
set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  Circle Packing Bench — SkillOpt"
echo "============================================================"

if [ -f .env ]; then
    echo "[OK] .env file found"
    set -a; source .env; set +a
else
    echo "[WARN] .env not found! Copy configs/circlepacking/.env.template"
    exit 1
fi

echo "[RUN] Starting Circle Packing SkillOpt training..."
python scripts/train.py --config configs/circlepacking/default.yaml
echo "[DONE] Training complete!"
