# ============================================================
# Circle Packing × SkillOpt — 一键运行脚本
# ============================================================
# 确保在 SkillOpt-main 目录下运行
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Circle Packing Bench — SkillOpt" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 检查 .env 文件
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Write-Host "[OK] .env 文件已找到" -ForegroundColor Green
} else {
    Write-Host "[WARN] .env 文件不存在！请创建 .env 并填入 AZURE_OPENAI_API_KEY" -ForegroundColor Yellow
    Write-Host "       参考 configs/circlepacking/.env.template" -ForegroundColor Yellow
    exit 1
}

# 运行训练
Write-Host "[RUN] 启动 Circle Packing SkillOpt 训练..." -ForegroundColor Green
python scripts/train.py --config configs/circlepacking/default.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host "[DONE] 训练完成！" -ForegroundColor Green
} else {
    Write-Host "[FAIL] 训练出错，请检查日志" -ForegroundColor Red
}
