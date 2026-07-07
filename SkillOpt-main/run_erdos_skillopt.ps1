# ============================================================
# Erdos Minimum Overlap × SkillOpt — 一键运行脚本
# ============================================================
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Erdos Min Overlap Bench — SkillOpt" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 检查 .env 文件
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Write-Host "[OK] .env 文件已找到" -ForegroundColor Green
} else {
    Write-Host "[WARN] .env 文件不存在！请创建 .env 并填入 AZURE_OPENAI_API_KEY" -ForegroundColor Yellow
    Write-Host "       参考 configs/erdos_min_overlap/.env.template" -ForegroundColor Yellow
    exit 1
}

# 设置 UTF-8 模式 (Windows 修复 GBK 编码问题)
$env:PYTHONUTF8 = "1"
# 运行训练
Write-Host "[RUN] 启动 Erdos Min Overlap SkillOpt 训练..." -ForegroundColor Green
python scripts/train.py --config configs/erdos_min_overlap/default.yaml

if ($LASTEXITCODE -eq 0) {
    Write-Host "[DONE] 训练完成！" -ForegroundColor Green
} else {
    Write-Host "[FAIL] 训练出错，请检查日志" -ForegroundColor Red
}
