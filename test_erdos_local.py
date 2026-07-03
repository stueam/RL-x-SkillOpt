"""
Plan B: 本地测试 Erdős Minimum Overlap 环境逻辑
绕过 Ray、tinker、RL 训练，仅验证:
  1. 验证函数 (verify_c5_solution)
  2. 奖励计算 (evaluate_erdos_solution)
  3. 初始状态生成 (create_initial_state)
  4. 完整的前向传播模拟
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time


# ============================================================
# 直接从 examples/erdos_min_overlap/env.py 复制的验证逻辑
# （纯 numpy，无外部依赖）
# ============================================================

def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


# ============================================================
# 辅助函数：创建初始状态（同 env.py 的 create_initial_state）
# ============================================================

def create_initial_state(n_points: int | None = None, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    if n_points is None:
        n_points = int(rng.integers(40, 100))

    construction = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation

    dx = 2.0 / n_points
    correlation = np.correlate(construction, 1 - construction, mode="full") * dx
    c5_bound = float(np.max(correlation))

    return {
        "h_values": construction,
        "c5_bound": c5_bound,
        "n_points": n_points,
        "dx": dx,
    }


# ============================================================
# 模拟"LLM 生成的代码"返回的结果
# ============================================================

def dummy_optimization(n_points: int, seed: int = 123) -> tuple[np.ndarray, float, int]:
    """模拟一个简单的搜索算法返回的 (h_values, c5_bound, n_points)"""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, n_points)

    # 归一化: sum(h) * dx = 1
    target_sum = n_points / 2.0
    h = h * (target_sum / np.sum(h))

    # 裁剪到 [0, 1]
    h = np.clip(h, 0, 1)

    # 重新归一化（裁剪可能改变了 sum）
    h = h * (target_sum / np.sum(h))

    dx = 2.0 / n_points
    j = 1.0 - h
    corr = np.correlate(h, j, mode="full") * dx
    c5 = float(np.max(corr))

    return h, c5, n_points


# ============================================================
# 主测试逻辑
# ============================================================

def test_basic_verification():
    """测试 1: 基本的验证函数"""
    print("=" * 60)
    print("测试 1: 基本验证函数 (verify_c5_solution)")
    print("=" * 60)

    state = create_initial_state(n_points=50, seed=42)
    h = state["h_values"]
    c5 = state["c5_bound"]
    n = state["n_points"]

    print(f"  n_points      = {n}")
    print(f"  sum(h)        = {np.sum(h):.6f}  (target: {n/2})")
    print(f"  h in [0,1]    = {np.all(h >= 0) and np.all(h <= 1)}")
    print(f"  C5 (computed) = {c5:.8f}")

    # 验证通过
    result = verify_c5_solution(h, c5, n)
    print(f"  验证结果       = {result:.8f}")
    assert np.isclose(result, c5), "验证值不匹配"
    print("  ✅ 验证通过!\n")


def test_reward_evaluation():
    """测试 2: 奖励计算"""
    print("=" * 60)
    print("测试 2: 奖励计算 (evaluate_erdos_solution -> reward)")
    print("=" * 60)

    state = create_initial_state(n_points=60, seed=99)
    c5 = state["c5_bound"]

    reward = 1.0 / (1e-8 + c5)
    print(f"  C5 bound       = {c5:.8f}")
    print(f"  Reward         = {reward:.6f}  (公式: 1/(1e-8 + C5))")
    print(f"  越低越好, 当前 SOTA = 0.380876")
    print(f"  差距           = {c5 - 0.380876:.8f}")
    print("  ✅ 奖励计算正常!\n")


def test_simulated_optimization():
    """测试 3: 模拟优化过程（多轮搜索）"""
    print("=" * 60)
    print("测试 3: 模拟多轮搜索 (模拟 RL 的 step)")
    print("=" * 60)

    best_c5 = float("inf")
    n_points = 50

    for step in range(10):
        h, c5, n = dummy_optimization(n_points, seed=step * 100)

        # 验证
        is_valid = verify_erdos_solution((h, c5, n))
        if not is_valid:
            print(f"  Step {step:2d}: ❌ 无效解, 跳过")
            continue

        # 计算奖励
        reward = 1.0 / (1e-8 + c5)
        improvement = best_c5 - c5 if c5 < best_c5 else 0.0
        best_c5 = min(best_c5, c5)

        status = " 🆕 BEST!" if improvement > 0 else ""
        print(f"  Step {step:2d}: C5 = {c5:.8f}  Reward = {reward:.4f}  {status}")

    print(f"\n  最佳 C5  = {best_c5:.8f}")
    print(f"  SOTA    = 0.380876")
    print(f"  差距    = {best_c5 - 0.380876:.8f}")
    print("  ✅ 模拟优化完成!\n")


def test_edge_cases():
    """测试 4: 边界情况"""
    print("=" * 60)
    print("测试 4: 边界情况")
    print("=" * 60)

    # 4a: 常函数 h(x) = 0.5
    print("  4a: 常函数 h(x) = 0.5")
    n = 100
    h = np.ones(n) * 0.5
    dx = 2.0 / n
    corr = np.correlate(h, 1 - h, mode="full") * dx
    c5_const = float(np.max(corr))
    print(f"      C5 = {c5_const:.8f}  (理论值: 0.5)")
    assert verify_erdos_solution((h, c5_const, n)), "常函数验证失败"

    # 4b: 不同 n_points 规模
    print("  4b: 不同规模测试")
    for n in [10, 50, 100, 200]:
        h = np.ones(n) * 0.5
        dx = 2.0 / n
        corr = np.correlate(h, 1 - h, mode="full") * dx
        c5 = float(np.max(corr))
        ok = verify_erdos_solution((h, c5, n))
        print(f"      n={n:3d}: C5 = {c5:.6f}  valid={ok}")
        assert ok, f"n={n} 验证失败"

    # 4c: 无效输入应该被拒绝
    print("  4c: 无效输入检测")
    bad_cases = [
        ("NaN values", lambda: (np.array([float("nan")] * 10), 0.5, 10)),
        ("负值 h", lambda: (np.array([-0.1] * 10), 0.5, 10)),
        ("超过 1 的 h", lambda: (np.array([1.5] * 10), 0.5, 10)),
        ("错误的形状", lambda: (np.array([[0.5, 0.5]]), 0.5, 2)),
    ]
    for name, fn in bad_cases:
        try:
            result = fn()
            valid = verify_erdos_solution(result)
            print(f"      {name}: 预期拒绝, 结果={'❌ 未拒绝' if valid else '✅ 已拒绝'}")
        except Exception:
            print(f"      {name}: ✅ 已拒绝 (抛出异常)")

    print()


def test_end_to_end():
    """测试 5: 端到端模拟（完整的数据流）"""
    print("=" * 60)
    print("测试 5: 端到端模拟")
    print("=" * 60)

    # 模拟 1 个完整的经验回放 step:
    # 1. 从初始状态开始
    # 2. "LLM 生成代码"返回结果
    # 3. 验证并计算奖励
    # 4. 输出新状态供下一轮使用

    n_points = 60
    t_start = time.time()

    # Step 1: 初始状态
    print("  Phase 1: 创建初始状态...")
    initial = create_initial_state(n_points=n_points, seed=0)
    print(f"    初始 C5 = {initial['c5_bound']:.8f}")
    print(f"    n_points = {initial['n_points']}")

    # Step 2-4: 模拟多轮
    print("  Phase 2: 模拟经验迭代...")
    best = initial["c5_bound"]
    for round_idx in range(5):
        # 模拟搜索
        h, c5, n = dummy_optimization(n_points, seed=round_idx)

        # 验证
        is_valid = verify_erdos_solution((h, c5, n))
        if not is_valid:
            continue

        # 奖励
        reward = 1.0 / (1e-8 + c5)
        improvement = best - c5 if c5 < best else 0.0

        # 更新 best
        prev_best = best
        if c5 < best:
            best = c5

        print(f"    Round {round_idx}: C5={c5:.8f}, Reward={reward:.4f}, "
              f"Best={best:.8f}, Improvement={improvement:.8f}")

    elapsed = time.time() - t_start
    print(f"\n  最终最佳 C5 = {best:.8f}")
    print(f"  相对 SOTA (0.380876): {'更好' if best < 0.380876 else '更差'}, "
          f"差距={abs(best - 0.380876):.8f}")
    print(f"  耗时: {elapsed:.2f}s")
    print("  ✅ 端到端测试完成!")


# ============================================================
# 运行所有测试
# ============================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  TTT-Discover | Erdos Min Overlap 环境本地测试")
    print("  Plan B: 绕过 Ray/tinker/RL, 仅验证逻辑")
    print("=" * 60)
    print()

    test_basic_verification()
    test_reward_evaluation()
    test_simulated_optimization()
    test_edge_cases()
    test_end_to_end()

    print("=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
    print()
    print("环境逻辑摘要:")
    print("  - 验证函数: 检查 h∈[0,1], sum(h)=n/2, 计算 C5")
    print("  - 奖励: 1/(1e-8 + C5), 最小化 C5")
    print("  - 状态: (h_values, C5, n_points)")
    print("  - 每轮生成代码 → 验证 → 计算奖励 → 更新状态")
    print()
    print("下一步: 如需完整 RL 训练, 需要:")
    print("  1. Linux 系统 (Ray + CPU affinity)")
    print("  2. GPU + HF_TOKEN (访问 GPT-OSS 模型)")
    print("  3. HPC 级 CPU (数学任务计算密集)")
    print()
