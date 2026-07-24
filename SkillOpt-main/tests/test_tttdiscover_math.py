from __future__ import annotations

import json

from skillopt.engine.high_score_buffer import HighScoreBuffer
from skillopt.envs.tttdiscover_math.adapter import TTTDiscoverMathAdapter
from skillopt.envs.tttdiscover_math.rollout import _score


def test_erdos_score_rewards_lower_raw_values() -> None:
    worse = _score("erdos_min_overlap", True, 0.50, 2.636, 0.38092)
    better = _score("erdos_min_overlap", True, 0.38, 2.636, 0.38092)

    assert better > worse


def test_reflection_current_best_uses_minimum_raw_for_erdos() -> None:
    adapter = TTTDiscoverMathAdapter(
        problem_type="erdos_min_overlap",
        reflection_raw_success_threshold=0.42,
    )
    results = [
        {
            "id": "high-c5",
            "task_type": "erdos_min_overlap",
            "valid": True,
            "raw_score": 0.50,
            "normalized_score": 0.75,
            "hard": 0.75,
            "soft": 0.75,
        },
        {
            "id": "low-c5",
            "task_type": "erdos_min_overlap",
            "valid": True,
            "raw_score": 0.38,
            "normalized_score": 1.0,
            "hard": 1.0,
            "soft": 1.0,
        },
    ]

    prepared = adapter._prepare_reflection_results(results)
    by_id = {row["id"]: row for row in prepared}

    assert by_id["low-c5"]["reflection_is_cur_best"] is True
    assert by_id["low-c5"]["reflection_is_high_score"] is True
    assert by_id["high-c5"]["reflection_is_cur_best"] is False


def test_high_score_buffer_orders_by_normalized_reward_for_minimization(tmp_path) -> None:
    buffer_path = tmp_path / "buffer.json"
    buffer = HighScoreBuffer(max_size=2, path=str(buffer_path))
    buffer.update(
        [
            {
                "id": "worse-raw",
                "valid": True,
                "raw_score": 0.50,
                "normalized_score": 0.76,
                "hard": 0.76,
            },
            {
                "id": "better-raw",
                "valid": True,
                "raw_score": 0.38,
                "normalized_score": 1.00,
                "hard": 1.00,
            },
        ]
    )

    assert [row["id"] for row in buffer.get_top(2)] == ["better-raw", "worse-raw"]
    saved = json.loads(buffer_path.read_text(encoding="utf-8"))
    assert saved[0]["_buffer_score"] == 1.0
