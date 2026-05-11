from __future__ import annotations

from sfmapi_hloc import api_benchmark
from sfmapi_hloc.backend import HLOC_ACTIONS, HLOC_BENCHMARK_ACTIONS


def test_api_benchmark_rejects_exposed_benchmark_actions(monkeypatch) -> None:
    expected = [{"action_id": action.action_id} for action in HLOC_ACTIONS]
    exposed = [{"action_id": HLOC_BENCHMARK_ACTIONS[0].action_id}]

    monkeypatch.setattr(
        api_benchmark,
        "_get_json",
        lambda url, *, timeout: {
            "items": [
                *expected,
                {"action_id": "hloc.runModule"},
                {"action_id": "hloc.runPipeline"},
                *exposed,
            ]
        },
    )

    result = api_benchmark.check_api_surface("http://testserver")

    assert not result.ok
    assert result.exposed_benchmark_actions == [HLOC_BENCHMARK_ACTIONS[0].action_id]


def test_api_benchmark_accepts_reusable_api_surface(monkeypatch) -> None:
    expected = [{"action_id": action.action_id} for action in HLOC_ACTIONS]

    monkeypatch.setattr(
        api_benchmark,
        "_get_json",
        lambda url, *, timeout: {
            "items": [*expected, {"action_id": "hloc.runModule"}, {"action_id": "hloc.runPipeline"}]
        },
    )

    result = api_benchmark.check_api_surface("http://testserver")

    assert result.ok
    assert result.action_count == len(HLOC_ACTIONS) + 2
