from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .backend import HLOC_ACTIONS, HLOC_BENCHMARK_ACTIONS


@dataclass(frozen=True)
class CheckResult:
    base_url: str
    elapsed_ms: float
    action_count: int
    missing_api_actions: list[str]
    exposed_benchmark_actions: list[str]

    @property
    def ok(self) -> bool:
        return not self.missing_api_actions and not self.exposed_benchmark_actions

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "base_url": self.base_url,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "action_count": self.action_count,
            "missing_api_actions": self.missing_api_actions,
            "exposed_benchmark_actions": self.exposed_benchmark_actions,
        }


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_api_surface(base_url: str, *, timeout: float = 30.0) -> CheckResult:
    base = base_url.rstrip("/") + "/"
    query = urlencode({"include_schemas": "true", "page_size": "500"})
    url = urljoin(base, f"v1/backend/actions?{query}")
    started = time.perf_counter()
    payload = _get_json(url, timeout=timeout)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    action_ids = {str(item["action_id"]) for item in payload.get("items", [])}
    expected = {action.action_id for action in HLOC_ACTIONS} | {
        "hloc.runModule",
        "hloc.runPipeline",
    }
    benchmark = {action.action_id for action in HLOC_BENCHMARK_ACTIONS}
    return CheckResult(
        base_url=base_url,
        elapsed_ms=elapsed_ms,
        action_count=len(action_ids),
        missing_api_actions=sorted(expected - action_ids),
        exposed_benchmark_actions=sorted(benchmark & action_ids),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check that a live sfmapi HLOC API exposes reusable actions only."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    result = check_api_surface(args.base_url, timeout=args.timeout)
    print(json.dumps(result.to_json(), indent=2, sort_keys=True))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
