from __future__ import annotations

from pathlib import Path

import pytest

from sfmapi_hloc.backend import HlocBackend


def _fake_hloc(root: Path) -> Path:
    (root / "hloc").mkdir(parents=True, exist_ok=True)
    (root / "setup.py").write_text("setup(name='hloc')\n", encoding="utf-8")
    return root


def test_sfmapi_http_discovery_surfaces_hloc_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sfmapi.runtime import register_backend
    from sfmapi.testing import reset_runtime_for_tests_sync

    root = _fake_hloc(tmp_path / "Hierarchical-Localization")
    monkeypatch.setenv("SFMAPI_BACKEND", "hloc")
    monkeypatch.setenv("SFMAPI_MCP_MODE", "off")
    from sfmapi.runtime import create_app

    reset_runtime_for_tests_sync(
        ephemeral=True,
        db_url="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true",
        blob_backend="memory",
        queue_backend="inline",
        inline_tasks=True,
        workspace_root=tmp_path / "workspace",
    )
    register_backend("hloc", lambda: HlocBackend(root))

    with TestClient(create_app()) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["backend"]["name"] == "hloc"
        assert capabilities["features"]["backend.actions"] is True
        # hloc now also publishes portable stage capabilities plus the
        # backing config-schema and artifact-contract discovery surfaces.
        assert capabilities["features"]["features.extract.superpoint"] is True
        assert capabilities["features"]["features.extract.sift"] is True
        assert capabilities["features"]["matchers.superglue"] is True
        assert capabilities["features"]["pairs.from_poses"] is True
        assert capabilities["features"]["map.incremental"] is True
        assert capabilities["features"]["triangulate.retri"] is True
        assert capabilities["features"]["localize.batch"] is True
        assert capabilities["features"]["matches.verify"] is False
        assert capabilities["features"]["backend.config_schemas"] is True
        assert capabilities["features"]["backend.artifact_contracts"] is True

        backend = client.get("/v1/backend").json()
        assert backend["name"] == "hloc"
        assert backend["action_count"] > 0
        assert backend["config_schema_count"] == 3

        actions = client.get("/v1/backend/actions?include_schemas=true&page_size=50").json()[
            "items"
        ]
        action_ids = {action["action_id"] for action in actions}
        assert "hloc.runPipeline" in action_ids
        assert "hloc.extractFeatures" in action_ids
        pipeline = next(action for action in actions if action["action_id"] == "hloc.runPipeline")
        assert "pairing_mode" in pipeline["input_schema"]["properties"]

        config_schemas = client.get("/v1/backend/config-schemas").json()["items"]
        assert {row["config_id"] for row in config_schemas} == {
            "hloc.features",
            "hloc.matcher",
            "hloc.pairs.retrieval",
        }
        artifact_contracts = client.get("/v1/backend/artifact-contracts").json()["items"]
        assert {row["contract_id"] for row in artifact_contracts} == {
            "hloc.features",
            "hloc.matches",
            "hloc.pairs",
        }
