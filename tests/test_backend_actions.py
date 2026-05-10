from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sfmapi_hloc.backend import HLOC_ACTIONS, HlocBackend


def _fake_hloc(root: Path) -> Path:
    (root / "hloc").mkdir(parents=True, exist_ok=True)
    (root / "hloc" / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    (root / "setup.py").write_text("setup(name='hloc')\n", encoding="utf-8")
    return root


def test_action_catalog_exposes_hloc_actions(tmp_path: Path) -> None:
    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))

    actions = backend.list_backend_actions(include_schemas=True)
    action_ids = {action["action_id"] for action in actions}

    assert "hloc.extractFeatures" in action_ids
    assert "hloc.pairsRetrieval" in action_ids
    assert "hloc.matchDense" in action_ids
    assert "hloc.reconstruct" in action_ids
    assert "hloc.localizeSfm" in action_ids
    assert "hloc.runPipeline" in action_ids
    assert "hloc.runModule" in action_ids
    assert (
        len([action for action in action_ids if action.startswith("hloc.")])
        == len(HLOC_ACTIONS) + 2
    )
    assert backend.capabilities() == set()
    extract = next(action for action in actions if action["action_id"] == "hloc.extractFeatures")
    assert extract["input_schema"]["properties"]["feature_conf"]["enum"]


def test_backend_contract_passes(tmp_path: Path) -> None:
    pytest.importorskip("app.adapters.backend_contract")
    from app.adapters.backend import Backend, SfmBackend
    from app.adapters.backend_contract import assert_backend_contract

    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))
    assert isinstance(backend, Backend)
    assert not isinstance(backend, SfmBackend)
    assert_backend_contract(backend)


def test_validate_rejects_unknown_feature_conf(tmp_path: Path) -> None:
    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))

    result = backend.validate_backend_action(
        "hloc.extractFeatures",
        {"image_dir": "images", "outputs_dir": "outputs", "feature_conf": "unknown"},
    )

    assert result["valid"] is False
    assert "feature_conf must be one of" in result["errors"][0]["message"]


def test_validate_rejects_unknown_input(tmp_path: Path) -> None:
    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))

    result = backend.validate_backend_action(
        "hloc.matchFeatures",
        {
            "pairs_path": "pairs.txt",
            "feature_path": "features.h5",
            "matches_path": "matches.h5",
            "typo": True,
        },
    )

    assert result["valid"] is False
    assert "unknown input(s): typo" in result["errors"][0]["message"]


def test_run_extract_features_uses_structured_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_hloc(tmp_path / "Hierarchical-Localization")
    backend = HlocBackend(root, python_executable="python")
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        output_path = Path(args[-1])
        output_path.write_text(json.dumps({"feature_path": "features.h5"}), encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "hloc.extractFeatures",
        {
            "image_dir": "C:/data/images",
            "outputs_dir": "C:/data/outputs",
            "feature_conf": "superpoint_aachen",
            "as_half": True,
        },
        workspace=tmp_path / "workspace",
    )

    assert result["returncode"] == 0
    assert result["result"] == {"feature_path": "features.h5"}
    args = captured["args"]
    assert args[:3] == ["python", "-m", "sfmapi_hloc.runner"]
    assert args[3] == "hloc.extractFeatures"


def test_run_pipeline_executes_ordered_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_hloc(tmp_path / "Hierarchical-Localization")
    backend = HlocBackend(root, python_executable="python")
    actions: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        actions.append(args[3])
        Path(args[-1]).write_text(json.dumps({}), encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = backend.run_backend_action(
        "hloc.runPipeline",
        {
            "image_dir": str(tmp_path / "images"),
            "outputs_dir": str(tmp_path / "outputs"),
            "pairing_mode": "retrieval",
            "feature_conf": "superpoint_aachen",
            "retrieval_conf": "netvlad",
            "matcher_conf": "superglue",
            "run_reconstruction": True,
        },
        workspace=tmp_path / "workspace",
    )

    assert actions == [
        "hloc.extractFeatures",
        "hloc.pairsRetrieval",
        "hloc.extractFeatures",
        "hloc.matchFeatures",
        "hloc.reconstruct",
    ]
    assert len(result["steps"]) == 5
