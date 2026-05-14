from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sfmapi_hloc.backend import (
    HLOC_ACTIONS,
    HLOC_BENCHMARK_ACTIONS,
    HLOC_BENCHMARK_CLI_MODULES,
    HLOC_CLI_MODULES,
    HLOC_UPSTREAM_CLI_MODULES,
    HlocBackend,
)


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
    assert "hloc.convertModel" in action_ids
    assert "hloc.runPipeline" in action_ids
    assert "hloc.runModule" in action_ids
    assert "hloc.pipelineAachenV11LoFTR" not in action_ids
    assert "hloc.pipelineFourSeasonsLocalize" not in action_ids
    assert not ({action.action_id for action in HLOC_BENCHMARK_ACTIONS} & action_ids)
    assert (
        len([action for action in action_ids if action.startswith("hloc.")])
        == len(HLOC_ACTIONS) + 2
    )
    # The backend now implements a real portable surface alongside its
    # action catalog; capabilities() reports exactly the wrapped stages.
    assert backend.capabilities() == {
        "features.extract.superpoint",
        "features.extract.disk",
        "features.extract.aliked",
        "features.extract.r2d2",
        "features.extract.d2net",
        "features.extract.sift",
        "features.extract.sosnet",
        "pairs.retrieval",
        "pairs.from_poses",
        "matchers.superglue",
        "matchers.lightglue",
        "matchers.loftr",
        "triangulate.retri",
        "map.incremental",
        "localize.from_memory",
        "localize.batch",
    }
    extract = next(action for action in actions if action["action_id"] == "hloc.extractFeatures")
    assert extract["input_schema"]["properties"]["feature_conf"]["enum"]


def test_cli_module_allowlist_covers_hloc_command_surfaces() -> None:
    expected_api = {
        "hloc.extract_features",
        "hloc.match_features",
        "hloc.match_dense",
        "hloc.pairs_from_exhaustive",
        "hloc.pairs_from_retrieval",
        "hloc.pairs_from_covisibility",
        "hloc.pairs_from_poses",
        "hloc.reconstruction",
        "hloc.triangulation",
        "hloc.localize_sfm",
        "hloc.utils.read_write_model",
    }
    expected_benchmarks = {
        "hloc.localize_inloc",
        "hloc.colmap_from_nvm",
        "hloc.pipelines.Aachen.pipeline",
        "hloc.pipelines.Aachen_v1_1.pipeline",
        "hloc.pipelines.Aachen_v1_1.pipeline_loftr",
        "hloc.pipelines.RobotCar.pipeline",
        "hloc.pipelines.RobotCar.colmap_from_nvm",
        "hloc.pipelines.CMU.pipeline",
        "hloc.pipelines.Cambridge.pipeline",
        "hloc.pipelines.7Scenes.pipeline",
        "hloc.pipelines.7Scenes.create_gt_sfm",
        "hloc.pipelines.4Seasons.prepare_reference",
        "hloc.pipelines.4Seasons.localize",
    }

    assert expected_api <= set(HLOC_CLI_MODULES)
    assert expected_benchmarks <= set(HLOC_BENCHMARK_CLI_MODULES)
    assert not (set(HLOC_CLI_MODULES) & set(HLOC_BENCHMARK_CLI_MODULES))


def test_cli_module_allowlist_matches_checked_out_hloc_scripts() -> None:
    hloc_root = Path(__file__).resolve().parents[1] / "third_party" / "hloc" / "hloc"
    discovered: set[str] = set()
    for path in hloc_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "__main__" not in text and "argparse.ArgumentParser" not in text:
            continue
        module = "hloc." + ".".join(path.relative_to(hloc_root).with_suffix("").parts)
        discovered.add(module)

    assert discovered == set(HLOC_UPSTREAM_CLI_MODULES)


def test_backend_contract_passes(tmp_path: Path) -> None:
    pytest.importorskip("sfmapi.backends")
    from sfmapi.backends import (
        Backend,
        BatchLocalizationBackend,
        FeatureBackend,
        LocalizationBackend,
        MappingBackend,
        RefinementBackend,
        SfmBackend,
        assert_backend_contract,
    )

    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))
    assert isinstance(backend, Backend)
    # hloc now implements the portable feature/match/verify,
    # single-image + batch localization, and incremental-mapping
    # protocol layers as thin runner wrappers.
    assert isinstance(backend, FeatureBackend)
    assert isinstance(backend, LocalizationBackend)
    assert isinstance(backend, BatchLocalizationBackend)
    assert isinstance(backend, MappingBackend)
    # RefinementBackend wants bundle_adjustment / relocalize / pgo too,
    # so the structural check fails even though triangulate() is wired.
    assert not isinstance(backend, RefinementBackend)
    # ... and not the full SfM protocol (no observation / export / etc.).
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


def test_benchmark_actions_are_not_backend_api_actions(tmp_path: Path) -> None:
    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))

    with pytest.raises(Exception, match="not found"):
        backend.validate_backend_action(
            "hloc.pipelineSevenScenes",
            {"scenes": ["bad-scene"]},
        )


def test_run_module_rejects_benchmark_modules(tmp_path: Path) -> None:
    backend = HlocBackend(_fake_hloc(tmp_path / "Hierarchical-Localization"))

    result = backend.validate_backend_action(
        "hloc.runModule",
        {"module": "hloc.pipelines.Aachen_v1_1.pipeline_loftr"},
    )

    assert result["valid"] is False
    assert "module must be one of" in result["errors"][0]["message"]


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


def test_run_pipeline_reports_schema_safe_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _fake_hloc(tmp_path / "Hierarchical-Localization")
    backend = HlocBackend(root, python_executable="python")
    phases: list[str] = []

    class Progress:
        def phase_progress(self, phase: str, *, current: int, total: int) -> None:
            phases.append(phase)
            assert current <= total

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(args[-1]).write_text(json.dumps({}), encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend.run_backend_action(
        "hloc.runPipeline",
        {
            "image_dir": str(tmp_path / "images"),
            "outputs_dir": str(tmp_path / "outputs"),
            "feature_conf": "sift",
            "matcher_conf": "NN-ratio",
            "pairing_mode": "exhaustive",
            "run_reconstruction": True,
        },
        workspace=tmp_path / "workspace",
        progress=Progress(),
    )

    assert phases
    assert set(phases) == {"backend_action"}


def test_benchmark_runners_remain_direct_utility_surface() -> None:
    from sfmapi_hloc.runner import RUNNERS

    assert "hloc.pipelineAachenV11LoFTR" in RUNNERS
    assert "hloc.pipelineFourSeasonsLocalize" in RUNNERS
