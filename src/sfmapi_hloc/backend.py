from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from sfmapi.errors import CapabilityUnavailableError, NotFoundError, ValidationError
except ModuleNotFoundError:  # pragma: no cover - allows adapter tests without sfmapi installed

    class CapabilityUnavailableError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, *, capability: str, reason: str = "") -> None:
            super().__init__(reason or capability)

    class NotFoundError(RuntimeError):  # type: ignore[no-redef]
        pass

    class ValidationError(RuntimeError):  # type: ignore[no-redef]
        pass


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HLOC_ROOT = REPO_ROOT / "third_party" / "hloc"

FEATURE_CONFIGS = (
    "superpoint_aachen",
    "superpoint_max",
    "superpoint_inloc",
    "r2d2",
    "d2net-ss",
    "sift",
    "sosnet",
    "disk",
    "aliked-n16",
    "dir",
    "netvlad",
    "openibl",
    "megaloc",
)
FEATURE_OUTPUTS = {
    "superpoint_aachen": "feats-superpoint-n4096-r1024",
    "superpoint_max": "feats-superpoint-n4096-rmax1600",
    "superpoint_inloc": "feats-superpoint-n4096-r1600",
    "r2d2": "feats-r2d2-n5000-r1024",
    "d2net-ss": "feats-d2net-ss",
    "sift": "feats-sift",
    "sosnet": "feats-sosnet",
    "disk": "feats-disk",
    "aliked-n16": "feats-aliked-n16",
    "dir": "global-feats-dir",
    "netvlad": "global-feats-netvlad",
    "openibl": "global-feats-openibl",
    "megaloc": "global-feats-megaloc",
}
RETRIEVAL_CONFIGS = ("dir", "netvlad", "openibl", "megaloc")
MATCHER_CONFIGS = (
    "superpoint+lightglue",
    "disk+lightglue",
    "aliked+lightglue",
    "superglue",
    "superglue-fast",
    "NN-superpoint",
    "NN-ratio",
    "NN-mutual",
    "adalam",
)
MATCHER_OUTPUTS = {
    "superpoint+lightglue": "matches-superpoint-lightglue",
    "disk+lightglue": "matches-disk-lightglue",
    "aliked+lightglue": "matches-aliked-lightglue",
    "superglue": "matches-superglue",
    "superglue-fast": "matches-superglue-it5",
    "NN-superpoint": "matches-NN-mutual-dist.7",
    "NN-ratio": "matches-NN-mutual-ratio.8",
    "NN-mutual": "matches-NN-mutual",
    "adalam": "matches-adalam",
}
DENSE_CONFIGS = ("loftr", "loftr_aachen", "loftr_superpoint")
PAIRING_MODES = ("exhaustive", "retrieval")


@dataclass(frozen=True)
class HlocAction:
    action_id: str
    display_name: str
    category: str
    description: str
    runner_action: str
    module: str | None = None
    gpu_required: bool = True


HLOC_ACTIONS: tuple[HlocAction, ...] = (
    HlocAction(
        "hloc.extractFeatures",
        "HLOC feature extraction",
        "features",
        "Extract local or global HLOC features into an HDF5 feature file.",
        "hloc.extractFeatures",
        "hloc.extract_features",
    ),
    HlocAction(
        "hloc.pairsExhaustive",
        "HLOC exhaustive pairs",
        "pairs",
        "Create exhaustive image pairs from an image list or feature file.",
        "hloc.pairsExhaustive",
        "hloc.pairs_from_exhaustive",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.pairsRetrieval",
        "HLOC retrieval pairs",
        "pairs",
        "Create image pairs from global retrieval descriptors.",
        "hloc.pairsRetrieval",
        "hloc.pairs_from_retrieval",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.pairsCovisibility",
        "HLOC covisibility pairs",
        "pairs",
        "Create database image pairs from an existing model's covisibility graph.",
        "hloc.pairsCovisibility",
        "hloc.pairs_from_covisibility",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.pairsPoses",
        "HLOC pose-neighbor pairs",
        "pairs",
        "Create image pairs from pose proximity in an existing model.",
        "hloc.pairsPoses",
        "hloc.pairs_from_poses",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.matchFeatures",
        "HLOC sparse matching",
        "matching",
        "Match sparse HLOC feature files for a pair list.",
        "hloc.matchFeatures",
        "hloc.match_features",
    ),
    HlocAction(
        "hloc.matchDense",
        "HLOC dense matching",
        "matching",
        "Run HLOC dense matching, such as LoFTR, for a pair list.",
        "hloc.matchDense",
        "hloc.match_dense",
    ),
    HlocAction(
        "hloc.reconstruct",
        "HLOC reconstruction",
        "mapping",
        "Import HLOC features/matches and run pycolmap reconstruction.",
        "hloc.reconstruct",
        "hloc.reconstruction",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.triangulate",
        "HLOC triangulation",
        "mapping",
        "Triangulate observations against an existing reference model.",
        "hloc.triangulate",
        "hloc.triangulation",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.localizeSfm",
        "HLOC SfM localization",
        "localization",
        "Localize query images against a reference SfM model.",
        "hloc.localizeSfm",
        "hloc.localize_sfm",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.convertModel",
        "HLOC COLMAP model conversion",
        "conversion",
        "Read a COLMAP binary or text model and optionally write it in another format.",
        "hloc.convertModel",
        "hloc.utils.read_write_model",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.listConfigs",
        "HLOC configuration catalog",
        "utility",
        "List HLOC feature extractor, sparse matcher, and dense matcher configurations.",
        "hloc.listConfigs",
        gpu_required=False,
    ),
)

HLOC_BENCHMARK_ACTIONS: tuple[HlocAction, ...] = (
    HlocAction(
        "hloc.localizeInLoc",
        "HLOC InLoc localization",
        "benchmark",
        "Run the dataset-specific HLOC InLoc localization workflow.",
        "hloc.localizeInLoc",
        "hloc.localize_inloc",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.colmapFromNvm",
        "HLOC NVM to COLMAP",
        "benchmark",
        "Convert benchmark NVM and intrinsics files into a COLMAP model/database.",
        "hloc.colmapFromNvm",
        "hloc.colmap_from_nvm",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.pipelineAachen",
        "HLOC Aachen pipeline",
        "benchmark",
        "Run the upstream Aachen Day-Night localization pipeline.",
        "hloc.pipelineAachen",
        "hloc.pipelines.Aachen.pipeline",
    ),
    HlocAction(
        "hloc.pipelineAachenV11",
        "HLOC Aachen v1.1 pipeline",
        "benchmark",
        "Run the upstream Aachen Day-Night v1.1 localization pipeline.",
        "hloc.pipelineAachenV11",
        "hloc.pipelines.Aachen_v1_1.pipeline",
    ),
    HlocAction(
        "hloc.pipelineAachenV11LoFTR",
        "HLOC Aachen v1.1 LoFTR pipeline",
        "benchmark",
        "Run the upstream Aachen v1.1 dense LoFTR localization pipeline.",
        "hloc.pipelineAachenV11LoFTR",
        "hloc.pipelines.Aachen_v1_1.pipeline_loftr",
    ),
    HlocAction(
        "hloc.pipelineRobotCar",
        "HLOC RobotCar pipeline",
        "benchmark",
        "Run the upstream RobotCar Seasons localization pipeline.",
        "hloc.pipelineRobotCar",
        "hloc.pipelines.RobotCar.pipeline",
    ),
    HlocAction(
        "hloc.pipelineRobotCarColmapFromNvm",
        "HLOC RobotCar NVM to COLMAP",
        "benchmark",
        "Convert RobotCar NVM and database files into a COLMAP model.",
        "hloc.pipelineRobotCarColmapFromNvm",
        "hloc.pipelines.RobotCar.colmap_from_nvm",
        gpu_required=False,
    ),
    HlocAction(
        "hloc.pipelineCMU",
        "HLOC CMU pipeline",
        "benchmark",
        "Run the upstream Extended CMU Seasons localization pipeline.",
        "hloc.pipelineCMU",
        "hloc.pipelines.CMU.pipeline",
    ),
    HlocAction(
        "hloc.pipelineCambridge",
        "HLOC Cambridge pipeline",
        "benchmark",
        "Run the upstream Cambridge Landmarks localization pipeline.",
        "hloc.pipelineCambridge",
        "hloc.pipelines.Cambridge.pipeline",
    ),
    HlocAction(
        "hloc.pipelineSevenScenes",
        "HLOC 7-Scenes pipeline",
        "benchmark",
        "Run the upstream 7-Scenes localization pipeline.",
        "hloc.pipelineSevenScenes",
        "hloc.pipelines.7Scenes.pipeline",
    ),
    HlocAction(
        "hloc.pipelineSevenScenesCorrectDepth",
        "HLOC 7-Scenes depth correction",
        "benchmark",
        "Correct 7-Scenes SfM models with ground-truth depth.",
        "hloc.pipelineSevenScenesCorrectDepth",
        "hloc.pipelines.7Scenes.create_gt_sfm",
    ),
    HlocAction(
        "hloc.pipelineFourSeasonsPrepareReference",
        "HLOC 4Seasons reference preparation",
        "benchmark",
        "Prepare the upstream 4Seasons reference SfM model.",
        "hloc.pipelineFourSeasonsPrepareReference",
        "hloc.pipelines.4Seasons.prepare_reference",
    ),
    HlocAction(
        "hloc.pipelineFourSeasonsLocalize",
        "HLOC 4Seasons localization",
        "benchmark",
        "Run the upstream 4Seasons sequence localization workflow.",
        "hloc.pipelineFourSeasonsLocalize",
        "hloc.pipelines.4Seasons.localize",
    ),
)
_ACTION_BY_ID = {action.action_id: action for action in HLOC_ACTIONS}
HLOC_CLI_MODULES = tuple(sorted({action.module for action in HLOC_ACTIONS if action.module}))
HLOC_BENCHMARK_CLI_MODULES = tuple(
    sorted({action.module for action in HLOC_BENCHMARK_ACTIONS if action.module})
)
HLOC_UPSTREAM_CLI_MODULES = tuple(sorted({*HLOC_CLI_MODULES, *HLOC_BENCHMARK_CLI_MODULES}))
_SCRIPT_MODULES = set(HLOC_CLI_MODULES)


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser()


def resolve_hloc_root(value: str | Path | None) -> Path | None:
    raw = value or os.environ.get("SFMAPI_HLOC_ROOT")
    path = _expand_path(raw) if raw else DEFAULT_HLOC_ROOT
    if (path / "setup.py").exists() and (path / "hloc").is_dir():
        return path.resolve()
    return None


def configure_hloc_environment(
    root: str | Path | None = None,
    *,
    python_executable: str | Path | None = None,
    validate: bool = False,
) -> Path | None:
    resolved_root = resolve_hloc_root(root)
    if resolved_root is None:
        if validate:
            raise ValueError(
                "HLOC checkout not found. Set SFMAPI_HLOC_ROOT or pass "
                "--hloc-root to sfmapi-hloc-api."
            )
        return None

    os.environ["SFMAPI_HLOC_ROOT"] = str(resolved_root)
    python = Path(python_executable or os.environ.get("SFMAPI_HLOC_PYTHON") or sys.executable)
    os.environ["SFMAPI_HLOC_PYTHON"] = str(python)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if str(resolved_root) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(resolved_root), *parts])
    return resolved_root


class HlocBackend:
    name = "hloc"
    version = "0.0.1"
    vendor = "Hierarchical Localization"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        python_executable: str | Path | None = None,
    ) -> None:
        self._root_override = _expand_path(root).resolve() if root else None
        self._python_executable = Path(
            python_executable or os.environ.get("SFMAPI_HLOC_PYTHON") or sys.executable
        )

    def capabilities(self) -> set[str]:
        return set()

    def runtime_versions(self) -> dict[str, str]:
        root = self._find_root()
        versions = {
            "backend": self.version,
            "hloc_root": str(root) if root else "missing",
            "hloc_python": str(self._python_executable),
        }
        if root is not None:
            commit = self._git_revision(root)
            if commit:
                versions["hloc_commit"] = commit
            version = self._read_hloc_version(root)
            if version:
                versions["hloc"] = version
        return versions

    def list_backend_actions(self, *, include_schemas: bool = False) -> list[dict[str, Any]]:
        actions = [self._pipeline_action(include_schemas=include_schemas)]
        actions.extend(
            self._action_descriptor(action, include_schemas=include_schemas)
            for action in HLOC_ACTIONS
        )
        actions.append(self._module_action(include_schemas=include_schemas))
        return sorted(actions, key=lambda action: str(action["action_id"]))

    def get_backend_action(self, action_id: str) -> dict[str, Any]:
        for action in self.list_backend_actions(include_schemas=True):
            if action["action_id"] == action_id:
                return action
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def validate_backend_action(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        except ValidationError as exc:
            return {
                "action_id": action_id,
                "valid": False,
                "errors": [{"field": None, "message": str(exc)}],
                "normalized_inputs": {},
            }
        return {
            "action_id": action_id,
            "valid": True,
            "errors": [],
            "normalized_inputs": normalized,
        }

    def run_backend_action(
        self,
        action_id: str,
        inputs: dict[str, Any],
        *,
        workspace: Path | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_action_inputs(action_id, dict(inputs or {}))
        if action_id == "hloc.runPipeline":
            return self._run_pipeline(normalized, workspace=workspace, progress=progress)
        if action_id == "hloc.runModule":
            return self._run_module_action(normalized)
        action = _ACTION_BY_ID.get(action_id)
        if action is None:
            raise NotFoundError(f"Backend action {action_id!r} not found")
        return self._run_runner_action(action.runner_action, normalized, workspace=workspace)

    def _find_root(self) -> Path | None:
        if self._root_override is not None:
            if (self._root_override / "setup.py").exists() and (
                self._root_override / "hloc"
            ).is_dir():
                return self._root_override
            return None
        return resolve_hloc_root(None)

    def _require_root(self) -> Path:
        root = self._find_root()
        if root is None:
            raise CapabilityUnavailableError(
                capability="backend.actions",
                reason=(
                    "HLOC checkout not found. Run `git submodule update --init "
                    "--recursive` and set SFMAPI_HLOC_ROOT if needed."
                ),
            )
        return root

    def _git_revision(self, root: Path) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except OSError:
            return None
        value = completed.stdout.strip()
        return value or None

    def _read_hloc_version(self, root: Path) -> str | None:
        init_file = root / "hloc" / "__init__.py"
        if not init_file.exists():
            return None
        for line in init_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("'\"")
        return None

    def _run_pipeline(
        self,
        inputs: dict[str, Any],
        *,
        workspace: Path | None,
        progress: Any | None,
    ) -> dict[str, Any]:
        outputs_dir = Path(str(inputs["outputs_dir"]))
        feature_conf = str(inputs.get("feature_conf", "superpoint_aachen"))
        retrieval_conf = str(inputs.get("retrieval_conf", "netvlad"))
        matcher_conf = str(inputs.get("matcher_conf", "superglue"))
        pairing_mode = str(inputs.get("pairing_mode", "exhaustive"))

        feature_path = Path(
            str(inputs.get("feature_path") or outputs_dir / f"{FEATURE_OUTPUTS[feature_conf]}.h5")
        )
        retrieval_path = Path(
            str(
                inputs.get("retrieval_path")
                or outputs_dir / f"{FEATURE_OUTPUTS[retrieval_conf]}.h5"
            )
        )
        pairs_path = Path(
            str(inputs.get("pairs_path") or outputs_dir / f"pairs-{pairing_mode}.txt")
        )
        matches_path = Path(
            str(
                inputs.get("matches_path")
                or outputs_dir
                / f"{feature_path.stem}_{MATCHER_OUTPUTS[matcher_conf]}_{pairs_path.stem}.h5"
            )
        )
        sfm_dir = Path(str(inputs.get("sfm_dir") or outputs_dir / "sfm"))

        steps: list[tuple[str, dict[str, Any]]] = []
        if pairing_mode == "retrieval":
            steps.append(
                (
                    "hloc.extractFeatures",
                    {
                        "image_dir": inputs["image_dir"],
                        "outputs_dir": str(outputs_dir),
                        "feature_conf": retrieval_conf,
                        "feature_path": str(retrieval_path),
                        "as_half": inputs.get("as_half", True),
                        "image_list": inputs.get("image_list"),
                        "overwrite": inputs.get("overwrite", False),
                        "timeout_seconds": inputs.get("timeout_seconds"),
                    },
                )
            )
            steps.append(
                (
                    "hloc.pairsRetrieval",
                    {
                        "descriptors_path": str(retrieval_path),
                        "pairs_path": str(pairs_path),
                        "num_matched": inputs.get("num_matched", 20),
                        "timeout_seconds": inputs.get("timeout_seconds"),
                    },
                )
            )

        steps.append(
            (
                "hloc.extractFeatures",
                {
                    "image_dir": inputs["image_dir"],
                    "outputs_dir": str(outputs_dir),
                    "feature_conf": feature_conf,
                    "feature_path": str(feature_path),
                    "as_half": inputs.get("as_half", True),
                    "image_list": inputs.get("image_list"),
                    "conf_override": inputs.get("feature_conf_override"),
                    "overwrite": inputs.get("overwrite", False),
                    "timeout_seconds": inputs.get("timeout_seconds"),
                },
            )
        )
        if pairing_mode == "exhaustive":
            steps.append(
                (
                    "hloc.pairsExhaustive",
                    {
                        "pairs_path": str(pairs_path),
                        "features_path": str(feature_path),
                        "timeout_seconds": inputs.get("timeout_seconds"),
                    },
                )
            )
        steps.append(
            (
                "hloc.matchFeatures",
                {
                    "pairs_path": str(pairs_path),
                    "feature_path": str(feature_path),
                    "matches_path": str(matches_path),
                    "matcher_conf": matcher_conf,
                    "conf_override": inputs.get("matcher_conf_override"),
                    "overwrite": inputs.get("overwrite", False),
                    "timeout_seconds": inputs.get("timeout_seconds"),
                },
            )
        )
        if bool(inputs.get("run_reconstruction", True)):
            steps.append(
                (
                    "hloc.reconstruct",
                    {
                        "sfm_dir": str(sfm_dir),
                        "image_dir": inputs["image_dir"],
                        "pairs_path": str(pairs_path),
                        "feature_path": str(feature_path),
                        "matches_path": str(matches_path),
                        "camera_mode": inputs.get("camera_mode", "AUTO"),
                        "image_list": inputs.get("image_list"),
                        "image_options": inputs.get("image_options"),
                        "mapper_options": inputs.get("mapper_options"),
                        "skip_geometric_verification": inputs.get(
                            "skip_geometric_verification", False
                        ),
                        "min_match_score": inputs.get("min_match_score"),
                        "verbose": inputs.get("verbose", False),
                        "timeout_seconds": inputs.get("timeout_seconds"),
                    },
                )
            )

        results: list[dict[str, Any]] = []
        total = len(steps)
        for index, (step_action, step_inputs) in enumerate(steps, start=1):
            self._progress(progress, step_action, index - 1, total)
            results.append(self._run_runner_action(step_action, step_inputs, workspace=workspace))
            self._progress(progress, step_action, index, total)
        return {
            "steps": results,
            "image_dir": str(inputs["image_dir"]),
            "outputs_dir": str(outputs_dir),
            "pairs_path": str(pairs_path),
            "feature_path": str(feature_path),
            "matches_path": str(matches_path),
            "sfm_dir": str(sfm_dir),
        }

    def _run_runner_action(
        self,
        action_id: str,
        inputs: dict[str, Any],
        *,
        workspace: Path | None,
    ) -> dict[str, Any]:
        root = self._require_root()
        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if workspace is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="sfmapi-hloc-")
            workdir = Path(temp_dir.name)
        else:
            workdir = Path(workspace)
            workdir.mkdir(parents=True, exist_ok=True)
        try:
            input_path = workdir / f"{action_id.replace('.', '_')}_input.json"
            output_path = workdir / f"{action_id.replace('.', '_')}_output.json"
            input_path.write_text(json.dumps(inputs, indent=2, sort_keys=True), encoding="utf-8")
            completed = self._run_python_module(
                "sfmapi_hloc.runner",
                [action_id, str(input_path), str(output_path)],
                cwd=root,
                timeout_seconds=inputs.get("timeout_seconds"),
            )
            result = None
            if output_path.exists():
                result = json.loads(output_path.read_text(encoding="utf-8"))
            return {
                "action_id": action_id,
                "args": [
                    str(self._python_executable),
                    "-m",
                    "sfmapi_hloc.runner",
                    action_id,
                    str(input_path),
                    str(output_path),
                ],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": result,
            }
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _run_module_action(self, inputs: dict[str, Any]) -> dict[str, Any]:
        module = str(inputs["module"])
        args = [str(arg) for arg in inputs.get("args", [])]
        completed = self._run_python_module(
            module,
            args,
            cwd=Path(str(inputs["cwd"])) if inputs.get("cwd") else None,
            extra_env={str(k): str(v) for k, v in dict(inputs.get("env") or {}).items()},
            timeout_seconds=inputs.get("timeout_seconds"),
        )
        return {
            "module": module,
            "args": [str(self._python_executable), "-m", module, *args],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def _run_python_module(
        self,
        module: str,
        args: list[str],
        *,
        cwd: Path | None = None,
        extra_env: dict[str, str] | None = None,
        timeout_seconds: int | float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        root = self._require_root()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), *[part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]]
        )
        if extra_env:
            env.update(extra_env)
        try:
            return subprocess.run(
                [str(self._python_executable), "-m", module, *args],
                check=True,
                capture_output=True,
                text=True,
                cwd=str(cwd or root),
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ValidationError(f"HLOC command failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(f"HLOC command timed out after {timeout_seconds}s") from exc

    def _pipeline_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "hloc.runPipeline",
            "backend": self.name,
            "display_name": "HLOC feature, matching, and reconstruction pipeline",
            "description": (
                "Run feature extraction, pair generation, matching, and optional "
                "pycolmap reconstruction."
            ),
            "category": "pipeline",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": True,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {
                "family": "hloc",
                "upstream_root": str(self._find_root() or DEFAULT_HLOC_ROOT),
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._pipeline_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _action_descriptor(self, action: HlocAction, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": action.action_id,
            "backend": self.name,
            "display_name": action.display_name,
            "description": action.description,
            "category": action.category,
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": action.gpu_required,
            "required_capabilities": [],
            "metadata": {
                "family": "hloc",
                "module": action.module,
                "upstream_root": str(self._find_root() or DEFAULT_HLOC_ROOT),
            },
        }
        if include_schemas:
            descriptor["input_schema"] = self._input_schema_for_action(action.action_id)
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _module_action(self, *, include_schemas: bool) -> dict[str, Any]:
        descriptor = {
            "action_id": "hloc.runModule",
            "backend": self.name,
            "display_name": "HLOC Python module",
            "description": "Run an allow-listed HLOC Python module with explicit args.",
            "category": "utility",
            "stability": "backend_extension",
            "side_effects": "write",
            "long_running": True,
            "supports_progress": False,
            "idempotent": False,
            "gpu_required": True,
            "required_capabilities": [],
            "metadata": {"family": "hloc", "allowlist": sorted(_SCRIPT_MODULES)},
        }
        if include_schemas:
            descriptor["input_schema"] = self._module_input_schema()
            descriptor["output_schema"] = self._run_output_schema()
        return descriptor

    def _input_schema_for_action(self, action_id: str) -> dict[str, Any]:
        common = self._common_properties()
        if action_id == "hloc.extractFeatures":
            return self._schema(
                {
                    **common,
                    "image_dir": {"type": "string"},
                    "outputs_dir": {"type": "string"},
                    "feature_conf": {"type": "string", "enum": list(FEATURE_CONFIGS)},
                    "feature_path": {"type": "string"},
                    "as_half": {"type": "boolean", "default": True},
                    "image_list": {"type": ["string", "array"], "items": {"type": "string"}},
                    "conf_override": {"type": "object"},
                },
                required=["image_dir", "outputs_dir"],
            )
        if action_id == "hloc.pairsExhaustive":
            return self._schema(
                {
                    **common,
                    "pairs_path": {"type": "string"},
                    "image_list": {"type": ["string", "array"], "items": {"type": "string"}},
                    "features_path": {"type": "string"},
                    "ref_list": {"type": ["string", "array"], "items": {"type": "string"}},
                    "ref_features_path": {"type": "string"},
                },
                required=["pairs_path"],
            )
        if action_id == "hloc.pairsRetrieval":
            return self._schema(
                {
                    **common,
                    "descriptors_path": {"type": "string"},
                    "pairs_path": {"type": "string"},
                    "num_matched": {"type": "integer", "minimum": 1, "default": 20},
                    "query_prefix": {"type": ["string", "array"], "items": {"type": "string"}},
                    "query_list": {"type": "string"},
                    "db_prefix": {"type": ["string", "array"], "items": {"type": "string"}},
                    "db_list": {"type": "string"},
                    "db_model": {"type": "string"},
                    "db_descriptors": {"type": ["string", "array"], "items": {"type": "string"}},
                },
                required=["descriptors_path", "pairs_path", "num_matched"],
            )
        if action_id in {"hloc.pairsCovisibility", "hloc.pairsPoses"}:
            properties = {
                **common,
                "model_path": {"type": "string"},
                "pairs_path": {"type": "string"},
                "num_matched": {"type": "integer", "minimum": 1},
            }
            if action_id == "hloc.pairsPoses":
                properties["rotation_threshold"] = {"type": "number", "default": 30.0}
            return self._schema(properties, required=["model_path", "pairs_path", "num_matched"])
        if action_id == "hloc.matchFeatures":
            return self._schema(
                {
                    **common,
                    "pairs_path": {"type": "string"},
                    "feature_path": {"type": "string"},
                    "matches_path": {"type": "string"},
                    "matcher_conf": {"type": "string", "enum": list(MATCHER_CONFIGS)},
                    "export_dir": {"type": "string"},
                    "features_ref_path": {"type": "string"},
                    "conf_override": {"type": "object"},
                },
                required=["pairs_path", "feature_path", "matches_path"],
            )
        if action_id == "hloc.matchDense":
            return self._schema(
                {
                    **common,
                    "pairs_path": {"type": "string"},
                    "image_dir": {"type": "string"},
                    "outputs_dir": {"type": "string"},
                    "dense_conf": {"type": "string", "enum": list(DENSE_CONFIGS)},
                    "matches_path": {"type": "string"},
                    "features_path": {"type": "string"},
                    "features_ref_path": {"type": ["string", "array"], "items": {"type": "string"}},
                    "max_kps": {"type": ["integer", "null"], "minimum": 1},
                    "conf_override": {"type": "object"},
                },
                required=["pairs_path", "image_dir", "outputs_dir"],
            )
        if action_id == "hloc.reconstruct":
            return self._schema(
                {
                    **common,
                    "sfm_dir": {"type": "string"},
                    "image_dir": {"type": "string"},
                    "pairs_path": {"type": "string"},
                    "feature_path": {"type": "string"},
                    "matches_path": {"type": "string"},
                    "camera_mode": {"type": "string", "default": "AUTO"},
                    "image_list": {"type": ["string", "array"], "items": {"type": "string"}},
                    "image_options": {"type": "object"},
                    "mapper_options": {"type": "object"},
                    "skip_geometric_verification": {"type": "boolean", "default": False},
                    "min_match_score": {"type": ["number", "null"]},
                    "verbose": {"type": "boolean", "default": False},
                },
                required=["sfm_dir", "image_dir", "pairs_path", "feature_path", "matches_path"],
            )
        if action_id == "hloc.triangulate":
            return self._schema(
                {
                    **common,
                    "sfm_dir": {"type": "string"},
                    "reference_sfm_model": {"type": "string"},
                    "image_dir": {"type": "string"},
                    "pairs_path": {"type": "string"},
                    "feature_path": {"type": "string"},
                    "matches_path": {"type": "string"},
                    "skip_geometric_verification": {"type": "boolean", "default": False},
                    "estimate_two_view_geometries": {"type": "boolean", "default": False},
                    "min_match_score": {"type": ["number", "null"]},
                    "mapper_options": {"type": "object"},
                    "verbose": {"type": "boolean", "default": False},
                },
                required=[
                    "sfm_dir",
                    "reference_sfm_model",
                    "image_dir",
                    "pairs_path",
                    "feature_path",
                    "matches_path",
                ],
            )
        if action_id == "hloc.localizeSfm":
            return self._schema(
                {
                    **common,
                    "reference_sfm": {"type": "string"},
                    "queries_path": {"type": "string"},
                    "retrieval_path": {"type": "string"},
                    "feature_path": {"type": "string"},
                    "matches_path": {"type": "string"},
                    "results_path": {"type": "string"},
                    "ransac_thresh": {"type": "number", "default": 12.0},
                    "covisibility_clustering": {"type": "boolean", "default": False},
                    "prepend_camera_name": {"type": "boolean", "default": False},
                    "config": {"type": "object"},
                },
                required=[
                    "reference_sfm",
                    "queries_path",
                    "retrieval_path",
                    "feature_path",
                    "matches_path",
                    "results_path",
                ],
            )
        if action_id == "hloc.localizeInLoc":
            return self._schema(
                {
                    **common,
                    "dataset_dir": {"type": "string"},
                    "retrieval_path": {"type": "string"},
                    "feature_path": {"type": "string"},
                    "matches_path": {"type": "string"},
                    "results_path": {"type": "string"},
                    "skip_matches": {"type": ["integer", "null"], "minimum": 1},
                },
                required=[
                    "dataset_dir",
                    "retrieval_path",
                    "feature_path",
                    "matches_path",
                    "results_path",
                ],
            )
        if action_id == "hloc.colmapFromNvm":
            return self._schema(
                {
                    **common,
                    "nvm_path": {"type": "string"},
                    "intrinsics_path": {"type": "string"},
                    "database_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "skip_points": {"type": "boolean", "default": False},
                },
                required=["nvm_path", "intrinsics_path", "database_path", "output_path"],
            )
        if action_id == "hloc.convertModel":
            return self._schema(
                {
                    **common,
                    "input_model": {"type": "string"},
                    "input_format": {
                        "type": "string",
                        "enum": ["", ".bin", ".txt"],
                        "default": "",
                    },
                    "output_model": {"type": "string"},
                    "output_format": {
                        "type": "string",
                        "enum": [".bin", ".txt"],
                        "default": ".txt",
                    },
                },
                required=["input_model"],
            )
        if action_id == "hloc.listConfigs":
            return self._schema(
                {
                    **common,
                    "include_values": {"type": "boolean", "default": False},
                },
                required=[],
            )
        if action_id in {
            "hloc.pipelineAachen",
            "hloc.pipelineAachenV11",
            "hloc.pipelineAachenV11LoFTR",
            "hloc.pipelineRobotCar",
        }:
            return self._schema(
                {
                    **common,
                    "dataset": {"type": "string"},
                    "outputs": {"type": "string"},
                    "num_covis": {"type": "integer", "minimum": 1, "default": 20},
                    "num_loc": {"type": "integer", "minimum": 1, "default": 50},
                },
                required=[],
            )
        if action_id == "hloc.pipelineRobotCarColmapFromNvm":
            return self._schema(
                {
                    **common,
                    "nvm_path": {"type": "string"},
                    "database_path": {"type": "string"},
                    "output_path": {"type": "string"},
                    "skip_points": {"type": "boolean", "default": False},
                },
                required=["nvm_path", "database_path", "output_path"],
            )
        if action_id == "hloc.pipelineCMU":
            return self._schema(
                {
                    **common,
                    "slices": {
                        "type": "string",
                        "default": "*",
                        "description": "One slice, an inclusive range like 2-6, or a Python list.",
                    },
                    "dataset": {"type": "string"},
                    "outputs": {"type": "string"},
                    "num_covis": {"type": "integer", "minimum": 1, "default": 20},
                    "num_loc": {"type": "integer", "minimum": 1, "default": 10},
                },
                required=[],
            )
        if action_id == "hloc.pipelineCambridge":
            return self._scene_pipeline_schema(
                ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch", "GreatCourt"],
                include_num_loc=True,
            )
        if action_id == "hloc.pipelineSevenScenes":
            schema = self._scene_pipeline_schema(
                ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
                include_num_loc=False,
            )
            schema["properties"]["use_dense_depth"] = {"type": "boolean", "default": False}
            return schema
        if action_id == "hloc.pipelineSevenScenesCorrectDepth":
            return self._schema(
                {
                    **common,
                    "scenes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "chess",
                                "fire",
                                "heads",
                                "office",
                                "pumpkin",
                                "redkitchen",
                                "stairs",
                            ],
                        },
                    },
                    "dataset": {"type": "string"},
                    "outputs": {"type": "string"},
                },
                required=[],
            )
        if action_id == "hloc.pipelineFourSeasonsPrepareReference":
            return self._schema(
                {
                    **common,
                    "dataset": {"type": "string"},
                    "outputs": {"type": "string"},
                },
                required=[],
            )
        if action_id == "hloc.pipelineFourSeasonsLocalize":
            return self._schema(
                {
                    **common,
                    "sequence": {
                        "type": "string",
                        "enum": ["training", "validation", "test0", "test1"],
                    },
                    "dataset": {"type": "string"},
                    "outputs": {"type": "string"},
                },
                required=["sequence"],
            )
        raise NotFoundError(f"Backend action {action_id!r} not found")

    def _pipeline_input_schema(self) -> dict[str, Any]:
        return self._schema(
            {
                **self._common_properties(),
                "image_dir": {"type": "string"},
                "outputs_dir": {"type": "string"},
                "pairing_mode": {
                    "type": "string",
                    "enum": list(PAIRING_MODES),
                    "default": "exhaustive",
                },
                "feature_conf": {
                    "type": "string",
                    "enum": list(FEATURE_CONFIGS),
                    "default": "superpoint_aachen",
                },
                "feature_conf_override": {"type": "object"},
                "retrieval_conf": {
                    "type": "string",
                    "enum": list(RETRIEVAL_CONFIGS),
                    "default": "netvlad",
                },
                "matcher_conf": {
                    "type": "string",
                    "enum": list(MATCHER_CONFIGS),
                    "default": "superglue",
                },
                "matcher_conf_override": {"type": "object"},
                "num_matched": {"type": "integer", "minimum": 1, "default": 20},
                "feature_path": {"type": "string"},
                "retrieval_path": {"type": "string"},
                "pairs_path": {"type": "string"},
                "matches_path": {"type": "string"},
                "sfm_dir": {"type": "string"},
                "image_list": {"type": ["string", "array"], "items": {"type": "string"}},
                "as_half": {"type": "boolean", "default": True},
                "run_reconstruction": {"type": "boolean", "default": True},
                "camera_mode": {"type": "string", "default": "AUTO"},
                "image_options": {"type": "object"},
                "mapper_options": {"type": "object"},
                "skip_geometric_verification": {"type": "boolean", "default": False},
                "min_match_score": {"type": ["number", "null"]},
                "verbose": {"type": "boolean", "default": False},
            },
            required=["image_dir", "outputs_dir"],
        )

    def _common_properties(self) -> dict[str, Any]:
        return {
            "timeout_seconds": {"type": "number"},
            "overwrite": {"type": "boolean", "default": False},
        }

    def _scene_pipeline_schema(
        self,
        scenes: list[str],
        *,
        include_num_loc: bool,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {
            **self._common_properties(),
            "scenes": {
                "type": "array",
                "items": {"type": "string", "enum": scenes},
                "default": scenes,
            },
            "dataset": {"type": "string"},
            "outputs": {"type": "string"},
            "num_covis": {"type": "integer", "minimum": 1, "default": 20},
        }
        if include_num_loc:
            properties["num_loc"] = {"type": "integer", "minimum": 1, "default": 10}
        return self._schema(properties, required=[])

    def _module_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["module"],
            "properties": {
                "module": {"type": "string", "enum": sorted(_SCRIPT_MODULES)},
                "args": {"type": "array", "items": {"type": "string"}},
                "cwd": {"type": "string"},
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "number", "boolean"]},
                },
                "timeout_seconds": {"type": "number"},
            },
        }

    def _schema(self, properties: dict[str, Any], *, required: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    def _run_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "returncode": {"type": "integer"},
                "args": {"type": "array", "items": {"type": "string"}},
                "stdout": {"type": "string"},
                "stderr": {"type": "string"},
                "result": {"type": ["object", "null"]},
            },
        }

    def _normalize_action_inputs(self, action_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        descriptor = self.get_backend_action(action_id)
        if action_id == "hloc.runModule":
            module = str(inputs.get("module") or "")
            if module not in _SCRIPT_MODULES:
                raise ValidationError(
                    f"module must be one of: {', '.join(sorted(_SCRIPT_MODULES))}"
                )
            args = inputs.get("args", [])
            if args is None:
                args = []
            if not isinstance(args, list):
                raise ValidationError("args must be an array of strings")
            inputs["args"] = [str(arg) for arg in args]
            return self._reject_unknown_inputs(descriptor, inputs)

        inputs = self._reject_unknown_inputs(descriptor, inputs)
        schema = descriptor.get("input_schema") or {}
        for required in schema.get("required") or []:
            if inputs.get(required) in (None, ""):
                raise ValidationError(f"{required} is required")
        self._validate_schema_enums(schema, inputs)
        self._validate_enums(action_id, inputs)
        if action_id == "hloc.pairsExhaustive" and not (
            inputs.get("image_list") or inputs.get("features_path")
        ):
            raise ValidationError("pairsExhaustive requires image_list or features_path")
        return inputs

    def _reject_unknown_inputs(
        self, descriptor: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        schema = descriptor.get("input_schema") or {}
        allowed = set((schema.get("properties") or {}).keys())
        unknown = sorted(set(inputs) - allowed)
        if unknown:
            raise ValidationError(f"unknown input(s): {', '.join(unknown)}")
        return inputs

    def _validate_schema_enums(self, schema: dict[str, Any], inputs: dict[str, Any]) -> None:
        properties = schema.get("properties") or {}
        for name, field_schema in properties.items():
            if name not in inputs or inputs[name] is None:
                continue
            allowed = field_schema.get("enum")
            if allowed is not None:
                value = str(inputs[name])
                if value not in allowed:
                    raise ValidationError(f"{name} must be one of: {', '.join(allowed)}")
                inputs[name] = value
                continue
            item_schema = field_schema.get("items") or {}
            item_allowed = item_schema.get("enum")
            if item_allowed is None:
                continue
            values = inputs[name]
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValidationError(f"{name} must be an array")
            bad = [str(value) for value in values if str(value) not in item_allowed]
            if bad:
                raise ValidationError(f"{name} values must be one of: {', '.join(item_allowed)}")
            inputs[name] = [str(value) for value in values]

    def _validate_enums(self, action_id: str, inputs: dict[str, Any]) -> None:
        for key, allowed in (
            ("feature_conf", FEATURE_CONFIGS),
            ("retrieval_conf", RETRIEVAL_CONFIGS),
            ("matcher_conf", MATCHER_CONFIGS),
            ("dense_conf", DENSE_CONFIGS),
            ("pairing_mode", PAIRING_MODES),
        ):
            if key in inputs and inputs[key] is not None:
                value = str(inputs[key])
                if value not in allowed:
                    raise ValidationError(f"{key} must be one of: {', '.join(allowed)}")
                inputs[key] = value
        if action_id == "hloc.extractFeatures" and "feature_conf" not in inputs:
            inputs["feature_conf"] = "superpoint_aachen"
        if action_id == "hloc.matchFeatures" and "matcher_conf" not in inputs:
            inputs["matcher_conf"] = "superglue"
        if action_id == "hloc.matchDense" and "dense_conf" not in inputs:
            inputs["dense_conf"] = "loftr"
        if action_id == "hloc.runPipeline":
            inputs.setdefault("pairing_mode", "exhaustive")
            inputs.setdefault("feature_conf", "superpoint_aachen")
            inputs.setdefault("retrieval_conf", "netvlad")
            inputs.setdefault("matcher_conf", "superglue")

    def _progress(self, progress: Any | None, _phase: str, current: int, total: int) -> None:
        if progress is None:
            return
        try:
            progress.phase_progress("backend_action", current=current, total=total)
        except Exception:
            return


__all__ = [
    "DEFAULT_HLOC_ROOT",
    "DENSE_CONFIGS",
    "FEATURE_CONFIGS",
    "HLOC_ACTIONS",
    "HLOC_BENCHMARK_ACTIONS",
    "HLOC_BENCHMARK_CLI_MODULES",
    "HLOC_CLI_MODULES",
    "HLOC_UPSTREAM_CLI_MODULES",
    "MATCHER_CONFIGS",
    "RETRIEVAL_CONFIGS",
    "HlocBackend",
    "configure_hloc_environment",
    "resolve_hloc_root",
]
