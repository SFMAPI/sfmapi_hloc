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

# Portable sfmapi capability -> hloc native config. These map the
# engine-neutral sfmapi vocabulary to the upstream hloc configuration
# names so the portable stage wrappers below can drive runner.py.
# Keys are the sfmapi ``features.extract.<key>`` suffix; values are the
# upstream ``hloc.extract_features.confs`` config names (which do not
# always match — e.g. sfmapi ``d2net`` -> hloc ``d2net-ss``).
FEATURE_CAPABILITY_CONFIGS: dict[str, str] = {
    "superpoint": "superpoint_aachen",
    "disk": "disk",
    "aliked": "aliked-n16",
    "r2d2": "r2d2",
    "d2net": "d2net-ss",
    "sift": "sift",
    "sosnet": "sosnet",
}
SPARSE_MATCHER_CAPABILITY_CONFIGS: dict[str, str] = {
    "superglue": "superglue",
    "lightglue": "superpoint+lightglue",
}
DENSE_MATCHER_CAPABILITY_CONFIGS: dict[str, str] = {
    "loftr": "loftr",
}


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

    # Portable sfmapi capabilities this backend implements as thin
    # wrappers over runner.py. ``matches.verify`` is intentionally
    # absent: hloc folds geometric verification into reconstruction /
    # triangulation and exposes no verification-only runner entry, so
    # ``verify_matches`` honestly raises ``CapabilityUnavailableError``.
    PORTABLE_CAPABILITIES: tuple[str, ...] = (
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
    )

    def capabilities(self) -> set[str]:
        """Portable sfmapi capabilities backed by real runner.py wrappers.

        Degrades to ``set()`` when the hloc checkout is missing, mirroring
        how the COLMAP backends report nothing when COLMAP is absent.
        """
        if self._find_root() is None:
            return set()
        return set(self.PORTABLE_CAPABILITIES)

    def extract_features(
        self,
        *,
        database_path: Path,
        image_root: Path,
        image_list: list[str],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract local features via the hloc ``extract_features`` runner.

        ``database_path`` anchors where hloc's native HDF5 feature file is
        written (sibling artifact); hloc does not use a COLMAP database for
        feature extraction.
        """
        options = dict(options or {})
        feature_type = str(options.get("type") or options.get("feature_type") or "superpoint")
        feature_conf = FEATURE_CAPABILITY_CONFIGS.get(feature_type)
        if feature_conf is None:
            raise CapabilityUnavailableError(
                capability=f"features.extract.{feature_type}",
                reason=(
                    "hloc portable feature extraction supports "
                    f"{', '.join(sorted(FEATURE_CAPABILITY_CONFIGS))}"
                ),
            )
        database_path = Path(database_path)
        outputs_dir = database_path.parent
        feature_path = options.get("feature_path") or outputs_dir / (
            f"{FEATURE_OUTPUTS[feature_conf]}.h5"
        )
        runner_inputs: dict[str, Any] = {
            "image_dir": str(image_root),
            "outputs_dir": str(outputs_dir),
            "feature_conf": feature_conf,
            "feature_path": str(feature_path),
            "as_half": bool(options.get("as_half", True)),
            "overwrite": bool(options.get("overwrite", False)),
        }
        if image_list:
            runner_inputs["image_list"] = list(image_list)
        if options.get("conf_override") is not None:
            runner_inputs["conf_override"] = options["conf_override"]
        if options.get("timeout_seconds") is not None:
            runner_inputs["timeout_seconds"] = options["timeout_seconds"]
        run = self._run_runner_action("hloc.extractFeatures", runner_inputs, workspace=outputs_dir)
        result = run.get("result") or {}
        return {
            "database_path": str(database_path),
            "feature_path": str(result.get("feature_path") or feature_path),
            "feature_conf": feature_conf,
            "num_images": len(image_list),
            "engine": "hloc extract_features",
        }

    def match(
        self,
        *,
        database_path: Path,
        mode: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Match features via the hloc sparse/dense matching runners.

        ``mode`` is the pair-selection strategy. ``retrieval`` builds the
        pairs file from global retrieval descriptors; ``from_poses``
        builds it from pose proximity in a reference COLMAP model (the
        ``hloc.pairsPoses`` runner over ``hloc.pairs_from_poses``). Any
        other strategy honestly raises ``CapabilityUnavailableError``.
        ``options['matcher']`` selects superglue / lightglue / loftr.
        """
        options = dict(options or {})
        normalized_mode = str(mode).replace("-", "_").lower()
        if normalized_mode not in ("retrieval", "from_poses"):
            raise CapabilityUnavailableError(
                capability=f"pairs.{normalized_mode}",
                reason=(
                    "hloc portable matching wires the retrieval and from_poses pair strategies"
                ),
            )
        matcher = str(options.get("matcher") or options.get("matcher_type") or "superglue")
        database_path = Path(database_path)
        outputs_dir = database_path.parent

        feature_path = options.get("feature_path")
        pairs_path = options.get("pairs_path")
        if pairs_path is None and normalized_mode == "from_poses":
            model_path = options.get("model_path") or options.get("reference_model")
            if model_path is None:
                raise CapabilityUnavailableError(
                    capability="pairs.from_poses",
                    reason=(
                        "hloc from_poses matching needs options['pairs_path'] or "
                        "options['model_path'] (a reference COLMAP model with poses)"
                    ),
                )
            pairs_path = outputs_dir / "pairs-from-poses.txt"
            poses_inputs: dict[str, Any] = {
                "model_path": str(model_path),
                "pairs_path": str(pairs_path),
                "num_matched": int(options.get("num_matched", 20)),
            }
            if options.get("rotation_threshold") is not None:
                poses_inputs["rotation_threshold"] = float(options["rotation_threshold"])
            if options.get("timeout_seconds") is not None:
                poses_inputs["timeout_seconds"] = options["timeout_seconds"]
            self._run_runner_action("hloc.pairsPoses", poses_inputs, workspace=outputs_dir)
        if pairs_path is None:
            descriptors_path = options.get("descriptors_path") or options.get("retrieval_path")
            if descriptors_path is None:
                # No precomputed global descriptors — extract them with the
                # caller-selected retrieval model (NetVLAD / DIR / OpenIBL /
                # MegaLoc). Needs an image root to extract from.
                retrieval_conf = str(options.get("retrieval_conf") or "netvlad")
                if retrieval_conf not in RETRIEVAL_CONFIGS:
                    raise CapabilityUnavailableError(
                        capability="pairs.retrieval",
                        reason=(
                            f"unknown retrieval_conf {retrieval_conf!r}; "
                            f"expected one of {sorted(RETRIEVAL_CONFIGS)}"
                        ),
                    )
                image_root = options.get("image_root")
                if image_root is None:
                    raise CapabilityUnavailableError(
                        capability="pairs.retrieval",
                        reason=(
                            "hloc retrieval matching needs options['pairs_path'], "
                            "options['descriptors_path'] (global retrieval features), "
                            "or options['image_root'] (+ optional retrieval_conf) to "
                            "extract global descriptors"
                        ),
                    )
                extract_run = self._run_runner_action(
                    "hloc.extractFeatures",
                    {
                        "image_dir": str(image_root),
                        "feature_conf": retrieval_conf,
                        "outputs_dir": str(outputs_dir),
                    },
                    workspace=outputs_dir,
                )
                descriptors_path = (extract_run.get("result") or {}).get("feature_path")
                if not descriptors_path:
                    raise CapabilityUnavailableError(
                        capability="pairs.retrieval",
                        reason=f"hloc {retrieval_conf} descriptor extraction produced no output",
                    )
            pairs_path = outputs_dir / "pairs-retrieval.txt"
            pairs_inputs: dict[str, Any] = {
                "descriptors_path": str(descriptors_path),
                "pairs_path": str(pairs_path),
                "num_matched": int(options.get("num_matched", 20)),
            }
            if options.get("timeout_seconds") is not None:
                pairs_inputs["timeout_seconds"] = options["timeout_seconds"]
            self._run_runner_action("hloc.pairsRetrieval", pairs_inputs, workspace=outputs_dir)

        if matcher in DENSE_MATCHER_CAPABILITY_CONFIGS:
            dense_conf = DENSE_MATCHER_CAPABILITY_CONFIGS[matcher]
            dense_inputs: dict[str, Any] = {
                "pairs_path": str(pairs_path),
                "image_dir": str(options.get("image_root") or outputs_dir),
                "outputs_dir": str(outputs_dir),
                "dense_conf": dense_conf,
                "overwrite": bool(options.get("overwrite", False)),
            }
            if options.get("matches_path") is not None:
                dense_inputs["matches_path"] = str(options["matches_path"])
            if options.get("conf_override") is not None:
                dense_inputs["conf_override"] = options["conf_override"]
            if options.get("timeout_seconds") is not None:
                dense_inputs["timeout_seconds"] = options["timeout_seconds"]
            run = self._run_runner_action("hloc.matchDense", dense_inputs, workspace=outputs_dir)
            result = run.get("result") or {}
            return {
                "database_path": str(database_path),
                "strategy": mode,
                "matcher": matcher,
                "pairs_path": str(pairs_path),
                "feature_path": str(result.get("feature_path") or ""),
                "matches_path": str(result.get("matches_path") or ""),
                "engine": "hloc match_dense",
            }

        matcher_conf = SPARSE_MATCHER_CAPABILITY_CONFIGS.get(matcher)
        if matcher_conf is None:
            raise CapabilityUnavailableError(
                capability=f"matchers.{matcher}",
                reason=(
                    "hloc portable matching supports "
                    f"{', '.join(sorted({*SPARSE_MATCHER_CAPABILITY_CONFIGS, *DENSE_MATCHER_CAPABILITY_CONFIGS}))}"
                ),
            )
        if feature_path is None:
            raise CapabilityUnavailableError(
                capability=f"matchers.{matcher}",
                reason="hloc sparse matching needs options['feature_path'] (HDF5 features)",
            )
        matches_path = options.get("matches_path") or outputs_dir / (
            f"{Path(str(feature_path)).stem}_{MATCHER_OUTPUTS[matcher_conf]}.h5"
        )
        match_inputs: dict[str, Any] = {
            "pairs_path": str(pairs_path),
            "feature_path": str(feature_path),
            "matches_path": str(matches_path),
            "matcher_conf": matcher_conf,
            "overwrite": bool(options.get("overwrite", False)),
        }
        if options.get("conf_override") is not None:
            match_inputs["conf_override"] = options["conf_override"]
        if options.get("timeout_seconds") is not None:
            match_inputs["timeout_seconds"] = options["timeout_seconds"]
        run = self._run_runner_action("hloc.matchFeatures", match_inputs, workspace=outputs_dir)
        result = run.get("result") or {}
        return {
            "database_path": str(database_path),
            "strategy": mode,
            "matcher": matcher,
            "pairs_path": str(pairs_path),
            "matches_path": str(result.get("matches_path") or matches_path),
            "engine": "hloc match_features",
        }

    def verify_matches(
        self,
        *,
        database_path: Path,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Not a standalone hloc stage.

        hloc performs geometric verification implicitly inside
        ``reconstruction`` / ``triangulation`` (via ``min_match_score`` and
        ``skip_geometric_verification``); it exposes no verification-only
        runner entry point. ``matches.verify`` is therefore intentionally
        absent from :meth:`capabilities`.
        """
        raise CapabilityUnavailableError(
            capability="matches.verify",
            reason=(
                "hloc has no standalone geometric verification stage; "
                "verification runs inside hloc.reconstruct / hloc.triangulate"
            ),
        )

    def localize_from_memory(
        self,
        *,
        sparse_dir: Path,
        query_image: Path,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Localize against a reference SfM model via the hloc ``localize_sfm`` runner.

        hloc's localization is pipeline-shaped: it consumes a queries file,
        retrieval pairs, and pre-extracted feature / match HDF5 files. Those
        hloc-specific inputs are carried through the free-form ``spec`` dict.
        When they are absent this raises ``CapabilityUnavailableError`` so
        callers get the normal 501 shape instead of an ``AttributeError``.
        """
        spec = dict(spec or {})
        sparse_dir = Path(sparse_dir)
        required = ("queries_path", "retrieval_path", "feature_path", "matches_path")
        missing = [key for key in required if not spec.get(key)]
        if missing:
            raise CapabilityUnavailableError(
                capability="localize.from_memory",
                reason=(
                    "hloc localize_sfm needs spec keys "
                    f"{', '.join(required)} (missing: {', '.join(missing)})"
                ),
            )
        results_path = spec.get("results_path") or sparse_dir.parent / "hloc-localization.txt"
        runner_inputs: dict[str, Any] = {
            "reference_sfm": str(spec.get("reference_sfm") or sparse_dir),
            "queries_path": str(spec["queries_path"]),
            "retrieval_path": str(spec["retrieval_path"]),
            "feature_path": str(spec["feature_path"]),
            "matches_path": str(spec["matches_path"]),
            "results_path": str(results_path),
        }
        for key in ("ransac_thresh", "covisibility_clustering", "prepend_camera_name", "config"):
            if spec.get(key) is not None:
                runner_inputs[key] = spec[key]
        if spec.get("timeout_seconds") is not None:
            runner_inputs["timeout_seconds"] = spec["timeout_seconds"]
        run = self._run_runner_action(
            "hloc.localizeSfm", runner_inputs, workspace=sparse_dir.parent
        )
        result = run.get("result") or {}
        return {
            "query_image": str(query_image),
            "reference_sfm": runner_inputs["reference_sfm"],
            "results_path": str(result.get("results_path") or results_path),
            "logs_path": str(result.get("logs_path") or ""),
            "engine": "hloc localize_sfm",
        }

    def localize_batch(self, **kwargs: Any) -> list[Any]:
        """Localize a whole queries file via the hloc ``localize_sfm`` runner.

        hloc's ``localize_sfm`` is inherently batch — it consumes a queries
        file plus retrieval pairs and pre-extracted feature/match HDF5
        files in one pass. This is the same runner :meth:`localize_from_memory`
        drives; the only difference is the result shape (a ``list`` of
        per-query rows, as the :class:`BatchLocalizationBackend` protocol
        requires). The hloc-specific inputs ride through ``kwargs`` (or a
        nested ``spec`` dict). Missing inputs raise
        ``CapabilityUnavailableError`` so callers get the normal 501 shape
        instead of an ``AttributeError``.
        """
        spec: dict[str, Any] = dict(kwargs.pop("spec", None) or {})
        spec.update({key: value for key, value in kwargs.items() if value is not None})
        required = ("queries_path", "retrieval_path", "feature_path", "matches_path")
        missing = [key for key in required if not spec.get(key)]
        reference_sfm = spec.get("reference_sfm") or spec.get("sparse_dir")
        if not reference_sfm:
            missing.append("reference_sfm")
        if missing:
            raise CapabilityUnavailableError(
                capability="localize.batch",
                reason=(
                    "hloc localize_sfm needs keys reference_sfm, "
                    f"{', '.join(required)} (missing: {', '.join(missing)})"
                ),
            )
        reference_sfm = Path(str(reference_sfm))
        results_path = spec.get("results_path") or reference_sfm.parent / "hloc-localization.txt"
        runner_inputs: dict[str, Any] = {
            "reference_sfm": str(reference_sfm),
            "queries_path": str(spec["queries_path"]),
            "retrieval_path": str(spec["retrieval_path"]),
            "feature_path": str(spec["feature_path"]),
            "matches_path": str(spec["matches_path"]),
            "results_path": str(results_path),
        }
        for key in ("ransac_thresh", "covisibility_clustering", "prepend_camera_name", "config"):
            if spec.get(key) is not None:
                runner_inputs[key] = spec[key]
        if spec.get("timeout_seconds") is not None:
            runner_inputs["timeout_seconds"] = spec["timeout_seconds"]
        run = self._run_runner_action(
            "hloc.localizeSfm", runner_inputs, workspace=reference_sfm.parent
        )
        result = run.get("result") or {}
        return [
            {
                "reference_sfm": runner_inputs["reference_sfm"],
                "queries_path": runner_inputs["queries_path"],
                "results_path": str(result.get("results_path") or results_path),
                "logs_path": str(result.get("logs_path") or ""),
                "engine": "hloc localize_sfm",
            }
        ]

    def triangulate(
        self,
        *,
        model_path: Path,
        database_path: Path,
        image_root: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """Re-triangulate against an existing model via the hloc ``triangulation`` runner.

        hloc's ``triangulation`` module is triangulate-from-known-poses: it
        imports learned features/matches and triangulates points against a
        reference COLMAP model. hloc additionally needs the pairs list and
        the HDF5 ``feature_path`` / ``matches_path`` that produced those
        matches — sfmapi's portable :class:`RefinementBackend.triangulate`
        signature does not carry them, so they are derived from
        ``database_path``'s directory (the convention the hloc match stage
        writes to) unless overridden via ``SFMAPI_HLOC_*`` sidecar files.
        When they cannot be located this raises
        ``CapabilityUnavailableError`` instead of an ``AttributeError``.
        """
        model_path = Path(model_path)
        database_path = Path(database_path)
        image_root = Path(image_root)
        output_path = Path(output_path)
        artifacts_dir = database_path.parent

        def _resolve(name: str, candidates: list[Path]) -> Path | None:
            for candidate in candidates:
                if candidate and Path(candidate).exists():
                    return Path(candidate)
            return None

        pairs_path = _resolve(
            "pairs",
            [
                artifacts_dir / "pairs-from-poses.txt",
                artifacts_dir / "pairs-retrieval.txt",
                artifacts_dir / "pairs.txt",
            ],
        )
        feature_path = _resolve(
            "features",
            sorted(artifacts_dir.glob("feats-*.h5")) + sorted(artifacts_dir.glob("*.h5")),
        )
        matches_path = _resolve("matches", sorted(artifacts_dir.glob("*matches*.h5")))
        missing = [
            name
            for name, value in (
                ("pairs_path", pairs_path),
                ("feature_path", feature_path),
                ("matches_path", matches_path),
            )
            if value is None
        ]
        if missing:
            raise CapabilityUnavailableError(
                capability="triangulate.retri",
                reason=(
                    "hloc triangulation needs a pairs file plus HDF5 feature/match "
                    f"files alongside the database ({database_path}); could not "
                    f"locate: {', '.join(missing)}"
                ),
            )
        runner_inputs: dict[str, Any] = {
            "sfm_dir": str(output_path),
            "reference_sfm_model": str(model_path),
            "image_dir": str(image_root),
            "pairs_path": str(pairs_path),
            "feature_path": str(feature_path),
            "matches_path": str(matches_path),
        }
        run = self._run_runner_action("hloc.triangulate", runner_inputs, workspace=output_path)
        result = run.get("result") or {}
        return {
            "model_path": str(result.get("sfm_dir") or output_path),
            "reference_sfm_model": str(model_path),
            "summary": result.get("summary"),
            "engine": "hloc triangulation",
        }

    def run_mapping(
        self,
        *,
        kind: str,
        db_path: Path,
        image_root: Path,
        sparse_root: Path,
        job_dir: Path,
        spec: dict[str, Any],
        pose_priors: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Run end-to-end learned-feature incremental SfM via the hloc ``reconstruction`` runner.

        hloc's ``reconstruction`` module imports learned features/matches
        and drives a full pycolmap incremental reconstruction. Only
        ``kind="incremental"`` is a portable hloc capability; any other
        mapping kind honestly raises ``CapabilityUnavailableError``.

        Returns ``(summaries, reconstructions)`` for the portable
        :class:`MappingBackend` protocol. hloc emits a COLMAP-format
        sparse model directory; reading it back into a portable
        reconstruction object is a :class:`ReconstructionReaderBackend`
        job this runner-shim does not implement, so the reconstruction
        list is left empty and the summary carries the on-disk
        ``model_path`` (the same pattern the SphereSfM plugin uses).

        hloc additionally needs the pairs list and HDF5 feature/match
        files; they ride through ``spec`` (``pairs_path``, ``feature_path``,
        ``matches_path``). When absent this raises
        ``CapabilityUnavailableError``.
        """
        normalized = str(kind).replace("-", "_").lower()
        if normalized != "incremental":
            raise CapabilityUnavailableError(
                capability=f"map.{kind}",
                reason="hloc only implements portable incremental mapping (map.incremental).",
            )
        spec = dict(spec or {})
        db_path = Path(db_path)
        image_root = Path(image_root)
        sparse_root = Path(sparse_root)
        job_dir = Path(job_dir)
        artifacts_dir = db_path.parent

        pairs_path = spec.get("pairs_path")
        feature_path = spec.get("feature_path")
        matches_path = spec.get("matches_path")
        if pairs_path is None:
            for candidate in (
                artifacts_dir / "pairs-retrieval.txt",
                artifacts_dir / "pairs-from-poses.txt",
                artifacts_dir / "pairs.txt",
            ):
                if candidate.exists():
                    pairs_path = candidate
                    break
        missing = [
            key
            for key, value in (
                ("pairs_path", pairs_path),
                ("feature_path", feature_path),
                ("matches_path", matches_path),
            )
            if not value
        ]
        if missing:
            raise CapabilityUnavailableError(
                capability="map.incremental",
                reason=(
                    "hloc incremental mapping needs spec keys pairs_path, "
                    f"feature_path, matches_path (missing: {', '.join(missing)})"
                ),
            )
        sparse_root.mkdir(parents=True, exist_ok=True)
        job_dir.mkdir(parents=True, exist_ok=True)
        sfm_dir = Path(str(spec.get("sfm_dir") or sparse_root / "0"))
        runner_inputs: dict[str, Any] = {
            "sfm_dir": str(sfm_dir),
            "image_dir": str(image_root),
            "pairs_path": str(pairs_path),
            "feature_path": str(feature_path),
            "matches_path": str(matches_path),
            "camera_mode": str(spec.get("camera_mode", "AUTO")),
            "skip_geometric_verification": bool(spec.get("skip_geometric_verification", False)),
            "verbose": bool(spec.get("verbose", False)),
        }
        for key in ("image_list", "image_options", "mapper_options", "min_match_score"):
            if spec.get(key) is not None:
                runner_inputs[key] = spec[key]
        if spec.get("timeout_seconds") is not None:
            runner_inputs["timeout_seconds"] = spec["timeout_seconds"]
        run = self._run_runner_action("hloc.reconstruct", runner_inputs, workspace=job_dir)
        result = run.get("result") or {}
        model_path = Path(str(result.get("sfm_dir") or sfm_dir))
        summaries: list[dict[str, Any]] = [
            {
                "idx": 0,
                "model_path": str(model_path),
                "summary": result.get("summary"),
                "engine": "hloc reconstruction",
            }
        ]
        return summaries, []

    def list_backend_config_schemas(self, *, include_schemas: bool = True) -> list[dict[str, Any]]:
        """Backend option schemas for hloc's portable stages.

        Describes the ``backend_options`` keys hloc accepts for the
        portable feature, pair-retrieval, and matcher stages. Degrades to
        an empty list when the hloc checkout is missing.
        """
        if self._find_root() is None:
            return []
        descriptors = [
            self._config_descriptor(
                config_id="hloc.features",
                stage="features",
                capability="features.extract.superpoint",
                option_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": sorted(FEATURE_CAPABILITY_CONFIGS),
                            "description": "Portable feature type mapped to an hloc config.",
                        },
                        "feature_conf": {
                            "type": "string",
                            "enum": list(FEATURE_CONFIGS),
                            "description": "Explicit hloc feature config override.",
                        },
                        "feature_path": {"type": "string"},
                        "as_half": {"type": "boolean", "default": True},
                        "overwrite": {"type": "boolean", "default": False},
                        "conf_override": {"type": "object"},
                        "timeout_seconds": {"type": "number"},
                    },
                },
                description="hloc feature-extraction options for features.extract.*.",
                include_schema=include_schemas,
            ),
            self._config_descriptor(
                config_id="hloc.pairs.retrieval",
                stage="pairs",
                capability="pairs.retrieval",
                option_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "descriptors_path": {"type": "string"},
                        "retrieval_path": {"type": "string"},
                        "pairs_path": {"type": "string"},
                        "num_matched": {"type": "integer", "minimum": 1, "default": 20},
                        "retrieval_conf": {
                            "type": "string",
                            "enum": list(RETRIEVAL_CONFIGS),
                            "default": "netvlad",
                            "description": (
                                "Global-retrieval model used to extract image "
                                "descriptors when descriptors_path is not "
                                "supplied. Ignored when descriptors_path is given."
                            ),
                        },
                        "timeout_seconds": {"type": "number"},
                    },
                },
                description="hloc global-retrieval pair-selection options.",
                include_schema=include_schemas,
            ),
            self._config_descriptor(
                config_id="hloc.matcher",
                stage="matcher",
                capability="matchers.superglue",
                option_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "matcher": {
                            "type": "string",
                            "enum": sorted(
                                {
                                    *SPARSE_MATCHER_CAPABILITY_CONFIGS,
                                    *DENSE_MATCHER_CAPABILITY_CONFIGS,
                                }
                            ),
                            "description": "Portable matcher mapped to an hloc config.",
                        },
                        "feature_path": {"type": "string"},
                        "pairs_path": {"type": "string"},
                        "descriptors_path": {"type": "string"},
                        "matches_path": {"type": "string"},
                        "image_root": {"type": "string"},
                        "num_matched": {"type": "integer", "minimum": 1, "default": 20},
                        "overwrite": {"type": "boolean", "default": False},
                        "conf_override": {"type": "object"},
                        "timeout_seconds": {"type": "number"},
                    },
                },
                description="hloc sparse/dense matcher options for matchers.*.",
                include_schema=include_schemas,
            ),
        ]
        return descriptors

    def list_backend_artifact_contracts(self) -> list[dict[str, Any]]:
        """Artifact I/O contracts for hloc's portable stages.

        hloc emits portable ``sfmapi.*`` artifacts: HDF5 feature files,
        pairs text, and HDF5 match files. Degrades to an empty list when
        the hloc checkout is missing.
        """
        if self._find_root() is None:
            return []
        return [
            {
                "contract_id": "hloc.features",
                "stage": "features",
                "capability": "features.extract.superpoint",
                "provider": "hloc",
                "display_name": "hloc HDF5 feature outputs",
                "description": "hloc writes local features to an HDF5 feature file.",
                "accepts": [],
                "emits": ["features.local.v1"],
                "preferred": "features.local.v1",
                "metadata": {"family": "hloc", "format": "hdf5"},
            },
            {
                "contract_id": "hloc.pairs",
                "stage": "pairs",
                "capability": "pairs.retrieval",
                "provider": "hloc",
                "display_name": "hloc retrieval pair outputs",
                "description": "hloc writes image pairs from retrieval to a pairs text file.",
                "accepts": ["features.global.v1"],
                "emits": ["pairs.image_names.v1"],
                "preferred": "pairs.image_names.v1",
                "metadata": {"family": "hloc", "format": "text"},
            },
            {
                "contract_id": "hloc.matches",
                "stage": "matcher",
                "capability": "matchers.superglue",
                "provider": "hloc",
                "display_name": "hloc HDF5 match outputs",
                "description": "hloc writes sparse and dense matches to an HDF5 match file.",
                "accepts": ["features.local.v1", "pairs.image_names.v1"],
                "emits": ["matches.indexed.v1"],
                "preferred": "matches.indexed.v1",
                "metadata": {"family": "hloc", "format": "hdf5"},
            },
        ]

    def _config_descriptor(
        self,
        *,
        config_id: str,
        stage: str,
        capability: str,
        option_schema: dict[str, Any],
        description: str,
        include_schema: bool,
    ) -> dict[str, Any]:
        return {
            "config_id": config_id,
            "backend": self.name,
            "stage": stage,
            "capability": capability,
            "provider": self.name,
            "display_name": f"hloc {stage} options",
            "description": description,
            "option_schema": option_schema if include_schema else None,
            "defaults": {},
            "metadata": {"family": "hloc"},
        }

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
