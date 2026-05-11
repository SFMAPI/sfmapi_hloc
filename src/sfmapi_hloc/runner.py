from __future__ import annotations

import ast
import importlib
import json
import runpy
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _path(value: Any) -> Path:
    return Path(str(value))


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return _path(value)


def _path_or_list(value: Any) -> Path | list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return _path(value)


def _path_list(value: Any) -> list[Path] | Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [_path(item) for item in value]
    return _path(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    result = deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _summary(reconstruction: Any) -> str | None:
    if reconstruction is None:
        return None
    summary = getattr(reconstruction, "summary", None)
    if callable(summary):
        return str(summary())
    return str(reconstruction)


def _camera_mode(value: str) -> Any:
    import pycolmap

    value = str(value).upper()
    try:
        return getattr(pycolmap.CameraMode, value)
    except AttributeError as exc:
        allowed = ", ".join(pycolmap.CameraMode.__members__.keys())
        raise ValueError(f"camera_mode must be one of: {allowed}") from exc


def _add_cli_option(args: list[str], name: str, value: Any, *, flag: str | None = None) -> None:
    if value in (None, ""):
        return
    option = flag or f"--{name}"
    if isinstance(value, bool):
        if value:
            args.append(option)
        return
    if isinstance(value, list):
        if value:
            args.append(option)
            args.extend(str(item) for item in value)
        return
    args.extend([option, str(value)])


def _run_module_as_main(module: str, args: list[str]) -> dict[str, Any]:
    old_argv = sys.argv[:]
    try:
        sys.argv = [module, *args]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv
    return {"module": module, "args": ["python", "-m", module, *args]}


def _run_dataset_pipeline(
    module: str, inputs: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    pipeline = importlib.import_module(module)
    args = SimpleNamespace(
        dataset=_path(inputs.get("dataset", defaults["dataset"])),
        outputs=_path(inputs.get("outputs", defaults["outputs"])),
        num_covis=int(inputs.get("num_covis", defaults["num_covis"])),
        num_loc=int(inputs.get("num_loc", defaults["num_loc"])),
    )
    pipeline.run(args)
    return {
        "module": module,
        "dataset": str(args.dataset),
        "outputs": str(args.outputs),
        "num_covis": args.num_covis,
        "num_loc": args.num_loc,
    }


def _pipeline_cli_args(inputs: dict[str, Any], option_names: list[str]) -> list[str]:
    args: list[str] = []
    for name in option_names:
        _add_cli_option(args, name, inputs.get(name))
    return args


def run_extract_features(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import extract_features

    conf = _deep_merge(
        extract_features.confs[str(inputs.get("feature_conf", "superpoint_aachen"))],
        inputs.get("conf_override"),
    )
    feature_path = extract_features.main(
        conf,
        _path(inputs["image_dir"]),
        export_dir=_optional_path(inputs.get("outputs_dir")),
        as_half=bool(inputs.get("as_half", True)),
        image_list=_path_or_list(inputs.get("image_list")),
        feature_path=_optional_path(inputs.get("feature_path")),
        overwrite=bool(inputs.get("overwrite", False)),
    )
    return {"feature_path": str(feature_path)}


def run_pairs_exhaustive(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import pairs_from_exhaustive

    output = _path(inputs["pairs_path"])
    pairs_from_exhaustive.main(
        output,
        image_list=_path_or_list(inputs.get("image_list")),
        features=_optional_path(inputs.get("features_path")),
        ref_list=_path_or_list(inputs.get("ref_list")),
        ref_features=_optional_path(inputs.get("ref_features_path")),
    )
    return {"pairs_path": str(output)}


def run_pairs_retrieval(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import pairs_from_retrieval

    output = _path(inputs["pairs_path"])
    pairs_from_retrieval.main(
        _path(inputs["descriptors_path"]),
        output,
        int(inputs["num_matched"]),
        query_prefix=inputs.get("query_prefix"),
        query_list=_optional_path(inputs.get("query_list")),
        db_prefix=inputs.get("db_prefix"),
        db_list=_optional_path(inputs.get("db_list")),
        db_model=_optional_path(inputs.get("db_model")),
        db_descriptors=_path_list(inputs.get("db_descriptors")),
    )
    return {"pairs_path": str(output)}


def run_pairs_covisibility(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import pairs_from_covisibility

    output = _path(inputs["pairs_path"])
    pairs_from_covisibility.main(_path(inputs["model_path"]), output, int(inputs["num_matched"]))
    return {"pairs_path": str(output)}


def run_pairs_poses(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import pairs_from_poses
    from hloc.utils.read_write_model import read_images_binary

    output = _path(inputs["pairs_path"])
    model_path = _path(inputs["model_path"])
    num_matched = int(inputs["num_matched"])
    image_count = len(read_images_binary(model_path / "images.bin"))
    if num_matched >= image_count:
        raise ValueError(
            "num_matched must be smaller than the number of registered images "
            f"in the model ({image_count})"
        )
    pairs_from_poses.main(
        model_path,
        output,
        num_matched,
        rotation_threshold=float(inputs.get("rotation_threshold", 30.0)),
    )
    return {"pairs_path": str(output)}


def run_match_features(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import match_features

    conf = _deep_merge(
        match_features.confs[str(inputs.get("matcher_conf", "superglue"))],
        inputs.get("conf_override"),
    )
    matches_path = match_features.main(
        conf,
        _path(inputs["pairs_path"]),
        _path(inputs["feature_path"]),
        export_dir=_optional_path(inputs.get("export_dir")),
        matches=_path(inputs["matches_path"]),
        features_ref=_optional_path(inputs.get("features_ref_path")),
        overwrite=bool(inputs.get("overwrite", False)),
    )
    return {"matches_path": str(matches_path)}


def run_match_dense(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import match_dense

    conf = _deep_merge(
        match_dense.confs[str(inputs.get("dense_conf", "loftr"))],
        inputs.get("conf_override"),
    )
    outputs_dir = _path(inputs["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    features_path, matches_path = match_dense.main(
        conf,
        _path(inputs["pairs_path"]),
        _path(inputs["image_dir"]),
        export_dir=outputs_dir,
        matches=_optional_path(inputs.get("matches_path")),
        features=_optional_path(inputs.get("features_path")),
        features_ref=_path_list(inputs.get("features_ref_path")),
        max_kps=inputs.get("max_kps", 8192),
        overwrite=bool(inputs.get("overwrite", False)),
    )
    return {"feature_path": str(features_path), "matches_path": str(matches_path)}


def run_reconstruct(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import reconstruction

    sfm_dir = _path(inputs["sfm_dir"])
    reconstruction_obj = reconstruction.main(
        sfm_dir,
        _path(inputs["image_dir"]),
        _path(inputs["pairs_path"]),
        _path(inputs["feature_path"]),
        _path(inputs["matches_path"]),
        camera_mode=_camera_mode(str(inputs.get("camera_mode", "AUTO"))),
        verbose=bool(inputs.get("verbose", False)),
        skip_geometric_verification=bool(inputs.get("skip_geometric_verification", False)),
        min_match_score=inputs.get("min_match_score"),
        image_list=inputs.get("image_list"),
        image_options=inputs.get("image_options"),
        mapper_options=inputs.get("mapper_options"),
    )
    return {"sfm_dir": str(sfm_dir), "summary": _summary(reconstruction_obj)}


def run_triangulate(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import triangulation

    sfm_dir = _path(inputs["sfm_dir"])
    reconstruction_obj = triangulation.main(
        sfm_dir,
        _path(inputs["reference_sfm_model"]),
        _path(inputs["image_dir"]),
        _path(inputs["pairs_path"]),
        _path(inputs["feature_path"]),
        _path(inputs["matches_path"]),
        skip_geometric_verification=bool(inputs.get("skip_geometric_verification", False)),
        estimate_two_view_geometries=bool(inputs.get("estimate_two_view_geometries", False)),
        min_match_score=inputs.get("min_match_score"),
        verbose=bool(inputs.get("verbose", False)),
        mapper_options=inputs.get("mapper_options"),
    )
    return {"sfm_dir": str(sfm_dir), "summary": _summary(reconstruction_obj)}


def run_localize_sfm(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import localize_sfm

    results = _path(inputs["results_path"])
    localize_sfm.main(
        _path(inputs["reference_sfm"]),
        _path(inputs["queries_path"]),
        _path(inputs["retrieval_path"]),
        _path(inputs["feature_path"]),
        _path(inputs["matches_path"]),
        results,
        ransac_thresh=float(inputs.get("ransac_thresh", 12.0)),
        covisibility_clustering=bool(inputs.get("covisibility_clustering", False)),
        prepend_camera_name=bool(inputs.get("prepend_camera_name", False)),
        config=inputs.get("config"),
    )
    return {"results_path": str(results), "logs_path": f"{results}_logs.pkl"}


def run_localize_inloc(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import localize_inloc

    results = _path(inputs["results_path"])
    localize_inloc.main(
        _path(inputs["dataset_dir"]),
        _path(inputs["retrieval_path"]),
        _path(inputs["feature_path"]),
        _path(inputs["matches_path"]),
        results,
        skip_matches=inputs.get("skip_matches"),
    )
    return {"results_path": str(results)}


def run_colmap_from_nvm(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import colmap_from_nvm

    output = _path(inputs["output_path"])
    colmap_from_nvm.main(
        _path(inputs["nvm_path"]),
        _path(inputs["intrinsics_path"]),
        _path(inputs["database_path"]),
        output,
        skip_points=bool(inputs.get("skip_points", False)),
    )
    return {"output_path": str(output), "database_path": str(inputs["database_path"])}


def run_convert_model(inputs: dict[str, Any]) -> dict[str, Any]:
    args: list[str] = []
    _add_cli_option(args, "input_model", inputs["input_model"])
    _add_cli_option(args, "input_format", inputs.get("input_format"))
    if inputs.get("output_model"):
        _path(inputs["output_model"]).mkdir(parents=True, exist_ok=True)
    _add_cli_option(args, "output_model", inputs.get("output_model"))
    _add_cli_option(args, "output_format", inputs.get("output_format"))
    return _run_module_as_main("hloc.utils.read_write_model", args)


def run_list_configs(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc import extract_features, match_dense, match_features

    include_values = bool(inputs.get("include_values", False))
    configs = {
        "feature_configs": extract_features.confs,
        "matcher_configs": match_features.confs,
        "dense_configs": match_dense.confs,
    }
    if include_values:
        return configs
    return {name: sorted(values) for name, values in configs.items()}


def run_pipeline_aachen(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run_dataset_pipeline(
        "hloc.pipelines.Aachen.pipeline",
        inputs,
        {
            "dataset": "datasets/aachen",
            "outputs": "outputs/aachen",
            "num_covis": 20,
            "num_loc": 50,
        },
    )


def run_pipeline_aachen_v11(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run_dataset_pipeline(
        "hloc.pipelines.Aachen_v1_1.pipeline",
        inputs,
        {
            "dataset": "datasets/aachen_v1.1",
            "outputs": "outputs/aachen_v1.1",
            "num_covis": 20,
            "num_loc": 50,
        },
    )


def run_pipeline_aachen_v11_loftr(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run_dataset_pipeline(
        "hloc.pipelines.Aachen_v1_1.pipeline_loftr",
        inputs,
        {
            "dataset": "datasets/aachen_v1.1",
            "outputs": "outputs/aachen_v1.1",
            "num_covis": 20,
            "num_loc": 50,
        },
    )


def run_pipeline_robotcar(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run_dataset_pipeline(
        "hloc.pipelines.RobotCar.pipeline",
        inputs,
        {
            "dataset": "datasets/robotcar",
            "outputs": "outputs/robotcar",
            "num_covis": 20,
            "num_loc": 20,
        },
    )


def run_pipeline_robotcar_colmap_from_nvm(inputs: dict[str, Any]) -> dict[str, Any]:
    from hloc.pipelines.RobotCar import colmap_from_nvm

    output = _path(inputs["output_path"])
    colmap_from_nvm.main(
        _path(inputs["nvm_path"]),
        _path(inputs["database_path"]),
        output,
        skip_points=bool(inputs.get("skip_points", False)),
    )
    return {"output_path": str(output), "database_path": str(inputs["database_path"])}


def _parse_cmu_slices(value: Any, all_slices: list[int]) -> list[int]:
    if value in (None, "", "*"):
        return list(all_slices)
    text = str(value)
    if "-" in text:
        minimum, maximum = text.split("-", 1)
        return list(range(int(minimum), int(maximum) + 1))
    parsed = ast.literal_eval(text)
    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, int) for item in parsed):
        return parsed
    raise ValueError("slices must be '*', an int, an inclusive range like 2-6, or a list[int]")


def run_pipeline_cmu(inputs: dict[str, Any]) -> dict[str, Any]:
    pipeline = importlib.import_module("hloc.pipelines.CMU.pipeline")
    slices = _parse_cmu_slices(inputs.get("slices", "*"), list(pipeline.TEST_SLICES))
    dataset = _path(inputs.get("dataset", "datasets/cmu_extended"))
    outputs = _path(inputs.get("outputs", "outputs/aachen_extended"))
    num_covis = int(inputs.get("num_covis", 20))
    num_loc = int(inputs.get("num_loc", 10))
    for slice_id in slices:
        pipeline.logger.info("Working on slice %s.", slice_id)
        pipeline.run_slice(f"slice{slice_id}", dataset, outputs, num_covis, num_loc)
    return {
        "module": "hloc.pipelines.CMU.pipeline",
        "slices": slices,
        "dataset": str(dataset),
        "outputs": str(outputs),
    }


def run_pipeline_cambridge(inputs: dict[str, Any]) -> dict[str, Any]:
    args = _pipeline_cli_args(
        inputs, ["scenes", "overwrite", "dataset", "outputs", "num_covis", "num_loc"]
    )
    return _run_module_as_main("hloc.pipelines.Cambridge.pipeline", args)


def run_pipeline_seven_scenes(inputs: dict[str, Any]) -> dict[str, Any]:
    args = _pipeline_cli_args(
        inputs,
        ["scenes", "overwrite", "dataset", "outputs", "use_dense_depth", "num_covis"],
    )
    return _run_module_as_main("hloc.pipelines.7Scenes.pipeline", args)


def run_pipeline_seven_scenes_correct_depth(inputs: dict[str, Any]) -> dict[str, Any]:
    correct_sfm_with_gt_depth = importlib.import_module(
        "hloc.pipelines.7Scenes.create_gt_sfm"
    ).correct_sfm_with_gt_depth

    scenes = inputs.get(
        "scenes",
        ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"],
    )
    dataset = _path(inputs.get("dataset", "datasets/7scenes"))
    outputs = _path(inputs.get("outputs", "outputs/7Scenes"))
    corrected: list[dict[str, str]] = []
    for scene in scenes:
        sfm_path = outputs / str(scene) / "sfm_superpoint+superglue"
        depth_path = dataset / f"depth/7scenes_{scene}/train/depth"
        output_path = outputs / str(scene) / "sfm_superpoint+superglue+depth"
        correct_sfm_with_gt_depth(sfm_path, depth_path, output_path)
        corrected.append(
            {
                "scene": str(scene),
                "input_model": str(sfm_path),
                "depth_path": str(depth_path),
                "output_model": str(output_path),
            }
        )
    return {"corrected": corrected}


def run_pipeline_four_seasons_prepare_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    args = _pipeline_cli_args(inputs, ["dataset", "outputs"])
    return _run_module_as_main("hloc.pipelines.4Seasons.prepare_reference", args)


def run_pipeline_four_seasons_localize(inputs: dict[str, Any]) -> dict[str, Any]:
    args = _pipeline_cli_args(inputs, ["sequence", "dataset", "outputs"])
    return _run_module_as_main("hloc.pipelines.4Seasons.localize", args)


RUNNERS = {
    "hloc.extractFeatures": run_extract_features,
    "hloc.pairsExhaustive": run_pairs_exhaustive,
    "hloc.pairsRetrieval": run_pairs_retrieval,
    "hloc.pairsCovisibility": run_pairs_covisibility,
    "hloc.pairsPoses": run_pairs_poses,
    "hloc.matchFeatures": run_match_features,
    "hloc.matchDense": run_match_dense,
    "hloc.reconstruct": run_reconstruct,
    "hloc.triangulate": run_triangulate,
    "hloc.localizeSfm": run_localize_sfm,
    "hloc.localizeInLoc": run_localize_inloc,
    "hloc.colmapFromNvm": run_colmap_from_nvm,
    "hloc.convertModel": run_convert_model,
    "hloc.listConfigs": run_list_configs,
    "hloc.pipelineAachen": run_pipeline_aachen,
    "hloc.pipelineAachenV11": run_pipeline_aachen_v11,
    "hloc.pipelineAachenV11LoFTR": run_pipeline_aachen_v11_loftr,
    "hloc.pipelineRobotCar": run_pipeline_robotcar,
    "hloc.pipelineRobotCarColmapFromNvm": run_pipeline_robotcar_colmap_from_nvm,
    "hloc.pipelineCMU": run_pipeline_cmu,
    "hloc.pipelineCambridge": run_pipeline_cambridge,
    "hloc.pipelineSevenScenes": run_pipeline_seven_scenes,
    "hloc.pipelineSevenScenesCorrectDepth": run_pipeline_seven_scenes_correct_depth,
    "hloc.pipelineFourSeasonsPrepareReference": run_pipeline_four_seasons_prepare_reference,
    "hloc.pipelineFourSeasonsLocalize": run_pipeline_four_seasons_localize,
}


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        raise SystemExit("usage: python -m sfmapi_hloc.runner ACTION INPUT_JSON OUTPUT_JSON")
    action_id, input_json, output_json = args
    runner = RUNNERS.get(action_id)
    if runner is None:
        raise SystemExit(f"unknown HLOC runner action: {action_id}")
    inputs = json.loads(Path(input_json).read_text(encoding="utf-8"))
    result = runner(inputs)
    Path(output_json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
