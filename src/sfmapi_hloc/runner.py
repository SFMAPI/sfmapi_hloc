from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
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

    try:
        return pycolmap.CameraMode[str(value)]
    except KeyError as exc:
        allowed = ", ".join(pycolmap.CameraMode.__members__.keys())
        raise ValueError(f"camera_mode must be one of: {allowed}") from exc


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

    output = _path(inputs["pairs_path"])
    pairs_from_poses.main(
        _path(inputs["model_path"]),
        output,
        int(inputs["num_matched"]),
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
    features_path, matches_path = match_dense.main(
        conf,
        _path(inputs["pairs_path"]),
        _path(inputs["image_dir"]),
        export_dir=_path(inputs["outputs_dir"]),
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
