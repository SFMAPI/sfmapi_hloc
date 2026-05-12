from __future__ import annotations

import tomllib
from collections.abc import Callable
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import pytest

from sfmapi_hloc.backend import HlocBackend
from sfmapi_hloc.plugin import MANIFEST, backend_factory, plugin


def test_plugin_manifest_matches_hub_expectations() -> None:
    manifest = plugin.get_plugin_manifest()

    assert manifest is MANIFEST
    assert manifest["plugin_id"] == "hloc"
    assert manifest["package_name"] == "sfmapi-hloc"
    assert manifest["entry_points"] == ["sfmapi_hloc.plugin:plugin"]
    assert [provider["provider_id"] for provider in manifest["providers"]] == ["hloc"]
    assert manifest["providers"][0]["backend_actions"] == ["hloc.*"]


def test_plugin_manifest_validates_against_sfm_hub_if_available() -> None:
    sfm_hub_models = pytest.importorskip("sfm_hub.models")

    validated = sfm_hub_models.PluginManifest.model_validate(MANIFEST)

    assert validated.plugin_id == "hloc"
    assert validated.provider_ids() == ["hloc"]


def test_pyproject_declares_importable_sfmapi_backend_entry_point() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    entry_points = pyproject["project"]["entry-points"]["sfmapi.backends"]

    assert entry_points == {"hloc": "sfmapi_hloc.plugin:plugin"}

    module_name, object_name = entry_points["hloc"].split(":", 1)
    loaded = getattr(import_module(module_name), object_name)

    assert loaded is plugin
    assert loaded.get_plugin_manifest()["plugin_id"] == "hloc"


def test_installed_entry_point_metadata_loads_plugin_object() -> None:
    entry_points = [
        entry_point
        for entry_point in metadata.entry_points().select(group="sfmapi.backends")
        if entry_point.name == "hloc" and entry_point.value == "sfmapi_hloc.plugin:plugin"
    ]
    if not entry_points:
        pytest.skip("sfmapi-hloc is not installed with entry-point metadata")

    assert entry_points[0].load() is plugin


def test_plugin_registers_hloc_backend_factory() -> None:
    registered: dict[str, Callable[[], Any]] = {}

    plugin.register(registered.__setitem__)

    assert set(registered) == {"hloc"}
    assert isinstance(registered["hloc"](), HlocBackend)


def test_backend_factory_returns_hloc_backend() -> None:
    assert isinstance(backend_factory(), HlocBackend)
