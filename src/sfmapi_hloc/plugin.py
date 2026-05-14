from __future__ import annotations

from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

from .backend import HlocBackend


class ProviderManifest(TypedDict):
    provider_id: str
    display_name: str
    capabilities: list[str]
    backend_actions: list[str]
    priority_hint: int


class UvRuntime(TypedDict):
    source: str
    url: str
    ref: str
    package: str


class RuntimeModes(TypedDict):
    uv: UvRuntime
    docker: dict[str, Any]


class Compatibility(TypedDict):
    sfmapi: str
    python: str
    os: list[str]
    cuda: str


class Conformance(TypedDict):
    status: str
    suite: str


class LicenseInfo(TypedDict):
    name: str


class UpstreamProject(TypedDict):
    name: str
    url: str
    license: NotRequired[str]


class PluginManifest(TypedDict):
    plugin_id: str
    display_name: str
    description: str
    package_name: str
    github_url: str
    entry_points: list[str]
    providers: list[ProviderManifest]
    runtime_modes: RuntimeModes
    capabilities: list[str]
    backend_actions: list[str]
    config_schemas: list[str]
    artifact_contracts: list[str]
    licenses: list[LicenseInfo]
    upstream_projects: list[UpstreamProject]
    compatibility: Compatibility
    conformance: Conformance
    trust_tier: str


# Must stay identical to ``HlocBackend.capabilities()`` — every entry is
# a portable sfmapi capability backed by a real runner.py wrapper.
# ``matches.verify`` is intentionally omitted: hloc has no standalone
# geometric-verification stage (see ``HlocBackend.verify_matches``).
CAPABILITIES = list(HlocBackend.PORTABLE_CAPABILITIES)

# Must stay identical to the ids emitted by
# ``HlocBackend.list_backend_config_schemas`` and
# ``HlocBackend.list_backend_artifact_contracts``.
CONFIG_SCHEMAS = ["hloc.features", "hloc.matcher", "hloc.pairs.retrieval"]
ARTIFACT_CONTRACTS = ["hloc.features", "hloc.matches", "hloc.pairs"]

MANIFEST: PluginManifest = {
    "plugin_id": "hloc",
    "display_name": "Hierarchical Localization",
    "description": "Backend plugin for hloc feature, retrieval, matching, and localization workflows.",
    "package_name": "sfmapi-hloc",
    "github_url": "https://github.com/SFMAPI/sfmapi_hloc.git",
    "entry_points": ["sfmapi_hloc.plugin:plugin"],
    "providers": [
        {
            "provider_id": "hloc",
            "display_name": "hloc",
            "capabilities": CAPABILITIES,
            "backend_actions": ["hloc.*"],
            "priority_hint": 10,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SFMAPI/sfmapi_hloc.git",
            "ref": "main",
            "package": "sfmapi-hloc",
        },
        "docker": {},
    },
    "capabilities": CAPABILITIES,
    "backend_actions": ["hloc.*"],
    "config_schemas": CONFIG_SCHEMAS,
    "artifact_contracts": ARTIFACT_CONTRACTS,
    "licenses": [{"name": "AGPL-3.0-or-later"}],
    "upstream_projects": [
        {
            "name": "Hierarchical Localization",
            "url": "https://github.com/cvg/Hierarchical-Localization",
            "license": "Apache-2.0",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.12,<3.13",
        "os": ["windows", "linux"],
        "cuda": "recommended",
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "official",
}


def backend_factory() -> HlocBackend:
    return HlocBackend()


class SfmapiHlocPlugin:
    backend_name = "hloc"
    backend_factory = staticmethod(backend_factory)
    manifest = MANIFEST

    def get_plugin_manifest(self) -> PluginManifest:
        return self.manifest

    def register(self, register_backend: Callable[..., None]) -> None:
        provider_ids = [str(provider["provider_id"]) for provider in self.manifest["providers"]]
        try:
            register_backend(self.backend_name, self.backend_factory, providers=provider_ids)
        except TypeError:
            # Older sfmapi without ``providers=`` kwarg on the registrar.
            register_backend(self.backend_name, self.backend_factory)


plugin = SfmapiHlocPlugin()


def get_plugin_manifest() -> PluginManifest:
    return MANIFEST


def register(register_backend: Callable[..., None]) -> None:
    plugin.register(register_backend)


__all__ = [
    "ARTIFACT_CONTRACTS",
    "CAPABILITIES",
    "CONFIG_SCHEMAS",
    "MANIFEST",
    "PluginManifest",
    "SfmapiHlocPlugin",
    "backend_factory",
    "get_plugin_manifest",
    "plugin",
    "register",
]
