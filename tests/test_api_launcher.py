from __future__ import annotations

from pathlib import Path

from sfmapi_hloc.api_launcher import build_parser, configure_environment


def _fake_hloc(root: Path) -> Path:
    (root / "hloc").mkdir(parents=True, exist_ok=True)
    (root / "setup.py").write_text("setup(name='hloc')\n", encoding="utf-8")
    return root


def test_configure_environment_sets_in_memory_sfmapi_defaults(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = _fake_hloc(tmp_path / "Hierarchical-Localization")
    parser = build_parser()
    args = parser.parse_args(["--hloc-root", str(root), "--python", "python", "--mcp", "local"])

    selected = configure_environment(args)

    assert selected == root.resolve()
    import os

    assert os.environ["SFMAPI_BACKEND"] == "hloc"
    assert os.environ["SFMAPI_HLOC_ROOT"] == str(root.resolve())
    assert os.environ["SFMAPI_HLOC_PYTHON"] == "python"
    assert os.environ["SFMAPI_MCP_MODE"] == "local"
    assert selected is not None
    assert selected.name == "Hierarchical-Localization"


def test_dry_run_uses_default_submodule_root(monkeypatch) -> None:
    monkeypatch.delenv("SFMAPI_HLOC_ROOT", raising=False)
    parser = build_parser()
    args = parser.parse_args(["--dry-run"])

    selected = configure_environment(args)

    assert selected is not None
    assert selected.name == "hloc"
