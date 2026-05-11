from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sfmapi_hloc.runner import run_pairs_poses


def test_pairs_poses_rejects_impossible_neighbor_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pairs_from_poses = SimpleNamespace(
        main=lambda *args, **kwargs: pytest.fail("upstream runner should not be called")
    )
    hloc = ModuleType("hloc")
    hloc.pairs_from_poses = pairs_from_poses
    utils = ModuleType("hloc.utils")
    read_write_model = ModuleType("hloc.utils.read_write_model")
    read_write_model.read_images_binary = lambda path: {1: object()}

    monkeypatch.setitem(sys.modules, "hloc", hloc)
    monkeypatch.setitem(sys.modules, "hloc.utils", utils)
    monkeypatch.setitem(sys.modules, "hloc.utils.read_write_model", read_write_model)

    with pytest.raises(ValueError, match="num_matched must be smaller"):
        run_pairs_poses(
            {
                "model_path": str(tmp_path / "model"),
                "pairs_path": str(tmp_path / "pairs.txt"),
                "num_matched": 1,
            }
        )
