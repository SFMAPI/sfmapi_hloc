# sfmapi HLOC backend

This package runs sfmapi with a Hierarchical Localization (HLOC) action catalog.
The wrapper is AGPL-3.0-or-later; the upstream HLOC project is included as a git
submodule under `third_party/hloc` and keeps its Apache-2.0 license.

The integration is intentionally action-based. HLOC stores features and matches
in HDF5 files and uses pycolmap for reconstruction/localization, so this wrapper
does not advertise portable sfmapi stage capabilities until those artifacts are
normalized into sfmapi resources. Discover and run native actions from
`/v1/backend/actions`.

## Layout

- `src/sfmapi_hloc/`: backend adapter, runner, and sfmapi launcher.
- `third_party/hloc/`: upstream HLOC submodule.
- `tests/`: lightweight contract and HTTP discovery tests.
- `LICENSES/`: copied upstream license notice.

## Setup

```powershell
git submodule update --init --recursive
uv venv
uv sync --extra dev --extra mcp --with-editable ..\sfmapi
```

Install HLOC's runtime dependencies in the Python environment used for real
jobs:

```powershell
uv pip install -e .\third_party\hloc
```

HLOC workflows commonly require PyTorch, pycolmap, model weights, and optionally
CUDA.

## Run sfmapi

```powershell
uv run sfmapi-hloc-api --hloc-root .\third_party\hloc --mcp local
```

Useful environment variables:

- `SFMAPI_HLOC_ROOT`: path to the upstream HLOC checkout.
- `SFMAPI_HLOC_PYTHON`: Python executable used to run HLOC modules.
- `SFMAPI_MCP_MODE=local`: mount sfmapi's MCP endpoint at `/mcp`.

The launcher configures an in-memory sfmapi demo server: SQLite memory DB,
memory blob storage, inline queue, and inline tasks.

## Native Actions

Discover actions:

```powershell
curl "http://127.0.0.1:8000/v1/backend/actions?include_schemas=true"
```

Primary actions:

- `hloc.extractFeatures`: run `hloc.extract_features`.
- `hloc.pairsExhaustive`: create exhaustive image pairs.
- `hloc.pairsRetrieval`: create pairs from global retrieval descriptors.
- `hloc.pairsCovisibility`: create pairs from a reference model's covisibility graph.
- `hloc.pairsPoses`: create pairs from pose proximity.
- `hloc.matchFeatures`: run sparse feature matching.
- `hloc.matchDense`: run LoFTR-style dense matching.
- `hloc.reconstruct`: run HLOC reconstruction with pycolmap.
- `hloc.triangulate`: triangulate against an existing reference model.
- `hloc.localizeSfm`: localize query images against a reference model.
- `hloc.convertModel`: read/write COLMAP binary or text models.
- `hloc.listConfigs`: list extractor, sparse matcher, and dense matcher configs.
- `hloc.runPipeline`: extract features, create pairs, match, and reconstruct.
- `hloc.runModule`: run an allow-listed reusable HLOC module with explicit args.

Benchmark and dataset-specific HLOC scripts are not exposed as backend API
actions. Workflows such as Aachen, RobotCar, CMU, Cambridge, 7-Scenes,
4Seasons, InLoc, and NVM benchmark conversion depend on fixed dataset layouts
and should remain separate utility/evaluation code rather than normal API
features.

Example action input:

```json
{
  "image_dir": "C:/data/images",
  "outputs_dir": "C:/data/hloc-output",
  "pairing_mode": "exhaustive",
  "feature_conf": "superpoint_aachen",
  "matcher_conf": "superglue"
}
```

## Tests

```powershell
uv run pytest -q
uv run ruff check src tests
```

The default tests mock subprocess execution and do not require CUDA, pycolmap,
COLMAP, model weights, or HLOC runtime dependencies.

Use the standalone `sfmapi-bench` repository to verify a running API exposes
only reusable actions and keeps benchmark workflows out of `/v1/backend/actions`:

```powershell
uv run --project ..\sfmapi-bench sfmapi-bench api-surface --preset hloc --base-url http://127.0.0.1:8000
```
