"""ASGI entrypoint that registers the HLOC backend before sfmapi starts."""
# ruff: noqa: E402,I001

from __future__ import annotations

import os

os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "hloc")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sfmapi_hloc.backend import configure_hloc_environment

configure_hloc_environment(validate=bool(os.environ.get("SFMAPI_HLOC_ROOT")))

import sfmapi_hloc  # noqa: F401 - import side effect registers backend

from app.main import create_app

app = create_app()
