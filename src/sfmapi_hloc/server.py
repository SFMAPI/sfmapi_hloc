"""ASGI entrypoint that registers the HLOC backend before sfmapi starts."""
# ruff: noqa: E402

from __future__ import annotations

import os

os.environ.setdefault("SFMAPI_EPHEMERAL", "true")
os.environ.setdefault("SFMAPI_BACKEND", "hloc")
os.environ.setdefault("SFMAPI_DB_URL", "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true")
os.environ.setdefault("SFMAPI_BLOB_BACKEND", "memory")
os.environ.setdefault("SFMAPI_QUEUE_BACKEND", "inline")
os.environ.setdefault("SFMAPI_INLINE_TASKS", "true")

from sfmapi_hloc.backend import configure_hloc_environment
from sfmapi_hloc.plugin import plugin

configure_hloc_environment(validate=bool(os.environ.get("SFMAPI_HLOC_ROOT")))

from sfmapi.runtime import create_app, register_backend

plugin.register(register_backend)

app = create_app()
