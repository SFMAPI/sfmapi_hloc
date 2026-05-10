from __future__ import annotations

import json

from .backend import HlocBackend


def main() -> None:
    backend = HlocBackend()
    print(json.dumps(backend.runtime_versions(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
