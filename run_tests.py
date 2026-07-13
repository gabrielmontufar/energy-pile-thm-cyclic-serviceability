"""Dependency-light test runner for clean-environment reproduction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    failures = 0
    executed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            executed += 1
            try:
                getattr(module, name)()
                print(f"PASS {path.name}::{name}")
            except Exception:
                failures += 1
                print(f"FAIL {path.name}::{name}")
                traceback.print_exc()
    print(f"Executed {executed} tests; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
