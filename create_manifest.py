from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "manuscript", "submission"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name == "MANIFEST.sha256":
        continue
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        continue
    rows.append(f"{sha256(path)}  {relative.as_posix()}")
(ROOT / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"Hashed {len(rows)} files")
