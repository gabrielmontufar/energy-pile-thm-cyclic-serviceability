from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
FILES = [
    "cyclic_law_model_selection.csv",
    "qiu2025_fig21_tidy.csv",
    "rafai2025_holdout_metrics.csv",
    "scenario_summary.csv",
    "uncertainty_summary.csv",
]


def committed_csv(name: str) -> pd.DataFrame:
    raw = subprocess.check_output(
        ["git", "show", f"HEAD:outputs/tables/{name}"], cwd=ROOT, text=True, encoding="utf-8"
    )
    return pd.read_csv(StringIO(raw))


for name in FILES:
    reference = committed_csv(name)
    regenerated = pd.read_csv(TABLES / name)
    if list(reference.columns) != list(regenerated.columns) or reference.shape != regenerated.shape:
        raise AssertionError(f"Schema or shape drift in {name}")
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            if not np.allclose(
                reference[column].to_numpy(float),
                regenerated[column].to_numpy(float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise AssertionError(f"Numeric drift in {name}:{column}")
        elif not reference[column].fillna("").equals(regenerated[column].fillna("")):
            raise AssertionError(f"Text drift in {name}:{column}")

for stem in range(1, 6):
    for suffix in ("pdf", "png"):
        candidates = list((ROOT / "outputs" / "figures").glob(f"figure_{stem}_*.{suffix}"))
        if len(candidates) != 1 or candidates[0].stat().st_size < 1000:
            raise AssertionError(f"Missing or empty figure {stem}.{suffix}")

print("Cross-platform reproduction validated within numerical tolerances")
