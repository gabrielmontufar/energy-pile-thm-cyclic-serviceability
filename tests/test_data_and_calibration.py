from pathlib import Path

import numpy as np

from energy_pile_screening.calibration import (
    fit_and_select_cyclic_law,
    load_qiu_fig21,
    predict_cyclic_settlement,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qiu_extractor_reads_all_five_levels() -> None:
    data = load_qiu_fig21(ROOT / "data" / "raw" / "qiu2025_fig21.xlsx")
    assert set(data.load_ratio) == {0.33, 0.50, 0.60, 0.70, 0.80}
    assert data.groupby("load_ratio").size().eq(31).all()
    assert len(data) == 155
    assert np.isclose(data.settlement_mm.max(), 2.2940, atol=1e-4)


def test_selected_law_is_zero_at_zero_cycles_and_monotone() -> None:
    data = load_qiu_fig21(ROOT / "data" / "raw" / "qiu2025_fig21.xlsx")
    selected, _ = fit_and_select_cyclic_law(data)
    cycles = np.arange(0, 31)
    prediction = predict_cyclic_settlement(0.60, cycles, selected)
    assert np.isclose(prediction[0], 0.0)
    assert np.all(np.diff(prediction) >= -1e-12)
    loads = np.linspace(0.0, 0.9, 20)
    by_load = predict_cyclic_settlement(loads, 10.0, selected)
    assert np.all(np.diff(by_load) >= -1e-12)
