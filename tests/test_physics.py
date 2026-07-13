from pathlib import Path

import numpy as np

from energy_pile_screening.calibration import fit_and_select_cyclic_law, load_qiu_fig21
from energy_pile_screening.model import (
    LayerParameters,
    PileScenario,
    exact_pore_pressure_response,
    simulate_scenario,
)


ROOT = Path(__file__).resolve().parents[1]


def _law():
    data = load_qiu_fig21(ROOT / "data" / "raw" / "qiu2025_fig21.xlsx")
    return fit_and_select_cyclic_law(data)[0]


def _layer(lambda_value: float = 7.0, cv: float = 1e-7) -> LayerParameters:
    return LayerParameters("test", 10.0, 3e-5, cv, lambda_value, 2.0, 1.0, "test")


def _scenario(load: float = 300.0, cycles: int = 3, amplitude: float = 10.0) -> PileScenario:
    return PileScenario("test", load, 1000.0, 100000.0, 200000.0, 25.0, amplitude, cycles)


def test_closed_form_constant_ramp() -> None:
    time = np.linspace(0.0, 100.0, 201)
    rate = 0.08
    temp = rate * time
    tau = 35.0
    lam = 6.0
    numerical = exact_pore_pressure_response(temp, time, lam, tau)
    analytical = lam * rate * tau * (1.0 - np.exp(-time / tau))
    assert np.max(np.abs(numerical - analytical)) < 1e-10


def test_zero_thermal_pressurization_gives_zero_pressure() -> None:
    time = np.linspace(0.0, 10.0, 101)
    temp = np.sin(time)
    assert np.allclose(exact_pore_pressure_response(temp, time, 0.0, 10.0), 0.0)


def test_undrained_limit_approaches_lambda_delta_t() -> None:
    time = np.linspace(0.0, 1.0, 101)
    temp = 5.0 * time
    u = exact_pore_pressure_response(temp, time, 8.0, 1e8)
    assert np.isclose(u[-1], 40.0, rtol=1e-5)


def test_zero_temperature_removes_thermal_and_hydraulic_terms() -> None:
    frame, _ = simulate_scenario(_scenario(amplitude=0.0), [_layer()], _law())
    assert np.allclose(frame.thermal_recoverable_mm, 0.0)
    assert np.allclose(frame.hydraulic_transient_mm, 0.0)


def test_zero_cycles_and_zero_load_have_no_cyclic_residual() -> None:
    frame, _ = simulate_scenario(_scenario(load=0.0, cycles=0), [_layer()], _law())
    assert np.allclose(frame.cyclic_residual_mm, 0.0)


def test_residual_persists_after_temperature_recovery() -> None:
    frame, summary = simulate_scenario(_scenario(), [_layer()], _law())
    assert np.isclose(frame.temperature_change_c.iloc[-1], 0.0)
    assert frame.cyclic_residual_mm.iloc[-1] > 0.0
    assert np.isclose(summary.residual_cyclic_mm.iloc[0], frame.cyclic_residual_mm.iloc[-1])


def test_load_and_cycle_monotonicity() -> None:
    low, _ = simulate_scenario(_scenario(load=300.0, cycles=3), [_layer()], _law())
    high, _ = simulate_scenario(_scenario(load=600.0, cycles=3), [_layer()], _law())
    more, _ = simulate_scenario(_scenario(load=300.0, cycles=5), [_layer()], _law())
    assert high.cyclic_residual_mm.iloc[-1] >= low.cyclic_residual_mm.iloc[-1]
    assert more.cyclic_residual_mm.iloc[-1] >= low.cyclic_residual_mm.iloc[-1]


def test_time_step_convergence() -> None:
    coarse, _ = simulate_scenario(_scenario(), [_layer()], _law(), points_per_cycle=73)
    fine, _ = simulate_scenario(_scenario(), [_layer()], _law(), points_per_cycle=365)
    assert abs(coarse.total_settlement_mm.max() - fine.total_settlement_mm.max()) < 0.05
