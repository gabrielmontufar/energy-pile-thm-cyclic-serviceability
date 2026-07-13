from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from .calibration import CyclicLawParameters, predict_cyclic_settlement


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class LayerParameters:
    name: str
    thickness_m: float
    mv_1_per_kpa: float
    cv_m2_s: float
    lambda_kpa_per_c: float
    drainage_path_m: float
    temperature_factor: float
    provenance: str

    @property
    def tau_days(self) -> float:
        """First Terzaghi mode: exp[-pi^2 cv t/(4 Hdr^2)]."""
        return 4.0 * self.drainage_path_m**2 / (math.pi**2 * self.cv_m2_s) / SECONDS_PER_DAY


@dataclass(frozen=True)
class PileScenario:
    name: str
    service_load_kn: float
    ultimate_capacity_kn: float
    vertical_stiffness_kn_m: float
    head_stiffness_kn_m: float
    allowable_settlement_mm: float
    thermal_amplitude_c: float
    cycles: int
    pile_length_m: float = 20.0
    pile_diameter_m: float = 0.8
    pile_modulus_kpa: float = 30.0e6
    pile_alpha_1_c: float = 10.0e-6
    provenance: str = "illustrative screening archetype; not a case-history validation"

    @property
    def load_ratio(self) -> float:
        return self.service_load_kn / self.ultimate_capacity_kn


def exact_pore_pressure_response(
    temperature_c: np.ndarray,
    time_days: np.ndarray,
    lambda_kpa_per_c: float,
    tau_days: float,
) -> np.ndarray:
    """Exact step update for du/dt + u/tau = Lambda dT/dt with linear T per step."""
    t = np.asarray(time_days, dtype=float)
    temp = np.asarray(temperature_c, dtype=float)
    if t.shape != temp.shape or np.any(np.diff(t) <= 0.0):
        raise ValueError("time and temperature must have equal shapes and increasing time")
    u = np.zeros_like(temp)
    if lambda_kpa_per_c == 0.0:
        return u
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        decay = math.exp(-dt / tau_days)
        ramp = (temp[i] - temp[i - 1]) / dt
        u[i] = u[i - 1] * decay + lambda_kpa_per_c * ramp * tau_days * (1.0 - decay)
    return u


def thermal_history(cycles: int, amplitude_c: float, points_per_cycle: int = 181) -> tuple[np.ndarray, np.ndarray]:
    days_per_cycle = 365.25
    total_days = cycles * days_per_cycle
    time = np.linspace(0.0, total_days, cycles * points_per_cycle + 1)
    temperature = amplitude_c * np.sin(2.0 * np.pi * time / days_per_cycle)
    temperature[-1] = 0.0
    return time, temperature


def completed_cycle_count(time_days: np.ndarray) -> np.ndarray:
    return np.floor(np.asarray(time_days) / 365.25 + 1e-10)


def simulate_scenario(
    scenario: PileScenario,
    layers: list[LayerParameters],
    cyclic_law: CyclicLawParameters,
    points_per_cycle: int = 181,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_days, head_temperature = thermal_history(
        scenario.cycles, scenario.thermal_amplitude_c, points_per_cycle=points_per_cycle
    )
    area = math.pi * scenario.pile_diameter_m**2 / 4.0
    pile_stiffness_kn_m = scenario.pile_modulus_kpa * area / scenario.pile_length_m
    restraint = scenario.head_stiffness_kn_m / (scenario.head_stiffness_kn_m + pile_stiffness_kn_m)
    mechanical_mm = 1000.0 * scenario.service_load_kn / scenario.vertical_stiffness_kn_m
    free_thermal_mm = 1000.0 * scenario.pile_alpha_1_c * scenario.pile_length_m * head_temperature
    thermal_recoverable_mm = (1.0 - restraint) * free_thermal_mm

    hydraulic_transient_mm = np.zeros_like(time_days)
    pressure_columns: dict[str, np.ndarray] = {}
    for layer in layers:
        layer_temperature = layer.temperature_factor * head_temperature
        pressure = exact_pore_pressure_response(
            layer_temperature, time_days, layer.lambda_kpa_per_c, layer.tau_days
        )
        # Signed drainage strain relative to the undrained thermo-poroelastic reference.
        undrained_pressure = layer.lambda_kpa_per_c * layer_temperature
        hydraulic_transient_mm += (
            1000.0 * layer.mv_1_per_kpa * layer.thickness_m * (undrained_pressure - pressure)
        )
        pressure_columns[f"u_{layer.name}_kpa"] = pressure

    n_completed = completed_cycle_count(time_days)
    cyclic_residual_mm = predict_cyclic_settlement(scenario.load_ratio, n_completed, cyclic_law)
    total_mm = mechanical_mm + thermal_recoverable_mm + hydraulic_transient_mm + cyclic_residual_mm
    frame = pd.DataFrame(
        {
            "time_days": time_days,
            "temperature_change_c": head_temperature,
            "completed_cycles": n_completed,
            "mechanical_mm": mechanical_mm,
            "thermal_recoverable_mm": thermal_recoverable_mm,
            "hydraulic_transient_mm": hydraulic_transient_mm,
            "cyclic_residual_mm": cyclic_residual_mm,
            "total_settlement_mm": total_mm,
            "allowable_settlement_mm": scenario.allowable_settlement_mm,
            **pressure_columns,
        }
    )
    last = frame.iloc[-1]
    summary = pd.DataFrame(
        [
            {
                "scenario": scenario.name,
                "service_load_kn": scenario.service_load_kn,
                "ultimate_capacity_kn": scenario.ultimate_capacity_kn,
                "load_ratio": scenario.load_ratio,
                "cycles": scenario.cycles,
                "thermal_amplitude_c": scenario.thermal_amplitude_c,
                "allowable_settlement_mm": scenario.allowable_settlement_mm,
                "initial_mechanical_mm": mechanical_mm,
                "peak_total_mm": float(frame["total_settlement_mm"].max()),
                "minimum_total_mm": float(frame["total_settlement_mm"].min()),
                "peak_abs_hydraulic_transient_mm": float(frame["hydraulic_transient_mm"].abs().max()),
                "residual_cyclic_mm": float(last["cyclic_residual_mm"]),
                "end_cycle_total_mm": float(last["total_settlement_mm"]),
                "fully_drained_residual_mm": float(mechanical_mm + last["cyclic_residual_mm"]),
                "peak_serviceability_ratio": float(frame["total_settlement_mm"].max() / scenario.allowable_settlement_mm),
                "fully_drained_residual_serviceability_ratio": float(
                    (mechanical_mm + last["cyclic_residual_mm"]) / scenario.allowable_settlement_mm
                ),
                "scenario_status": "illustrative screening archetype",
            }
        ]
    )
    return frame, summary
