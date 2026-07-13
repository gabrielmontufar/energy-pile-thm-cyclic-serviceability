from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import platform
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import openpyxl
import pandas as pd

from .calibration import (
    CyclicLawParameters,
    fit_and_select_cyclic_law,
    load_qiu_fig21,
    predict_cyclic_settlement,
    residual_bootstrap_predictions,
)
from .model import LayerParameters, PileScenario, exact_pore_pressure_response, simulate_scenario


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
EXTERNAL = ROOT / "data" / "external"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"


LAYERS = [
    LayerParameters(
        "upper_silty_clay", 6.0, 4.5e-5, 2.0e-8, 10.0, 2.0, 1.00,
        "illustrative fine-grained layer; mv, cv and Lambda varied in uncertainty analysis",
    ),
    LayerParameters(
        "sandy_silt", 7.0, 2.5e-5, 1.2e-7, 6.0, 2.0, 0.96,
        "illustrative transitional layer; mv, cv and Lambda varied in uncertainty analysis",
    ),
    LayerParameters(
        "dense_sand", 7.0, 1.2e-5, 7.0e-7, 3.0, 2.0, 0.92,
        "illustrative granular layer; mv, cv and Lambda varied in uncertainty analysis",
    ),
]


SCENARIOS = [
    PileScenario("Building foundation", 3000.0, 7500.0, 260000.0, 350000.0, 25.0, 9.0, 10),
    PileScenario("Bridge-abutment retrofit", 4200.0, 8400.0, 360000.0, 550000.0, 15.0, 8.0, 10),
    PileScenario("Equipment-supported mat", 2200.0, 6285.7142857, 520000.0, 800000.0, 10.0, 7.0, 10),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _uncertainty(
    scenario: PileScenario,
    law: CyclicLawParameters,
    n: int = 600,
    seed: int = 20260713,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + int(scenario.service_load_kn))
    records: list[dict[str, float | str]] = []
    for sample in range(n):
        load_factor = rng.uniform(0.90, 1.10)
        q_ult_factor = rng.uniform(0.90, 1.10)
        thermal_factor = rng.uniform(0.80, 1.20)
        stiffness_factor = np.exp(rng.uniform(np.log(0.70), np.log(1.30)))
        mv_factor = rng.uniform(0.60, 1.40)
        cv_factor = np.exp(rng.uniform(np.log(0.30), np.log(3.00)))
        lambda_factor = rng.uniform(0.60, 1.40)
        sampled_scenario = replace(
            scenario,
            service_load_kn=scenario.service_load_kn * load_factor,
            ultimate_capacity_kn=scenario.ultimate_capacity_kn * q_ult_factor,
            thermal_amplitude_c=scenario.thermal_amplitude_c * thermal_factor,
            vertical_stiffness_kn_m=scenario.vertical_stiffness_kn_m * stiffness_factor,
        )
        sampled_layers = [
            replace(
                layer,
                mv_1_per_kpa=layer.mv_1_per_kpa * mv_factor,
                cv_m2_s=layer.cv_m2_s * cv_factor,
                lambda_kpa_per_c=layer.lambda_kpa_per_c * lambda_factor,
            )
            for layer in LAYERS
        ]
        frame, summary = simulate_scenario(sampled_scenario, sampled_layers, law, points_per_cycle=73)
        tm_peak = float((frame["mechanical_mm"] + frame["thermal_recoverable_mm"]).max())
        full_peak = float(summary.loc[0, "peak_total_mm"])
        records.append(
            {
                "scenario": scenario.name,
                "sample": sample,
                "load_ratio": sampled_scenario.load_ratio,
                "thermal_amplitude_c": sampled_scenario.thermal_amplitude_c,
                "mv_factor": mv_factor,
                "cv_factor": cv_factor,
                "lambda_factor": lambda_factor,
                "tm_peak_mm": tm_peak,
                "full_peak_mm": full_peak,
                "residual_mm": float(summary.loc[0, "fully_drained_residual_mm"]),
                "full_exceeds": full_peak > scenario.allowable_settlement_mm,
                "false_safe": tm_peak <= scenario.allowable_settlement_mm < full_peak,
            }
        )
    samples = pd.DataFrame(records)
    stats = pd.DataFrame(
        [
            {
                "scenario": scenario.name,
                "samples": n,
                "peak_p05_mm": samples["full_peak_mm"].quantile(0.05),
                "peak_p50_mm": samples["full_peak_mm"].quantile(0.50),
                "peak_p95_mm": samples["full_peak_mm"].quantile(0.95),
                "residual_p50_mm": samples["residual_mm"].quantile(0.50),
                "residual_p95_mm": samples["residual_mm"].quantile(0.95),
                "probability_exceedance": samples["full_exceeds"].mean(),
                "probability_false_safe": samples["false_safe"].mean(),
            }
        ]
    )
    return samples, stats


def run() -> None:
    for folder in (FIGURES, TABLES):
        folder.mkdir(parents=True, exist_ok=True)
    qiu_path = RAW / "qiu2025_fig21.xlsx"
    qiu = load_qiu_fig21(qiu_path)
    qiu.to_csv(TABLES / "qiu2025_fig21_tidy.csv", index=False)
    selected, comparison = fit_and_select_cyclic_law(qiu)
    comparison.to_csv(TABLES / "cyclic_law_model_selection.csv", index=False)

    qiu_grid = pd.DataFrame(
        [(r, n) for r in sorted(qiu.load_ratio.unique()) for n in np.linspace(0, 30, 121)],
        columns=["load_ratio", "cycle"],
    )
    intervals = residual_bootstrap_predictions(qiu, selected, qiu_grid)
    intervals.to_csv(TABLES / "qiu2025_cyclic_predictions_with_intervals.csv", index=False)

    rafai = pd.read_csv(EXTERNAL / "rafai2025_holdout.csv")
    rafai["predicted_mm"] = predict_cyclic_settlement(
        rafai["load_ratio"].to_numpy(), rafai["cycles"].to_numpy(), selected
    )
    rafai["error_mm"] = rafai["predicted_mm"] - rafai["observed_residual_mm"]
    rafai.to_csv(TABLES / "rafai2025_external_transfer_check.csv", index=False)
    holdout_metrics = pd.DataFrame(
        [
            {
                "rmse_mm": float(np.sqrt(np.mean(rafai.error_mm**2))),
                "mae_mm": float(np.mean(np.abs(rafai.error_mm))),
                "maximum_absolute_error_mm": float(np.max(np.abs(rafai.error_mm))),
                "status": "external transferability check; not recalibration",
            }
        ]
    )
    holdout_metrics.to_csv(TABLES / "rafai2025_holdout_metrics.csv", index=False)

    frames: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    uncertainty_samples: list[pd.DataFrame] = []
    uncertainty_stats: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        frame, summary = simulate_scenario(scenario, LAYERS, selected)
        frame.insert(0, "scenario", scenario.name)
        frames.append(frame)
        summaries.append(summary)
        samples, stats = _uncertainty(scenario, selected)
        uncertainty_samples.append(samples)
        uncertainty_stats.append(stats)
    histories = pd.concat(frames, ignore_index=True)
    scenario_summary = pd.concat(summaries, ignore_index=True)
    uncertainty = pd.concat(uncertainty_samples, ignore_index=True)
    uncertainty_summary = pd.concat(uncertainty_stats, ignore_index=True)
    histories.to_csv(TABLES / "scenario_time_histories.csv", index=False)
    scenario_summary.to_csv(TABLES / "scenario_summary.csv", index=False)
    uncertainty.to_csv(TABLES / "uncertainty_samples.csv", index=False)
    uncertainty_summary.to_csv(TABLES / "uncertainty_summary.csv", index=False)

    # Closed-form verification for a constant temperature ramp.
    verification_time = np.linspace(0.0, 120.0, 241)
    ramp_rate = 0.05
    verification_temp = ramp_rate * verification_time
    tau = 40.0
    lam = 7.0
    numerical = exact_pore_pressure_response(verification_temp, verification_time, lam, tau)
    analytical = lam * ramp_rate * tau * (1.0 - np.exp(-verification_time / tau))
    verification = pd.DataFrame(
        {
            "time_days": verification_time,
            "numerical_kpa": numerical,
            "analytical_kpa": analytical,
            "absolute_error_kpa": np.abs(numerical - analytical),
        }
    )
    verification.to_csv(TABLES / "hydraulic_closed_form_verification.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for color, level in zip(colors, sorted(qiu.load_ratio.unique())):
        obs = qiu[qiu.load_ratio == level]
        pred = intervals[intervals.load_ratio == level]
        ax.scatter(obs.cycle, obs.settlement_mm, s=18, color=color, label=f"Observed {level:.0%}")
        ax.plot(pred.cycle, pred.predicted_mm, color=color)
        ax.fill_between(pred.cycle, pred.lower95_mm, pred.upper95_mm, color=color, alpha=0.12)
    ax.set(xlabel="Completed thermal cycles, N", ylabel="Residual settlement (mm)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    _save_figure(fig, "figure_1_qiu_calibration")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(rafai.observed_residual_mm, rafai.predicted_mm, s=55, c=rafai.load_ratio, cmap="plasma")
    limit = max(rafai.observed_residual_mm.max(), rafai.predicted_mm.max()) * 1.08
    ax.plot([0, limit], [0, limit], "k--", lw=1)
    for row in rafai.itertuples():
        ax.annotate(f"{row.load_ratio:.0%}", (row.observed_residual_mm, row.predicted_mm), xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Rafai et al. observed residual settlement (mm)", ylabel="Qiu-calibrated prediction (mm)", xlim=(0, limit), ylim=(0, limit))
    ax.grid(alpha=0.25)
    _save_figure(fig, "figure_2_external_transfer")

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    for ax, scenario in zip(axes, SCENARIOS):
        data = histories[histories.scenario == scenario.name]
        ax.plot(data.time_days / 365.25, data.total_settlement_mm, label="Total")
        ax.plot(data.time_days / 365.25, data.mechanical_mm + data.thermal_recoverable_mm, label="TM", alpha=0.8)
        ax.plot(data.time_days / 365.25, data.cyclic_residual_mm, label="Cyclic residual", alpha=0.8)
        ax.axhline(scenario.allowable_settlement_mm, color="crimson", ls="--", lw=1, label="Screening limit")
        ax.set_ylabel("Settlement (mm)")
        ax.set_title(scenario.name, loc="left", fontsize=10)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (years)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, "figure_3_scenario_decomposition")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(uncertainty_summary))
    med = uncertainty_summary.peak_p50_mm.to_numpy()
    lower = med - uncertainty_summary.peak_p05_mm.to_numpy()
    upper = uncertainty_summary.peak_p95_mm.to_numpy() - med
    ax.errorbar(x, med, yerr=np.vstack([lower, upper]), fmt="o", capsize=5, label="Peak: median and 5–95%")
    ax.scatter(x, uncertainty_summary.residual_p50_mm, marker="s", label="Residual median")
    ax.set_xticks(x, uncertainty_summary.scenario, rotation=12, ha="right")
    ax.set_ylabel("Settlement (mm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save_figure(fig, "figure_4_uncertainty")

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(verification.time_days, verification.analytical_kpa, label="Closed form", lw=2)
    ax.plot(verification.time_days, verification.numerical_kpa, "--", label="Exact step update")
    ax.set(xlabel="Time (days)", ylabel="Excess pore pressure (kPa)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, "figure_5_hydraulic_verification")

    metadata = {
        "qiu_workbook_sha256": _sha256(qiu_path),
        "qiu_levels": sorted(float(v) for v in qiu.load_ratio.unique()),
        "qiu_points": int(len(qiu)),
        "qiu_maximum_mm": float(qiu.settlement_mm.max()),
        "selected_cyclic_law": asdict(selected),
        "hydraulic_closed_form_max_error_kpa": float(verification.absolute_error_kpa.max()),
        "rafai_holdout": holdout_metrics.iloc[0].to_dict(),
        "software_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "openpyxl": openpyxl.__version__,
        },
        "claim_boundary": (
            "The cyclic relation is an empirical screening envelope calibrated to Qiu et al.; "
            "Rafai et al. is an external transferability check. The three infrastructure cases are illustrative."
        ),
    }
    (OUTPUTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
