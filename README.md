# Reduced-order THM and cyclic-settlement screening for energy piles

This repository reproduces the calculations, tables and algorithmic figures supporting the manuscript **“Reduced-Order Thermo-Hydro-Mechanical Model for Cyclic Settlement and Serviceability Screening of Energy Piles.”**

## Reproduce

Use Python 3.11 or newer:

```powershell
python -m pip install -r requirements.txt
python run_tests.py
$env:PYTHONPATH='src'
python -m energy_pile_screening.pipeline
```

The pipeline writes all numerical results to `outputs/tables`, figures in PDF and 600-dpi PNG to `outputs/figures`, and a machine-readable run record to `outputs/run_metadata.json`.

## Evidence boundaries

- The pore-pressure equation is a reduced, linear, saturated thermo-poroelastic screen. Its analytical solution and limiting cases are verified; it is not presented as a replacement for a site-calibrated nonlinear THM finite-element model.
- The cyclic relation is an empirical envelope calibrated to the five load levels in Qiu et al. (2025). Model selection uses leave-one-load-level-out cross-validation and the one-standard-error rule.
- Rafai et al. (2025) is retained as an external transferability check and is never used to fit the parameters.
- The building, bridge and equipment cases are illustrative design archetypes, not field validations.

## Data

`data/raw/qiu2025_fig21.xlsx` is the authors' open workbook from Zenodo record 17122550 (CC BY 4.0). Its SHA-256 digest is recorded in `outputs/run_metadata.json`. The compact Rafai holdout table transcribes numerical values stated in the open-access article, with DOI and role recorded in every row.

## Licenses

Code is released under the MIT License. Original third-party data remain under their source licenses and must be cited. Manuscript text and author-generated figures are intended for release under CC BY 4.0 after journal-submission checks.
