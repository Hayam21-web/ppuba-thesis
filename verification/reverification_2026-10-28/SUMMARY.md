# Clean re-run of release v1.4-thesis — 28 October 2026

**Source:** this package (release `v1.4-thesis`). The pipeline files (scripts 00–09, `ppuba_lib/`, `config.py`,
`requirements.txt`, `run_all.py`, `verify_thesis_values.py`, `calibrated_intensity.json`) are byte-identical to release
`v1.0-thesis` (commit `7cf843dce8fe5f702b5f3359cc1ea32b07bf4abd`). `verification/verify_extended_values.py` is
byte-identical to the version in `v1.1-thesis` used for the 24 October 2026 re-run. The exploratory script
`10_external_validation.py` (v1.2/v1.3) is not part of the thesis and is not included.

**Environment:** fresh virtual environment, Python 3.13.18, pinned versions from `requirements.txt`
(NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, SHAP 0.53.0, Matplotlib 3.10.8, dp-accounting 0.6.0);
Opacus cross-check in a separate environment (Python 3.12, `verification/requirements_opacus.txt`: torch 2.5.1, Opacus 1.5.2).
`data/` and `outputs/` absent at the start (`rerun_env.txt`).

| Check | Script | Result |
|---|---|---|
| Pipeline, scripts 00–07 and 09 (seeds 42–46) | `run_all.py` | completed without error (`rerun_run_all.log`, 374 s) |
| Intensity sensitivity (seeds 42–44, scales 1 and 3) | `08_generator_sensitivity.py` | completed without error (`rerun_08.log`) |
| Table values (Tables 4.4, 5.1–5.4, C.2; Sec. 5.3.3; App. C) | `verify_thesis_values.py` | **183 / 183 match** (`rerun_verify.log`) |
| Values quoted in the running text (Ch. 1, 3–7; App. B–C) | `verification/verify_extended_values.py` | **150 / 150 match** (`rerun_verify_extended.log`) |
| DP accounting (Table D.2) | `verification/dp_accounting_check.py` | all noise multipliers reproduced; all accountant ε below nominal; `dpcheck.json` identical |
| DP implementation (Table D.3) | `verification/dp_implementation_check.py` | **27 / 27 tests passed**; `dp_implementation_check.json` identical |
| Cross-check against Opacus (Table D.5) | `verification/dp_opacus_crosscheck.py` | ALL PASSED; values identical to Table D.5 (differences ≤ 1e-17) |
| Empirical privacy audit (Table D.6) | `verification/dp_empirical_audit.py` | no violation in 6 configurations; `dp_empirical_audit.json` and all `audit_stats/*.npz` arrays identical |
| Raw outputs vs. `ppuba_outputs_v1.0.zip` (SHA-256 `8ae80427…bf853`) | `compare_outputs_vs_v1.0.txt` | 54 / 100 files byte-identical; the rest differ only in timing fields, absolute paths, the Python patch version, xlsx metadata, and round-off ≤ 3 × 10⁻¹⁰ in centralized-LR scores/thresholds; no reported value affected |
| Figures | `compare_outputs_vs_v1.0.txt` | 9 / 10 PNG files pixel-identical; the SHAP beeswarm (Figure 5.6) differs only in the random vertical jitter of the points |

These are tests and empirical audits; they are not a formal verification of differential privacy.
