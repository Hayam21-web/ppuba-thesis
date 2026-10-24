# Independent clean re-run — 24 October 2026

**Source:** public GitHub repository `Hayam21-web/ppuba-thesis`, release tag `v1.0-thesis`
(commit `7cf843dce8fe5f702b5f3359cc1ea32b07bf4abd`); code identical to Supplementary Material S1 (only README wording differs).
**Environment:** fresh virtual environment, Python 3.13.17, pinned versions from `requirements.txt`
(NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, SciPy 1.17.1, SHAP 0.53.0, Matplotlib 3.10.8, dp-accounting 0.6.0).
**Procedure:** `data/` and `outputs/` absent; `python run_all.py` (scripts 00–07, 09; 6 min 33 s), then
`08_generator_sensitivity.py --seeds 42 43 44 --scales 1 3`, `verify_thesis_values.py`,
`verification/verify_extended_values.py`, `verification/dp_accounting_check.py`, `verification/dp_implementation_check.py`.

## Results

| Check | Result |
|---|---|
| Pipeline (00–09, 08) | completed without error (`rerun_run_all.log`, `rerun_08.log`) |
| Table values (Tables 4.4, 5.1–5.4, C.2, Sec. 5.3.3, App. C) — `verify_thesis_values.py` | **183 / 183 match** |
| Values quoted in the running text (Ch. 1, 3–7, App. B–C) — `verify_extended_values.py` | **150 / 150 match** (after correcting one value in Section 3.4: 0.96 → 0.97) |
| DP accounting — `dp_accounting_check.py` | all six noise multipliers reproduced; all accountant ε below nominal (Table D.2) |
| DP implementation — `dp_implementation_check.py` | **27 / 27 tests passed** (Table D.3) |
| Raw outputs vs. submitted outputs (S1) | split, FL+DP, SHAP-value, MIA and aggregated result files identical; centralized-LR scores/thresholds differ by ≤ 3 × 10⁻¹⁰ (multithreaded BLAS round-off), no printed value affected; timing fields differ as expected |
| Figures vs. submitted outputs | 8 / 9 PNG files byte-identical; SHAP beeswarm (Figure 5.6) differs only in the random vertical jitter of the plotted points (SHAP values identical) |
| Figures in the thesis vs. regenerated files | Figures 5.1, 5.3, 5.4, 5.5, 5.7 pixel-identical; Figure 5.6 identical up to cropping/jitter; Figure 5.2 is a re-styled rendering of `fig5_2_confusion_matrices.png` with identical counts (14,895/68/28/9; 14,917/46/37/0; 14,948/15/29/8; 14,947/16/37/0) and F1 values |

The only value in the thesis that did not match the raw outputs was the maximum single-seed univariate ROC-AUC in
Section 3.4 (printed 0.96; raw value 0.9654 → 0.97). It has been corrected.
