# PPUBA – reproducibility package (Supplementary Material S1)

Code, configuration, random seeds and raw outputs for the MSc thesis
*Privacy-Preserving AI for Cybersecurity: Evaluating Governance Compliance of User Behavior Analytics Systems in Saudi Arabia*
(Hayam Alrawili, Northern Border University, 2026). Table and section numbers below refer to the final thesis.

## 1. Run

```bash
pip install -r requirements.txt          # pinned versions (Table 4.4)
python run_all.py                        # scripts 00-07 and 09, seeds 42-46
python 08_generator_sensitivity.py --seeds 42 43 44 --scales 1 3   # Appendix C: attack-intensity sensitivity
python verify_thesis_values.py           # compares outputs/ with the values printed in Chapter 5 and Appendix C
pip install -r verification/requirements.txt
python verification/dp_accounting_check.py                         # Table D.2: independent DP-accounting check
```

Delete `data/` and `outputs/` before a clean re-run. Everything is seeded; with the pinned versions the same seeds give
the same numbers. Newer library versions run, but Random Forest values change slightly (Section 3.8).

## 2. Configuration and seeds (Appendix D)

| Item | Where |
|---|---|
| All parameters (corpus, detectors, FL + DP, audit, rubric) | `config.py` (dumped to `outputs/run_config.json` for every run) |
| Evaluation seeds | 42, 43, 44, 45, 46 (`config.SEEDS`); representative seed 43 for figures |
| Calibration seeds (attack intensity, a priori) | 1001-1008 (`00_calibrate_intensity.py` -> `calibrated_intensity.json`) |
| Intensity-sensitivity seeds | 42-44 (`08_generator_sensitivity.py`) |
| Bootstrap / Dirichlet seed | 2026 (`09_weight_sensitivity_paired.py`); 10,000 resamples, 100,000 draws |
| DP settings | T = 150 noisy steps, delta = 1e-5, C = 3.0, B = 256 (`config.py`; Tables 3.4 and 4.4) |

## 3. Scripts

| Script | Produces |
|---|---|
| 00 | a-priori intensity calibration on separate seeds (Section 3.4, Appendix B) |
| 01 | seeded synthetic CERT-v6.2-schema corpus per seed: 400 users x 150 days, 10 malicious, 15 archetypes |
| 02 | keyed-hash pseudonyms, k-anonymity / l-diversity, identity-aware split 240/60/100 users (6/2/2 malicious), scaling |
| 03 | Isolation Forest, Random Forest, Deep Autoencoder, centralized LR: metrics, validation thresholds, error cases |
| 04 | FedAvg + DP-SGD-style logistic regression (NumPy, `ppuba_lib/fl.py`) over the epsilon grid; round-composition run |
| 05 | membership inference, re-identification, SHAP, leakage checks (i)-(v) |
| 06 | governance-readiness index, leave-one-control-out, reproducibility hash check |
| 07 | all tables (CSV + `outputs/ALL_RESULTS.xlsx`), figures (PNG 300 dpi), `outputs/RESULTS_REPORT.md` |
| 08 | sensitivity of the results to the attack intensity |
| 09 | governance weight sensitivity (Table 5.4) and per-seed F1 with paired differences (Table C.2) |
| `verify_thesis_values.py` | automatic comparison of the outputs with the thesis values |
| `verification/dp_accounting_check.py` | independent check of the DP accounting (Table D.2), not part of the pipeline |

## 4. Thesis item -> raw output (Table D.1)

| Thesis item | Raw output (under `outputs/`) |
|---|---|
| Table 5.1 | `tables/T5_3a_detection_performance.csv`; per seed `tables/T5_3a_per_seed.csv` |
| Table 5.2, Figure 5.3, retention / Pareto (Section 5.3.2) | `tables/T5_3b_privacy_utility.csv`, `tables/T5_3b_pareto_operating_point.csv` |
| Figures 5.1, 5.2 (seed 43) | `figures/fig5_1_roc_baselines.png`, `figures/fig5_2_confusion_matrices.png` |
| Figure 5.4 (loss convergence) | `figures/fig5_3b_loss_convergence.png` |
| Section 5.3.3, Figure 5.5 | `tables/T5_3_3_membership_inference.csv`, `tables/T5_3_3_kanonymity_ldiversity.csv`, `figures/fig5_4_mia_vs_epsilon.png` |
| Section 5.3.4, Figure 5.6 | `tables/T5_3_4_shap_ranking_rep_seed.csv`, `tables/T5_3_4_shap_mean_over_seeds.csv`, `figures/fig5_5_shap_*.png` |
| Table 5.3, Figure 5.7 | `tables/T5_3c_governance.csv`, `governance_report.json`, `figures/fig5_6_governance_scores.png` |
| Table 5.4 | `tables/T5_4_governance_weight_sensitivity.csv` |
| Section 5.5, Appendix C cases | `tables/AppendixC_all_error_cases_FP_FN.csv`, `tables/T5_5_*` |
| Section 3.4 / Appendix C leakage checks | `tables/AppC_ii..v_*.csv` |
| Table C.1 | `tables/TC_1_partition_summary.csv` |
| Table C.2 | `tables/TC_2_per_seed_f1_paired.csv` |
| Appendix C intensity sensitivity | `tables/SENS_intensity_summary.csv`, `tables/SENS_intensity_all_runs.csv` |
| Table D.2 | `verification/dpcheck.json` |
| Everything in one workbook | `ALL_RESULTS.xlsx` |

## 5. Notes

1. **Differential privacy.** Noise is calibrated with the analytic Gaussian mechanism for (eps/150, delta/150) per noisy
   local step and composed by basic sequential composition; no subsampling amplification is claimed. The class weight
   multiplies the per-example gradient *before* clipping (`ppuba_lib/fl.py`), so the averaged clipped gradient has
   sensitivity C/B. Reported epsilons are nominal budgets; `verification/dp_accounting_check.py` shows they are
   conservative upper bounds for this mechanism. The code has not been formally audited.
2. **Governance rubric.** `governance_rubric_input.csv` is intentionally empty (per-criterion scores were not recorded),
   so script 06 uses the researcher-assigned control totals in `config.GOVERNANCE_TOTALS` and script 09 bounds the weight
   sensitivity (Section 4.4.4).
3. **Corpus.** The corpus is a CERT-inspired synthetic corpus produced by `ppuba_lib/data.py`; it is not the CERT v6.2 archive.
4. PR-AUC is computed as average precision (`sklearn.metrics.average_precision_score`).
5. Figure file names keep the pipeline's internal names; the mapping to thesis figure numbers is in Section 4 above.

## 6. Verified run (raw outputs)

The raw outputs are attached to the GitHub release `v1.0-thesis` as `ppuba_outputs_v1.0.zip`; unzip it inside this folder to obtain `outputs/`. They are the raw output of a clean run from scratch (scripts 00-09 and 08 with seeds 42-44, scales 1 and 3) with the
pinned versions in `requirements.txt` (run on Python 3.13; the thesis values were first produced on Python 3.12.3 with
identical results). `outputs/verify_thesis_values_report.txt` shows that all 183 checked values match Chapter 5 and
Appendix C; `outputs/verification_dp_accounting.txt` is the output of the DP-accounting check (Table D.2).
The intermediate corpora in `data/` (about 120 MB) are not included; scripts 01-02 regenerate them deterministically
from the seeds. `outputs/SHA256SUMS.txt` lists checksums of every output file (run `sha256sum -c outputs/SHA256SUMS.txt` in this folder).
