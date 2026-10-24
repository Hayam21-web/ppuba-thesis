# Verification run — 25 October 2026 (release v1.2-thesis)

Adds the checks requested after the supervisor's review; the pipeline code (scripts 00–09, `ppuba_lib/`) is unchanged.

| Check | Script | Result |
|---|---|---|
| External validation, ADFA-LD and Enron POI (Section 5.6, Tables 5.5–5.6) | `10_external_validation.py` | ran twice (package and clean environment); all output tables byte-identical |
| Values quoted in the text, incl. 31 external-validation values | `verification/verify_extended_values.py` | **181 / 181 match** (`rerun_verify_extended.log`) |
| Cross-check of one DP step against Opacus 1.5.2 (Table D.5) | `verification/dp_opacus_crosscheck.py` | clipped mean: max. relative difference 2.2e-7; noise SD within sampling error for all six ε (`rerun_dp_opacus.log`) |
| Empirical privacy audit (Table D.6) | `verification/dp_empirical_audit.py` | no lower bound above the theoretical replace-one ε in 6 configurations; strongest test reaches 69 % of theory (`dp_empirical_audit.log`) |
| DP implementation tests (Table D.3) | `verification/dp_implementation_check.py` | 27 / 27 passed |

These checks are tests and empirical audits; they are not a formal verification of differential privacy.
