"""Run the complete pipeline (scripts 00-07 and 09; add --sensitivity for script 08)."""
import subprocess, sys, time
steps = ["00_calibrate_intensity.py", "01_generate_corpus.py", "02_preprocess_split.py", "03_baselines.py", "04_federated_dp.py", "05_audit_shap_mia_leakage.py", "06_governance_report.py", "07_aggregate_tables_figures.py", "09_weight_sensitivity_paired.py"]
if "--sensitivity" in sys.argv: steps.insert(-1, "08_generator_sensitivity.py")
t0 = time.time()
for s in steps:
    print(f"\n=== {s} ==="); r = subprocess.run([sys.executable, s]); 
    if r.returncode: sys.exit(f"{s} failed")
print(f"\nDone in {time.time()-t0:.0f}s. Results: outputs/ (ALL_RESULTS.xlsx, tables/, figures/, RESULTS_REPORT.md)")
