"""Compare the pipeline outputs with the values printed in the thesis (Chapter 5, Appendix C).
Run after run_all.py (and 08 for the intensity check). Prints PASS/FAIL per value; exit code 1 if any value differs.
Values are compared after rounding to the precision printed in the thesis."""
import json, os, sys, numpy as np, pandas as pd
import config as C

J = lambda p: json.load(open(p))
seeds = C.SEEDS
B = {s: J(os.path.join(C.SEED_DIR, f"baselines_seed_{s}.json"))["metrics"] for s in seeds}
F = {s: J(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json")) for s in seeds}
A = {s: J(os.path.join(C.SEED_DIR, f"audit_seed_{s}.json")) for s in seeds}
S = {s: J(os.path.join(C.SEED_DIR, f"split_seed_{s}.json")) for s in seeds}
mean = lambda v: float(np.mean(v)); sd = lambda v: float(np.std(v, ddof=1))
res = []
def chk(item, got, exp, nd=3):
    ok = round(got, nd) == round(exp, nd) or abs(got - exp) < 0.5 * 10 ** -nd + 1e-12
    res.append((ok, item, round(got, nd), exp))

# ---- Table 5.1 (mean +- SD over 5 seeds)
T51 = {"Isolation Forest": dict(pr_auc=(0.061, 0.024), precision=(0.073, 0.026), recall=(0.226, 0.119), f1=(0.102, 0.032), roc_auc=(0.879, 0.074)),
       "Random Forest": dict(pr_auc=(0.003, 0.001), precision=(0.000, 0.001), recall=(0.200, 0.447), f1=(0.001, 0.002), roc_auc=(0.555, 0.170)),
       "Deep Autoencoder": dict(pr_auc=(0.177, 0.032), precision=(0.177, 0.124), recall=(0.365, 0.173), f1=(0.187, 0.092), roc_auc=(0.905, 0.074))}
for n, d in T51.items():
    for k, (m, s) in d.items():
        v = [B[x][n][k] for x in seeds]; chk(f"T5.1 {n} {k} mean", mean(v), m); chk(f"T5.1 {n} {k} SD", sd(v), s)
for k, (m, s) in dict(pr_auc=(0.009, 0.015), precision=(0.009, 0.015), recall=(0.060, 0.100), f1=(0.015, 0.026), roc_auc=(0.570, 0.131)).items():
    v = [F[x]["metrics"]["5.0"][k] for x in seeds]; chk(f"T5.1 FL+DP eps=5 {k} mean", mean(v), m); chk(f"T5.1 FL+DP eps=5 {k} SD", sd(v), s)
for k, (m, s) in dict(f1=(0.068, 0.095), roc_auc=(0.618, 0.286)).items():
    v = [F[x]["metrics"]["inf"][k] for x in seeds]; chk(f"T5.1 FedAvg eps=inf {k} mean", mean(v), m); chk(f"T5.1 FedAvg eps=inf {k} SD", sd(v), s)
v = [B[x]["Centralized LR"]["f1"] for x in seeds]; chk("Centralized LR F1 mean", mean(v), 0.075); chk("Centralized LR F1 SD", sd(v), 0.104)

# ---- Table 5.2
T52 = {"0.5": (0.014, 0.027, 0.014, 0.018), "1.0": (0.046, 0.083, 0.021, 0.039), "2.0": (0.019, 0.025, 0.018, 0.021),
       "5.0": (0.015, 0.026, 0.009, 0.015), "10.0": (0.029, 0.028, 0.093, 0.188), "20.0": (0.033, 0.072, 0.028, 0.055), "inf": (0.068, 0.095, 0.041, 0.053)}
for k, (fm, fs, pm, ps) in T52.items():
    f = [F[x]["metrics"][k]["f1"] for x in seeds]; p = [F[x]["metrics"][k]["pr_auc"] for x in seeds]
    chk(f"T5.2 eps={k} F1 mean", mean(f), fm); chk(f"T5.2 eps={k} F1 SD", sd(f), fs); chk(f"T5.2 eps={k} PR-AUC mean", mean(p), pm); chk(f"T5.2 eps={k} PR-AUC SD", sd(p), ps)
L = {k: [F[x]["details"][k]["loss_curve"][-1] for x in seeds] for k in ("0.5", "1.0")}
chk("Final loss eps=0.5 mean", mean(L["0.5"]), 1.49, 2); chk("Final loss eps=0.5 SD", sd(L["0.5"]), 0.70, 2)
chk("Final loss eps=1 mean", mean(L["1.0"]), 0.76, 2); chk("Final loss eps=1 SD", sd(L["1.0"]), 0.20, 2)
rc = [mean([F[x]["composition_rounds_sensitivity"][k]["f1"] for x in seeds]) for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")]
chk("Round composition F1 min", min(rc), 0.024); chk("Round composition F1 max", max(rc), 0.130)
sig = {k: F[seeds[0]]["details"][k]["plan"]["noise_multiplier"] for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")}
for k, e in zip(sig, (1037.4, 543.0, 283.3, 119.4, 61.9, 32.1)): chk(f"T4.4 noise multiplier eps={k}", sig[k], e, 1)

# ---- Table C.2 per-seed F1
TC2 = {42: (0.080, 0.000, 0.237, 0.162, 0.195, 0.004, 0.190, 0.000, 0.017, 0.028, 0.004),
       43: (0.158, 0.000, 0.267, 0.000, 0.000, 0.000, 0.042, 0.058, 0.000, 0.045, 0.000),
       44: (0.095, 0.000, 0.182, 0.213, 0.147, 0.062, 0.000, 0.004, 0.059, 0.067, 0.162),
       45: (0.087, 0.000, 0.033, 0.000, 0.000, 0.000, 0.000, 0.003, 0.000, 0.000, 0.000),
       46: (0.092, 0.004, 0.215, 0.000, 0.000, 0.003, 0.000, 0.029, 0.000, 0.004, 0.000)}
for s, row in TC2.items():
    got = [B[s][n]["f1"] for n in ("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR")] + \
          [F[s]["metrics"][k]["f1"] for k in ("inf", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0")]
    for name, g, e in zip(("IF", "RF", "DAE", "LR", "inf", "0.5", "1", "2", "5", "10", "20"), got, row): chk(f"TC.2 seed {s} {name}", g, e)

# ---- Section 5.3.3 membership inference and k-anonymity
mia_fl = [mean([A[x]["mia"][f"FL+DP eps={k}"]["mia_auc"] for x in seeds]) for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")]
chk("MIA FL+DP min over eps", min(mia_fl), 0.489); chk("MIA FL+DP max over eps", max(mia_fl), 0.505)
v = [A[x]["mia"]["FL+DP eps=inf"]["mia_auc"] for x in seeds]; chk("MIA non-private mean", mean(v), 0.494); chk("MIA non-private SD", sd(v), 0.013)
cen = [mean([A[x]["mia"][n]["mia_auc"] for x in seeds]) for n in A[seeds[0]]["mia"] if not n.startswith("FL+DP")]
chk("MIA centralized min", min(cen), 0.485); chk("MIA centralized max", max(cen), 0.506)
k = [S[x]["anonymity"]["k_anonymity_records"] for x in seeds]; chk("k-anonymity min", min(k), 1123, 0); chk("k-anonymity max", max(k), 1684, 0)

# ---- Section 5.3.4 SHAP (seed 43) and 5.3.5 governance
sh = {r["feature"]: r for r in A[C.REP_SEED]["shap"]["ranking"]}
for f, e in (("dept_risk_score", 0.0947), ("sensitive_file_access", 0.0725), ("file_access_count", 0.0471)):
    key = [k for k in sh[f] if "mean" in k and "abs" in k.lower()][0]; chk(f"SHAP seed 43 {f}", sh[f][key], e, 4)
g = J(os.path.join(C.OUT_DIR, "governance_report.json")); chk("Governance index", g["overall_index"], 85.4, 1)
chk("Leave-one-out min", g["loo_min"], 84.0, 1); chk("Leave-one-out max", g["loo_max"], 87.0, 1)

# ---- Table 5.4 (script 09)
t54 = pd.read_csv(os.path.join(C.TAB_DIR, "T5_4_governance_weight_sensitivity.csv"))
exp54 = [(82.8, 91.4), (80.9, 87.3), (83.3, 91.6), (80.6, 87.0), (83.8, 91.4), (80.5, 86.7), (84.2, 90.5), (80.9, 86.4), (84.6, 89.3), (81.8, 86.1), (80.8, 89.7), (83.3, 87.4)]
for (_, r), (lo, hi) in zip(t54.iterrows(), exp54):
    chk(f"T5.4 {r.Perturbation} low", r["Index min"], lo, 1); chk(f"T5.4 {r.Perturbation} high", r["Index max"], hi, 1)

# ---- Appendix C intensity sensitivity (script 08, scale 3, seeds 42-44), if run
p = os.path.join(C.TAB_DIR, "SENS_intensity_all_runs.csv")
if os.path.exists(p):
    d = pd.read_csv(p); d = d[(d.intensity_scale == 3.0) & d.seed.isin([42, 43, 44])]
    for n, e in (("Deep Autoencoder", 0.836), ("Random Forest", 0.349), ("FedAvg eps=inf", 0.664), ("FL+DP eps=5", 0.246)):
        if (d.model == n).any(): chk(f"AppC scale 3 {n} F1", d[d.model == n].f1.mean(), e)

bad = [r for r in res if not r[0]]
for ok, item, got, exp in res: print(f"{'PASS' if ok else 'FAIL'}  {item}: pipeline {got}  thesis {exp}")
print(f"\n{len(res) - len(bad)}/{len(res)} values match the thesis.")
sys.exit(1 if bad else 0)
