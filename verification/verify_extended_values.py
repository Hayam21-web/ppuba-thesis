"""Extended check of the numbers printed in the thesis text (Chapters 1, 3-7 and Appendices B-C).

verify_thesis_values.py checks the 183 values of the main tables (Tables 5.1-5.4, 4.4, C.2, Section 5.3.3, Appendix C
intensity sensitivity). This script checks the remaining numbers that are quoted in the running text, so that every
numerical statement in the thesis is traced to a raw output file. Run it from the package root after run_all.py and 08:

    python verification/verify_extended_values.py        (exit code 1 if any value differs)

Values are compared after rounding to the precision printed in the thesis.
"""
import ast, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

J = lambda p: json.load(open(p))
seeds = C.SEEDS
T = lambda n, **kw: pd.read_csv(os.path.join(C.TAB_DIR, n + ".csv"), **kw)
B = {s: J(os.path.join(C.SEED_DIR, f"baselines_seed_{s}.json"))["metrics"] for s in seeds}
F = {s: J(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json")) for s in seeds}
A = {s: J(os.path.join(C.SEED_DIR, f"audit_seed_{s}.json")) for s in seeds}
S = {s: J(os.path.join(C.SEED_DIR, f"split_seed_{s}.json")) for s in seeds}
mean = lambda v: float(np.mean(v)); sd = lambda v: float(np.std(v, ddof=1))
first = lambda s: float(str(s).split("±")[0])
res = []


def chk(where, item, got, exp, nd=3):
    ok = round(float(got), nd) == round(exp, nd) or abs(float(got) - exp) < 0.5 * 10 ** -nd + 1e-12
    res.append((ok, where, item, round(float(got), nd), exp))


# ---------------- Partitions (Sections 3.4, 3.8, 5.2; Tables 3.2, C.1)
for part, (m, lo, hi) in {"train": (78, 69, 85), "val": (27, 23, 32), "test": (28, 20, 37)}.items():
    v = [S[s]["summary"][part]["positives"] for s in seeds]
    chk("Table C.1", f"{part} positives mean", mean(v), m, 0); chk("Table C.1", f"{part} positives min", min(v), lo, 0)
    chk("Table C.1", f"{part} positives max", max(v), hi, 0)
v = [S[s]["summary"]["test"]["positives"] for s in seeds]
chk("Sec 5.2", "test positive rate (%)", 100 * mean(v) / 15000, 0.18, 2)
chk("Sec 5.5", "malicious test user-days pooled over 5 seeds", sum(v), 138, 0)
chk("Sec 4.4.3", "training positive rate (%)", 100 * mean([S[s]["summary"]["train"]["positives"] for s in seeds]) / 36000, 0.22, 2)
for part, (u, mu) in {"train": (240, 6), "val": (60, 2), "test": (100, 2)}.items():
    for s in seeds:
        chk("Table 3.2", f"seed {s} {part} users", S[s]["summary"][part]["users"], u, 0)
        chk("Table 3.2", f"seed {s} {part} malicious users", S[s]["summary"][part]["malicious_users"], mu, 0)

# ---------------- Anonymity metrics (Tables 1.2, 4.2, 4.5, C.3; Sections 5.2, 5.3.3, 7.1)
kr = [S[s]["anonymity"]["k_anonymity_records"] for s in seeds]; ku = [S[s]["anonymity"]["k_anonymity_users"] for s in seeds]
chk("Table C.3", "record-level k min", min(kr), 1123, 0); chk("Table C.3", "record-level k max", max(kr), 1684, 0)
chk("Table C.3", "user-level k min", min(ku), 12, 0); chk("Table C.3", "user-level k max", max(ku), 18, 0)
for s, (a, b) in zip(seeds, ((1684, 15), (1238, 13), (1123, 15), (1613, 18), (1569, 12))):
    chk("Table C.3", f"seed {s} record-level k", S[s]["anonymity"]["k_anonymity_records"], a, 0)
    chk("Table C.3", f"seed {s} user-level k", S[s]["anonymity"]["k_anonymity_users"], b, 0)
chk("Sec 5.3.3", "max record-level re-identification risk (%)", 100 / min(kr), 0.09, 2)
chk("Sec 5.3.3", "max user-level re-identification risk (%)", 100 / min(ku), 8.3, 1)
chk("Sec 5.3.3", "l-diversity (all seeds)", max(S[s]["anonymity"]["l_diversity"] for s in seeds), 1, 0)
chk("Sec 5.3.3", "number of QI groups", S[seeds[0]]["anonymity"]["n_groups"], 24, 0)

# ---------------- Detection text (Sections 5.3.1, 5.4, 5.5)
d = T("T5_3a_detection_performance").set_index("Model")
chk("Sec 5.3.1", "DAE FPR (%)", 100 * first(d.loc["Deep Autoencoder", "FPR"]), 1.2, 1)
chk("Sec 5.3.1", "DAE FPR SD (%)", 100 * float(d.loc["Deep Autoencoder", "FPR"].split("±")[1]), 2.0, 1)
chk("Sec 5.5", "IF FPR (%)", 100 * first(d.loc["Isolation Forest", "FPR"]), 0.6, 1)
chk("Sec 5.5", "RF FPR (%)", 100 * mean([B[s]["Random Forest"]["fpr"] for s in seeds]), 24.0, 1)
chk("Sec 5.5", "RF FPR SD (%)", 100 * sd([B[s]["Random Forest"]["fpr"] for s in seeds]), 42.8, 1)
chk("Sec 5.5", "FL+DP eps=5 FPR (%)", 100 * first(d.loc["FL + DP (eps = 5)", "FPR"]), 0.8, 1)
chk("Sec 5.4", "DAE recall at FPR<=1% (%)", 100 * first(d.loc["Deep Autoencoder", "Recall@FPR<=1%"]), 41.7, 1)
chk("Sec 5.4", "IF recall at FPR<=1% (%)", 100 * first(d.loc["Isolation Forest", "Recall@FPR<=1%"]), 32.2, 1)
chk("Sec 5.4", "DAE miss rate (%)", 100 * (1 - first(d.loc["Deep Autoencoder", "Recall"])), 63.5, 1)
chk("Sec 5.5", "FedAvg eps=inf recall", first(d.loc["FedAvg non-private (eps = inf)", "Recall"]), 0.067)
th = T("T3_7_threshold_consistency").set_index("model")["threshold CV"]
chk("Sec 5.3.1/5.5", "RF threshold CV", th["Random Forest"], 0.97, 2); chk("Sec 5.5", "DAE threshold CV", th["Deep Autoencoder"], 0.31, 2)
chk("Sec 5.5", "FL+DP eps=5 threshold CV", th["FL + DP (eps = 5)"], 0.35, 2)
e = T("T5_5_error_counts_by_role"); tot = e.groupby(["detector", "error_type"])["count"].sum()
chk("Sec 5.5", "DAE false alarms pooled", tot[("Deep Autoencoder", "FP")], 874, 0)
chk("Sec 5.5", "IF false alarms pooled", tot[("Isolation Forest", "FP")], 450, 0)
chk("Sec 5.5", "DAE missed malicious user-days", tot[("Deep Autoencoder", "FN")], 91, 0)
er = e.set_index(["detector", "error_type", "role"])["count"]
chk("Sec 5.5", "DAE FP IT_Admin", er[("Deep Autoencoder", "FP", "IT_Admin")], 175, 0)
chk("Sec 5.5", "DAE FP Sales", er[("Deep Autoencoder", "FP", "Sales")], 169, 0)
r = T("T5_5_per_scenario_recall"); r = r[r.detector == "Deep Autoencoder"].set_index("scenario")["mean_recall"]
for sc, v in (("mass_download", 0.63), ("slow_exfil_mixed", 0.57), ("post_termination_access", 0.50), ("privilege_escalation", 0.47),
              ("night_access_sensitive", 0.25), ("data_exfil_usb", 0.0), ("data_exfil_email", 0.0), ("insider_recon", 0.0)):
    chk("Sec 5.5", f"DAE recall {sc}", r[sc], v, 2)
t = T("T5_5_time_to_detection_summary").set_index("detector")
chk("Sec 5.4", "DAE users detected", t.loc["Deep Autoencoder", "users_detected"], 7, 0)
chk("Sec 5.4", "DAE mean days to first alert", t.loc["Deep Autoencoder", "mean_days_to_first_alert"], 0.14, 2)

# ---------------- Privacy-utility text (Section 5.3.2)
pu = T("T5_3b_privacy_utility", dtype={"epsilon": str}).set_index("epsilon")
for e_, v in zip(("0.5", "1.0", "2.0", "5.0", "10.0", "20.0"), (19.9, 67.8, 27.4, 22.3, 42.0, 48.5)):
    chk("Sec 5.3.2", f"F1 retention eps={e_} (%)", pu.loc[e_, "F1 retention vs eps=inf (%)"], v, 1)
roc = [first(pu.loc[e_, "ROC-AUC"]) for e_ in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")]
chk("Sec 5.3.2", "FL+DP ROC-AUC min", min(roc), 0.39, 2); chk("Sec 5.3.2", "FL+DP ROC-AUC max", max(roc), 0.65, 2)
for e_, v in (("0.5", 0.39), ("2.0", 0.26), ("10.0", 0.21), ("20.0", 0.20)):
    chk("Sec 5.3.2", f"FPR eps={e_}", first(pu.loc[e_, "FPR"]), v, 2)
for e_ in ("10.0", "inf"):
    chk("Sec 4.4.3/5.3.2", f"final loss eps={e_}", first(pu.loc[e_, "final train loss"]), 0.14, 2)
po = T("T5_3b_pareto_operating_point").set_index("item")["value"]
chk("Sec 5.3.2", "60% rule operating point", float(po["recommended_operating_point"]), 1.0, 1)
chk("Sec 5.3.2", "Pareto frontier = {0.5, 1}", float(ast.literal_eval(po["pareto_frontier_eps"]) == ["0.5", "1.0"]), 1, 0)
fsd = [sd([F[s]["metrics"][k]["f1"] for s in seeds]) for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "inf")]
chk("Sec 5.3.2", "between-seed F1 SD min", min(fsd), 0.025); chk("Sec 5.3.2", "between-seed F1 SD max", max(fsd), 0.095)

# ---------------- Leakage checks (Section 3.4, Appendix C)
u = T("AppC_ii_univariate_auc").set_index("feature")
chk("Sec 3.4 (ii)", "max 5-seed mean univariate AUC (val)", u["val"].max(), 0.74, 2)
chk("Sec 3.4 (ii)", "argmax is file_access_count", float(u["val"].idxmax() == "file_access_count"), 1, 0)
per_seed_max = max(max(A[s]["leakage"]["ii_univariate_auc"]["val"].values()) for s in seeds)
chk("Sec 3.4 (ii)", "max single-seed univariate AUC (val)", per_seed_max, 0.97, 2)
ab = T("AppC_iii_ablation").set_index("model")
chk("Sec 3.4 (iii)", "DAE F1 without 2 features", first(ab.loc["Deep Autoencoder", "F1 without 2 features"]), 0.169)
chk("Sec 3.4 (iii)", "IF F1 without 2 features", first(ab.loc["Isolation Forest", "F1 without 2 features"]), 0.061)
iv = T("AppC_iv_dept_risk_score")
chk("Sec 3.4 (iv)", "min Welch p (all > 0.2)", float(iv.window_vs_nonwindow_welch_p.min() > 0.2), 1, 0)
sh = T("AppC_v_shuffled_labels").set_index("model")
chk("Sec 3.4 (v)", "max shuffled-label F1", max(first(x) for x in sh["F1"]), 0.017)
chk("Sec 3.4 (v)", "min shuffled-label ROC-AUC", min(first(x) for x in sh["ROC-AUC"]), 0.25, 2)
chk("Sec 3.4 (v)", "max shuffled-label ROC-AUC", max(first(x) for x in sh["ROC-AUC"]), 0.62, 2)

# ---------------- SHAP (Section 5.3.4)
rep = {x["feature"]: x for x in A[C.REP_SEED]["shap"]["ranking"]}
for f, v in (("email_external_count", 0.0427), ("session_duration_min", 0.0423), ("email_sent_count", 0.0393), ("login_count", 0.0238),
             ("role_Engineering", 0.0221), ("role_IT_Admin", 0.0202), ("role_Finance", 0.0198)):
    chk("Sec 5.3.4", f"SHAP seed 43 {f}", rep[f]["mean_abs_shap"], v, 4)
ms = T("T5_3_4_shap_mean_over_seeds").set_index("feature")
chk("Sec 5.3.4", "dept_risk_score mean |SHAP| over seeds", ms.loc["dept_risk_score", "mean_abs_shap"], 0.076)
chk("Sec 5.3.4", "dept_risk_score first on average", float(ms["mean_abs_shap"].idxmax() == "dept_risk_score"), 1, 0)
chk("Sec 5.3.4", "sensitive_file_access mean Spearman", ms.loc["sensitive_file_access", "mean_spearman"], 0.82, 2)
chk("Sec 5.3.4", "email_sent_count mean Spearman", ms.loc["email_sent_count", "mean_spearman"], -0.67, 2)
chk("Sec 5.3.4", "dept_risk_score mean Spearman", ms.loc["dept_risk_score", "mean_spearman"], 0.05, 2)
top3 = list(ms["mean_abs_shap"].sort_values(ascending=False).index[:3])
chk("Abstract/Sec 6.1.3", "5-seed top-3 = dept, sensitive, file", float(top3 == ["dept_risk_score", "sensitive_file_access", "file_access_count"]), 1, 0)

# ---------------- MIA text
mia = T("T5_3_3_membership_inference").set_index("model")
allv = [first(x) for x in mia.iloc[:, 0]]
chk("Sec 5.7", "MIA min over all models", min(allv), 0.485); chk("Sec 5.7", "MIA max over all models", max(allv), 0.506)

# ---------------- Calibration (Section 3.4, Appendix B)
cal = J(os.path.join(C.BASE_DIR, "calibrated_intensity.json")); g0 = [g for g in cal["grid"] if g["scale"] == 1.0][0]
chk("Sec 3.4", "calibration median primary AUC", g0["median_primary_auc"], 0.914)
chk("Sec 3.4", "calibration min archetype AUC", min(g0["per_archetype"].values()), 0.77, 2)
chk("Sec 3.4", "calibration max archetype AUC", max(g0["per_archetype"].values()), 0.97, 2)
chk("Sec 5.5", "calibration AUC data_exfil_usb", g0["per_archetype"]["data_exfil_usb"], 0.85, 2)
chk("Sec 5.5", "calibration AUC insider_recon", g0["per_archetype"]["insider_recon"], 0.83, 2)
chk("Sec 3.4", "chosen intensity scale", cal["chosen_intensity_scale"], 1.0, 1)

# ---------------- Archetype occurrences over the 50 malicious users (Table B.2), from the regenerated corpora
raw = sorted(glob.glob(os.path.join(C.DATA_DIR, "raw_corpus_seed_4[2-6].csv")))
if len(raw) == 5:
    occ = pd.Series(dtype=int)
    for p in raw:
        x = pd.read_csv(p, usecols=["user_name", "scenario", "is_malicious"]); x = x[x.is_malicious == 1].drop_duplicates("user_name")
        occ = occ.add(x.scenario.value_counts(), fill_value=0)
    chk("Table B.2", "malicious users over 5 seeds", occ.sum(), 50, 0)
    for sc, v in (("data_exfil_usb", 8), ("brute_force_then_access", 8), ("print_exfiltration", 4), ("night_access_sensitive", 4)):
        chk("Table B.2", f"occurrences {sc}", occ.get(sc, 0), v, 0)
    rest = occ.drop(["data_exfil_usb", "brute_force_then_access", "print_exfiltration", "night_access_sensitive"])
    chk("Table B.2", "other archetypes: count", len(rest), 11, 0); chk("Table B.2", "other archetypes: min", rest.min(), 1, 0)
    chk("Table B.2", "other archetypes: max", rest.max(), 3, 0)

# ---------------- Composition-over-rounds sensitivity and paired differences
cr = T("T5_3b_composition_rounds_sensitivity", dtype={"epsilon": str}).set_index("epsilon")
vals = {e_: first(cr.loc[e_].iloc[0]) for e_ in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")}
chk("Sec 5.3.2", "round-composition argmin is eps=5", float(min(vals, key=vals.get) == "5.0"), 1, 0)
chk("Sec 5.3.2", "round-composition argmax is eps=10", float(max(vals, key=vals.get) == "10.0"), 1, 0)
tc = T("TC_2_per_seed_f1_paired")
chk("Sec 5.6", "ties at zero (seed x eps pairs)", sum(int(F[s]["metrics"]["inf"]["f1"] == 0 and F[s]["metrics"][k]["f1"] == 0)
                                                     for s in seeds for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0")), 11, 0)
chk("Sec 5.6", "seeds with non-private F1 = 0", sum(int(F[s]["metrics"]["inf"]["f1"] == 0) for s in seeds), 3, 0)
dd = [B[s]["Deep Autoencoder"]["f1"] - B[s]["Isolation Forest"]["f1"] for s in seeds]
chk("Sec 5.6", "DAE > IF in n seeds", sum(x > 0 for x in dd), 4, 0); chk("Sec 5.6", "DAE - IF mean paired diff", mean(dd), 0.084)

bad = [x for x in res if not x[0]]
for ok, where, item, got, exp in res: print(f"{'PASS' if ok else 'FAIL'}  [{where}] {item}: pipeline {got}  thesis {exp}")
print(f"\n{len(res) - len(bad)}/{len(res)} additional text values match the thesis.")
sys.exit(1 if bad else 0)
