"""Script 07: aggregate every seed into the thesis tables (CSV + one Excel workbook), figures (PNG, 300 dpi) and RESULTS_REPORT.md."""
import glob, json, os, platform, importlib.metadata as im, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import config as C
from sklearn.metrics import roc_curve, confusion_matrix

J = lambda p: json.load(open(p))
seeds = [s for s in C.SEEDS if all(os.path.exists(os.path.join(C.SEED_DIR, f"{k}_seed_{s}.json")) for k in ("baselines", "fl", "audit", "split"))]
B = {s: J(os.path.join(C.SEED_DIR, f"baselines_seed_{s}.json")) for s in seeds}; F = {s: J(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json")) for s in seeds}
A = {s: J(os.path.join(C.SEED_DIR, f"audit_seed_{s}.json")) for s in seeds}; S = {s: J(os.path.join(C.SEED_DIR, f"split_seed_{s}.json")) for s in seeds}
rep = C.REP_SEED if C.REP_SEED in seeds else seeds[0]
T = {}   # name -> DataFrame (saved to CSV + Excel)
def ms(v, nd=3):
    v = np.asarray(v, float); return f"{np.nanmean(v):.{nd}f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.{nd}f}"
def num(v): v = np.asarray(v, float); return float(np.nanmean(v)), float(np.nanstd(v, ddof=1) if len(v) > 1 else 0)
KEYS = [("pr_auc", "PR-AUC"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("fpr", "FPR"), ("fnr", "FNR"),
        ("recall_at_fpr_0.001", "Recall@FPR<=0.1%"), ("recall_at_fpr_0.01", "Recall@FPR<=1%")]
dets = {"Isolation Forest": lambda s: B[s]["metrics"]["Isolation Forest"], "Random Forest": lambda s: B[s]["metrics"]["Random Forest"],
        "Deep Autoencoder": lambda s: B[s]["metrics"]["Deep Autoencoder"], "Centralized LR (reference)": lambda s: B[s]["metrics"]["Centralized LR"],
        f"FL + DP (eps = {C.REPORT_EPS:g})": lambda s: F[s]["metrics"][str(C.REPORT_EPS)], "FedAvg non-private (eps = inf)": lambda s: F[s]["metrics"]["inf"]}
# ---- Table 5.3a
T["T5_3a_detection_performance"] = pd.DataFrame([{"Model": n, **{lab: ms([fn(s)[k] for s in seeds]) for k, lab in KEYS}} for n, fn in dets.items()])
# ---- Per-seed detail
rows = []
for s in seeds:
    for n, fn in dets.items(): m = fn(s); rows.append({"seed": s, "model": n, **{lab: m[k] for k, lab in KEYS}, "threshold": m["threshold"], "TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"]})
T["T5_3a_per_seed"] = pd.DataFrame(rows)
# ---- Table 5.3b (epsilon sweep)
keys = [k for k in F[rep]["metrics"]]; f1inf = np.mean([F[s]["metrics"]["inf"]["f1"] for s in seeds]); rows = []
for k in keys:
    g = lambda m: [F[s]["metrics"][k][m] for s in seeds]; pl = F[rep]["details"][k]["plan"]
    rows.append({"epsilon": k, "F1": ms(g("f1")), "PR-AUC": ms(g("pr_auc")), "Precision": ms(g("precision")), "Recall": ms(g("recall")), "ROC-AUC": ms(g("roc_auc")), "FPR": ms(g("fpr"), 4),
                 "F1 retention vs eps=inf (%)": f"{100 * np.mean(g('f1')) / f1inf:.1f}" if f1inf > 0 else "n/a",
                 "MIA AUC": ms([A[s]["mia"][f"FL+DP eps={k}"]["mia_auc"] for s in seeds]), "noise multiplier (sigma)": pl["noise_multiplier"],
                 "noise SD per coordinate": pl["noise_std_per_coordinate"], "eps per noisy step": pl["eps_per_composition"], "delta per noisy step": pl["delta_per_composition"],
                 "final train loss": ms(g("final_train_loss"))})
T["T5_3b_privacy_utility"] = pd.DataFrame(rows)
mean_f1 = {k: np.mean([F[s]["metrics"][k]["f1"] for s in seeds]) for k in keys}
fin = [k for k in keys if k != "inf"]; front = [k for k in fin if not any((float(j) < float(k) and mean_f1[j] >= mean_f1[k]) for j in fin)]
ok = [k for k in fin if f1inf > 0 and mean_f1[k] >= C.UTILITY_RETENTION_RULE * f1inf]
op = dict(non_private_mean_f1=f1inf, pareto_frontier_eps=front, eps_meeting_60pct_rule=ok, recommended_operating_point=(min(ok, key=float) if ok else None),
          retention_at_recommended=(mean_f1[min(ok, key=float)] / f1inf if ok else None))
T["T5_3b_pareto_operating_point"] = pd.DataFrame([{"item": k, "value": str(v)} for k, v in op.items()])
# ---- thresholds consistency, latency/throughput, time-to-detection
rows = []
for n, fn in dets.items():
    th = [fn(s)["threshold"] for s in seeds]; rows.append({"model": n, "threshold mean": np.mean(th), "threshold SD": np.std(th, ddof=1) if len(th) > 1 else 0, "threshold CV": (np.std(th, ddof=1) / abs(np.mean(th)) if np.mean(th) else np.nan)})
T["T3_7_threshold_consistency"] = pd.DataFrame(rows)
T["T3_7_latency_throughput"] = pd.DataFrame([{"model": n, "latency ms/record": ms([B[s]["metrics"][n]["latency_ms_per_record"] for s in seeds], 4), "throughput records/s": ms([B[s]["metrics"][n]["throughput_records_per_s"] for s in seeds], 0)} for n in ("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR")] +
    [{"model": f"FL+DP eps={C.REPORT_EPS:g}", "latency ms/record": ms([F[s]["metrics"][str(C.REPORT_EPS)]["latency_ms_per_record"] if "latency_ms_per_record" in F[s]["metrics"][str(C.REPORT_EPS)] else np.nan for s in seeds], 4), "throughput records/s": "n/a (linear model; see train_seconds)"}])
def ttd_rows(src, getn):
    out = []
    for s in seeds:
        for n, lst in src[s]["time_to_detection"].items():
            for r in lst: out.append({"seed": s, "detector": getn(n), **r})
    return pd.DataFrame(out)
ttd = pd.concat([ttd_rows(B, lambda n: n), ttd_rows(F, lambda n: f"FL+DP eps={n}")]); T["T5_5_time_to_detection_cases"] = ttd
T["T5_5_time_to_detection_summary"] = ttd.groupby("detector").agg(malicious_users=("user", "count"), users_detected=("first_alert_day", lambda x: int(x.notna().sum())), detection_rate=("first_alert_day", lambda x: x.notna().mean()),
    mean_days_to_first_alert=("first_alert_day", "mean"), median_days_to_first_alert=("first_alert_day", "median"), mean_window_days_flagged_share=("days_flagged", "sum")).reset_index()
# ---- error analysis
err = pd.concat([pd.read_csv(p) for s in seeds for p in (os.path.join(C.SEED_DIR, f"error_cases_baselines_seed_{s}.csv"), os.path.join(C.SEED_DIR, f"error_cases_fl_seed_{s}.csv"))])
err.to_csv(os.path.join(C.TAB_DIR, "AppendixC_all_error_cases_FP_FN.csv"), index=False)
T["T5_5_error_counts_by_role"] = err.groupby(["detector", "error_type", "role"]).size().rename("count").reset_index()
T["T5_5_error_counts_by_scenario"] = err[err.error_type == "FN"].groupby(["detector", "scenario"]).size().rename("missed_user_days").reset_index()
T["T5_5_per_scenario_recall"] = ttd.assign(recall=lambda d: d.days_flagged / d.window_days).groupby(["detector", "scenario"]).agg(windows=("user", "count"), mean_recall=("recall", "mean")).reset_index()
T["T5_5_false_positive_profile"] = err[err.error_type == "FP"].groupby("detector")[C.NUMERIC_FEATURES].mean().reset_index()
# ---- scalability and composition sensitivity (script 04 extras)
if all("scalability" in F[s] for s in seeds):
    rows = []
    for n in C.SCALABILITY_CLIENTS:
        for k in ("inf", str(C.REPORT_EPS)):
            g = lambda m: [F[s]["scalability"][str(n)][k][m] for s in seeds]; rows.append({"clients": n, "epsilon": k, "F1": ms(g("f1")), "PR-AUC": ms(g("pr_auc")), "sec/round": ms(g("seconds_per_round"), 4)})
    T["T3_7_scalability_clients"] = pd.DataFrame(rows)
    T["T5_3b_composition_rounds_sensitivity"] = pd.DataFrame([{"epsilon": k, "F1 (composition over rounds)": ms([F[s]["composition_rounds_sensitivity"][k]["f1"] for s in seeds]),
        "PR-AUC": ms([F[s]["composition_rounds_sensitivity"][k]["pr_auc"] for s in seeds])} for k in keys])
# ---- privacy audit tables
mia_names = list(A[rep]["mia"]); T["T5_3_3_membership_inference"] = pd.DataFrame([{"model": n, "MIA AUC (mean ± SD over seeds)": ms([A[s]["mia"][n]["mia_auc"] for s in seeds]), "SD over draws (within seed, mean)": np.mean([A[s]["mia"][n]["mia_auc_sd_over_draws"] for s in seeds])} for n in mia_names])
T["T5_3_3_kanonymity_ldiversity"] = pd.DataFrame([{"seed": s, **A[s]["reidentification"]} for s in seeds])
# ---- SHAP
sh = pd.DataFrame(A[rep]["shap"]["ranking"]); T["T5_3_4_shap_ranking_rep_seed"] = sh
allsh = pd.concat([pd.DataFrame(A[s]["shap"]["ranking"]).assign(seed=s) for s in seeds]); T["T5_3_4_shap_mean_over_seeds"] = allsh.groupby("feature").agg(mean_abs_shap=("mean_abs_shap", "mean"), sd=("mean_abs_shap", "std"), mean_rank=("rank", "mean"), mean_spearman=("spearman_value_vs_shap", "mean")).sort_values("mean_abs_shap", ascending=False).reset_index()
# ---- leakage checks (Appendix C)
if all("leakage" in A[s] for s in seeds):
    ua = pd.DataFrame({p: {f: np.mean([A[s]["leakage"]["ii_univariate_auc"][p][f] for s in seeds]) for f in C.NUMERIC_FEATURES} for p in ("train", "val", "test")}); T["AppC_ii_univariate_auc"] = ua.reset_index().rename(columns={"index": "feature"})
    ab = A[rep]["leakage"]["iii_ablation_without_sensitive_file_access_and_dept_risk_score"]; rows = []
    for n in ab:
        full = (B[rep]["metrics"].get(n) or F[rep]["metrics"].get(n.replace("FL+DP eps=", ""))); g = lambda m: [A[s]["leakage"]["iii_ablation_without_sensitive_file_access_and_dept_risk_score"][n][m] for s in seeds]
        ref = [(B[s]["metrics"].get(n) or F[s]["metrics"].get(n.replace("FL+DP eps=", "")))["f1"] for s in seeds]
        rows.append({"model": n, "F1 without 2 features": ms(g("f1")), "F1 full feature set": ms(ref), "PR-AUC without": ms(g("pr_auc")), "ROC-AUC without": ms(g("roc_auc"))})
    T["AppC_iii_ablation"] = pd.DataFrame(rows)
    T["AppC_iv_dept_risk_score"] = pd.DataFrame([{"seed": s, **A[s]["leakage"]["iv_dept_risk_score"]} for s in seeds])
    sc = A[rep]["leakage"]["v_shuffled_label_control"]; T["AppC_v_shuffled_labels"] = pd.DataFrame([{"model": n, "F1": ms([A[s]["leakage"]["v_shuffled_label_control"][n]["f1"] for s in seeds]), "PR-AUC": ms([A[s]["leakage"]["v_shuffled_label_control"][n]["pr_auc"] for s in seeds]), "ROC-AUC": ms([A[s]["leakage"]["v_shuffled_label_control"][n]["roc_auc"] for s in seeds]),
        "test prevalence (chance PR-AUC)": np.mean([A[s]["leakage"]["v_shuffled_label_control"]["test_prevalence"] for s in seeds])} for n in sc if n != "test_prevalence"])
# ---- Table C.1 partition summary, B.1, B.2, 3.5b, 4.5, software versions
rows = []
for p in ("train", "val", "test"):
    g = lambda m: [S[s]["summary"][p][m] for s in seeds]; pos = g("positives")
    rows.append({"Partition": p, "Users": g("users")[0], "Records (average)": np.mean(g("records")), "Malicious users": g("malicious_users")[0], "Positive user-days (average)": np.mean(pos), "range": f"{min(pos)}–{max(pos)}", "positive rate %": 100 * np.mean(pos) / np.mean(g("records"))})
T["TC_1_partition_summary"] = pd.DataFrame(rows)
T["TB_1_feature_definitions"] = pd.DataFrame([{"Feature": f, "Definition": d} for f, d in zip(C.NUMERIC_FEATURES + ["role_*"], [
    "Failed logon attempts in the user-day", "Logon events", "Logons outside working hours", "Total session duration (minutes)", "File-access events", "Accesses to files labeled sensitive",
    "E-mails sent", "E-mails to external recipients", "Web upload events", "Removable-device connections", "Print jobs",
    "Department risk value (12 departments, uniform on [0.1, 1.0], independent of role and of scenario injection) plus daily jitter N(0, 0.03^2); tertiles define the Low/Medium/High tier", "One-hot organizational role (8 indicators)"])])
T["TB_2_generator_parameters"] = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in [
    ("Users / days per user", f"{C.N_USERS} / {C.DAYS_PER_USER}"), ("Roles", ", ".join(C.ROLES)), ("Malicious users / split", f"{C.N_MALICIOUS}; {C.MALICIOUS_SPLIT} (train/val/test)"),
    ("Per-feature distribution", "Poisson(lambda) per feature (session duration: Normal(mean, 60 min)), lambda = role baseline x user multiplier ~ LogNormal(0, 0.25^2); weekend (day%7 in {5,6}) activity x 0.15"),
    ("Role baselines (failed, login, afterhours, session, file, sensitive, mail, mail_ext, webup, usb, print)", json.dumps(C.ROLE_BASE)), ("Attack effect", "mean shift of z x s benign SDs (primary z=4, secondary z=2.5), s ~ U(intensity range) x INTENSITY_SCALE"),
    ("INTENSITY_SCALE", C.INTENSITY_SCALE), ("Archetypes: window length, intensity, effects", json.dumps(C.SCENARIOS)), ("Attack-window start", "Uniform integer in [20, 150 - length)"),
    ("RNG", C.RNG_TYPE), ("Seeds", C.SEEDS)]])
pl = F[rep]["details"]; T["T3_5b_dp_accounting"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in [
    ("delta (total)", C.DELTA), ("Composition", f"basic sequential over {C.ROUNDS * C.LOCAL_STEPS} noisy local steps (config.COMPOSITION='{C.COMPOSITION}')"), ("Minibatch size", C.BATCH_SIZE),
    ("Sampling / amplification", "Stratified minibatch (up to %d minority examples + negatives); NO privacy amplification by subsampling applied" % C.MIN_POS_PER_BATCH),
    ("Sensitivity assumed", "add/remove-one record: C / batch size (C = %.1f)" % C.CLIP_C), ("Per-step allocation", "eps_step = eps / T and delta_step = delta / T with T = %d" % pl["5.0"]["plan"]["n_compositions"]),
    ("Noise multiplier per eps (analytic Gaussian)", json.dumps({k: round(v["plan"]["noise_multiplier"], 4) for k, v in pl.items()})),
    ("Noise SD per coordinate per eps", json.dumps({k: round(v["plan"]["noise_std_per_coordinate"], 4) for k, v in pl.items()})), ("Accountant verification", "none (nominal budgets)")]])
ver = {p: (im.version(p) if p != "python" else platform.python_version()) for p in ["python", "numpy", "pandas", "scikit-learn", "scipy", "shap", "matplotlib", "openpyxl"] if p == "python" or im.distribution(p)}
T["T4_5_implementation_config"] = pd.DataFrame([{"Item": k, "Value": v} for k, v in [("Clients", f"{C.N_CLIENTS}; {C.SHARDS_PER_ROLE} shards per role dealt round-robin (non-IID by role)"), ("Rounds / local steps", f"{C.ROUNDS} / {C.LOCAL_STEPS}"),
    ("Optimizer", f"learning rate {C.LR}; L2 {C.L2}; minibatch {C.BATCH_SIZE}"), ("Class weight / clip", f"{C.MINORITY_WEIGHT} / {C.CLIP_C}"), ("epsilon grid / delta", f"{C.EPSILON_GRID} / {C.DELTA}"),
    ("Isolation Forest", C.IF_PARAMS), ("Random Forest", C.RF_PARAMS), ("Autoencoder", C.AE_PARAMS), ("Seeds", C.SEEDS), ("Software versions", ver)]])
g = json.load(open(os.path.join(C.OUT_DIR, "governance_report.json"))); T["T5_3c_governance"] = pd.DataFrame([{"Control": k, "Score": v} for k, v in g["scores"].items()] + [{"Control": "OVERALL (unweighted mean)", "Score": g["overall_index"]},
    {"Control": "Leave-one-out range", "Score": f"{g['loo_min']:.1f}–{g['loo_max']:.1f}"}])
json.dump(dict(config={k: v for k, v in vars(C).items() if k.isupper() and not isinstance(v, bytes)}, versions=ver, operating_point=op, seeds=seeds), open(os.path.join(C.OUT_DIR, "run_config.json"), "w"), indent=1, default=str)
# ---- write tables
with pd.ExcelWriter(os.path.join(C.OUT_DIR, "ALL_RESULTS.xlsx")) as xw:
    for n, d in T.items():
        d.to_csv(os.path.join(C.TAB_DIR, n + ".csv"), index=False)
        if len(d) < 100000: d.astype(object).where(d.notna(), "").to_excel(xw, sheet_name=n[:31], index=False)
# ---- figures
plt.rcParams.update({"font.size": 10, "figure.dpi": 100}); sv = lambda n: plt.savefig(os.path.join(C.FIG_DIR, n + ".png"), dpi=300, bbox_inches="tight"); 
bz = np.load(os.path.join(C.SEED_DIR, f"scores_baselines_seed_{rep}.npz")); fz = np.load(os.path.join(C.SEED_DIR, f"scores_fl_seed_{rep}.npz")); y = bz["y_test"]
plt.figure(figsize=(6, 5))
for n in ["Isolation Forest", "Random Forest", "Deep Autoencoder"]:
    f, t, _ = roc_curve(y, bz[f"{n}__test"]); plt.plot(f, t, label=f"{n} (AUC={B[rep]['metrics'][n]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k:", label="Chance"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC curves (seed {rep})"); plt.legend(); sv("fig5_1_roc_baselines"); plt.close()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4)); names = [("Isolation Forest", B[rep]["metrics"]["Isolation Forest"]), ("Random Forest", B[rep]["metrics"]["Random Forest"]), ("Deep Autoencoder", B[rep]["metrics"]["Deep Autoencoder"]), (f"FL+DP eps={C.REPORT_EPS:g}", F[rep]["metrics"][str(C.REPORT_EPS)])]
for a_, (n, m) in zip(ax, names):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]]); a_.imshow(np.log1p(cm), cmap="Blues")
    for i in range(2):
        for j in range(2): a_.text(j, i, cm[i, j], ha="center", va="center")
    a_.set_xticks([0, 1], ["Pred benign", "Pred malicious"]); a_.set_yticks([0, 1], ["True benign", "True malicious"]); a_.set_title(f"{n}\nF1={m['f1']:.3f}")
sv("fig5_2_confusion_matrices"); plt.close()
fk = [k for k in keys if k != "inf"]; x = [float(k) for k in fk]; fig, ax = plt.subplots(figsize=(6.5, 4.2))
for met, lab in (("f1", "F1"), ("pr_auc", "PR-AUC")):
    mu = [num([F[s]["metrics"][k][met] for s in seeds])[0] for k in fk]; sd = [num([F[s]["metrics"][k][met] for s in seeds])[1] for k in fk]; ax.errorbar(x, mu, yerr=sd, marker="o", capsize=3, label=lab)
ax.axhline(f1inf, ls="--", c="gray", label="non-private F1 (eps = inf)"); ax.set_xscale("log"); ax.set_xlabel("epsilon (log scale)"); ax.set_ylabel("score"); ax.set_title("Privacy-utility trade-off (mean ± SD, 5 seeds)"); ax.legend(); sv("fig5_3_privacy_utility"); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for k in ["0.5", "1.0", "10.0", "inf"]:
    L = np.array([F[s]["details"][k]["loss_curve"] for s in seeds]); ax.plot(L.mean(0), label=("eps=inf (non-private)" if k == "inf" else f"eps={k}")); ax.fill_between(range(L.shape[1]), L.mean(0) - L.std(0), L.mean(0) + L.std(0), alpha=.15)
ax.set_xlabel("Communication round"); ax.set_ylabel("Global training loss (binary cross-entropy)"); ax.set_title("FedAvg convergence (mean ± SD over seeds)"); ax.legend(); sv("fig5_3b_loss_convergence"); plt.close()
fig, ax = plt.subplots(figsize=(6.5, 4.2)); xs = np.arange(len(keys)); mu = [num([A[s]["mia"][f"FL+DP eps={k}"]["mia_auc"] for s in seeds])[0] for k in keys]; sd = [num([A[s]["mia"][f"FL+DP eps={k}"]["mia_auc"] for s in seeds])[1] for k in keys]
ax.errorbar(xs, mu, yerr=sd, marker="o", capsize=3); ax.axhline(0.5, ls=":", c="k", label="chance"); ax.set_xticks(xs, keys); ax.set_xlabel("epsilon"); ax.set_ylabel("MIA AUC"); ax.set_title("Membership-inference AUC vs epsilon"); ax.legend(); sv("fig5_4_mia_vs_epsilon"); plt.close()
import shap
z = np.load(os.path.join(C.SEED_DIR, f"shap_values_seed_{rep}.npz"), allow_pickle=True); shap.summary_plot(z["shap"], z["X"], feature_names=list(z["features"]), show=False); sv("fig5_5_shap_beeswarm"); plt.close()
shap.summary_plot(z["shap"], z["X"], feature_names=list(z["features"]), plot_type="bar", show=False); sv("fig5_5b_shap_bar"); plt.close()
gs = pd.DataFrame(T["T5_3c_governance"]).iloc[:-2]; fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(gs.Control, gs.Score.astype(float)); ax.axvline(g["overall_index"], c="r", ls="--", label=f"overall = {g['overall_index']:.1f}")
ax.set_xlim(0, 100); ax.set_xlabel("Score (0-100)"); ax.legend(); ax.set_title("Control-effectiveness scores"); sv("fig5_6_governance_scores"); plt.savefig(os.path.join(C.FIG_DIR, "fig4_3_control_effectiveness.png"), dpi=300, bbox_inches="tight"); plt.close()
# ---- markdown report
md = ["# PPUBA results report (auto-generated)", f"Seeds: {seeds}; representative seed {rep}. All numbers are mean ± SD (ddof=1) over seeds unless noted.\n"]
for n in ["T5_3a_detection_performance", "T5_3b_privacy_utility", "T5_3b_pareto_operating_point", "TC_1_partition_summary", "T5_3_3_membership_inference", "T5_3_4_shap_ranking_rep_seed", "T5_3c_governance"]:
    md += [f"## {n}", T[n].head(20).to_markdown(index=False), ""]
open(os.path.join(C.OUT_DIR, "RESULTS_REPORT.md"), "w").write("\n".join(md))
print(f"[07] wrote {len(T)} tables, figures and RESULTS_REPORT.md to {C.OUT_DIR}")
