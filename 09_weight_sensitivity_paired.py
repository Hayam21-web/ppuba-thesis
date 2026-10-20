"""Script 09: (a) sensitivity of the governance-readiness index to the rubric weights (Table 5.4) and
(b) per-seed test F1 with paired differences FL + DP vs the non-private FedAvg reference (Appendix C, Table C.2).

(a) The per-criterion scores behind each control total were not recorded (governance_rubric_input.csv is empty, Section 4.4.4),
    so the effect of changing a criterion weight is BOUNDED: for each perturbed weight vector w' (one weight x0.5 or x1.5,
    renormalized), a linear program finds, for every control, the minimum and maximum score sum_j w'_j x_j over all criterion
    profiles x in [0, 100]^5 that reproduce the reported total under the original weights (sum_j w_j x_j = total). The index range
    is the mean over controls of these minima and maxima (worst case). In addition, the unweighted mean over controls is replaced
    by a weighted mean with Dirichlet(alpha) control weights (alpha = 1 and 5; 100,000 draws).
(b) Reads outputs/per_seed/{baselines,fl}_seed_*.json written by scripts 03-04. Paired difference d_s = F1(eps) - F1(inf) per seed;
    95% percentile-bootstrap interval from 10,000 resamples of the five seed-level differences (descriptive only, Section 5.6).
"""
import json, os, numpy as np, pandas as pd
from scipy.optimize import linprog
import config as C

RNG_SEED = 2026; N_DRAWS = 100_000; N_BOOT = 10_000; ALPHAS = (1.0, 5.0)

# ---------------------------------------------------------------- (a) governance weight sensitivity
W = C.RUBRIC_WEIGHTS; crit = list(W); w0 = np.array([W[c] for c in crit])
totals = np.array(list(C.GOVERNANCE_TOTALS.values()), float)

def bounds(w_new, total):
    A_eq, b_eq, bnd = [w0], [total], [(0, 100)] * len(crit)
    lo = linprog(w_new, A_eq=A_eq, b_eq=b_eq, bounds=bnd, method="highs").fun
    hi = -linprog(-w_new, A_eq=A_eq, b_eq=b_eq, bounds=bnd, method="highs").fun
    return lo, hi

LABEL = {"implementation": "Implementation completeness", "evidence": "Evidence availability", "auditability": "Auditability",
         "coverage": "Regulatory coverage", "effectiveness": "Privacy-control effectiveness"}
rows = []
for j, c in enumerate(crit):
    for f in (0.5, 1.5):
        w = w0.copy(); w[j] *= f; w = w / w.sum()
        b = np.array([bounds(w, t) for t in totals])
        rows.append({"Perturbation": f"{LABEL[c]} x {f:g}", "Renormalized criterion weights (impl./evid./audit./cov./effect.)": " / ".join(f"{x:.2f}" for x in w),
                     "Index min": round(b[:, 0].mean(), 1), "Index max": round(b[:, 1].mean(), 1),
                     "Governance-readiness index (worst-case range or distribution)": f"{b[:, 0].mean():.1f} - {b[:, 1].mean():.1f}"})
rng = np.random.default_rng(RNG_SEED)
for a in ALPHAS:
    wts = rng.dirichlet(np.full(len(totals), a), N_DRAWS); idx = wts @ totals
    lo, med, hi = np.percentile(idx, [2.5, 50, 97.5])
    rows.append({"Perturbation": f"Control-aggregation weights ~ Dirichlet(alpha = {a:g}), {N_DRAWS:,} draws",
                 "Renormalized criterion weights (impl./evid./audit./cov./effect.)": "criterion weights unchanged",
                 "Index min": round(lo, 1), "Index max": round(hi, 1),
                 "Governance-readiness index (worst-case range or distribution)":
                     f"median {med:.1f}; 95% interval {lo:.1f} - {hi:.1f}; {100 * (idx >= 80).mean():.1f}% of draws >= 80"})
T54 = pd.DataFrame(rows)
T54.to_csv(os.path.join(C.TAB_DIR, "T5_4_governance_weight_sensitivity.csv"), index=False)
print("[09] Table 5.4 (unweighted index = %.1f)" % totals.mean()); print(T54.iloc[:, [0, 4]].to_string(index=False))

# ---------------------------------------------------------------- (b) per-seed F1 and paired differences
J = lambda p: json.load(open(p))
seeds = [s for s in C.SEEDS if os.path.exists(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json"))]
B = {s: J(os.path.join(C.SEED_DIR, f"baselines_seed_{s}.json"))["metrics"] for s in seeds}
F = {s: J(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json"))["metrics"] for s in seeds}
eps = [k for k in F[seeds[0]] if k != "inf"]
cols = {"IF": lambda s: B[s]["Isolation Forest"]["f1"], "RF": lambda s: B[s]["Random Forest"]["f1"],
        "DAE": lambda s: B[s]["Deep Autoencoder"]["f1"], "LR": lambda s: B[s]["Centralized LR"]["f1"], "eps = inf": lambda s: F[s]["inf"]["f1"]}
for k in eps: cols[f"eps = {float(k):g}"] = (lambda s, k=k: F[s][k]["f1"])
tab = pd.DataFrame([{"Seed": s, **{n: fn(s) for n, fn in cols.items()}} for s in seeds])

def boot(d):
    r = np.random.default_rng(RNG_SEED); d = np.asarray(d, float)
    m = r.choice(d, (N_BOOT, len(d)), replace=True).mean(1)
    return d.mean(), *np.percentile(m, [2.5, 97.5])

out = [{k: (f"{v:.3f}" if k != "Seed" else str(v)) for k, v in row.items()} for row in tab.to_dict("records")]
out.append({"Seed": "Mean", **{n: f"{tab[n].mean():.3f}" for n in cols}})
pd_row = {"Seed": "Paired diff vs eps = inf [95% CI]", **{n: "-" for n in ["IF", "RF", "DAE", "LR", "eps = inf"]}}
for k in eps:
    m, l, h = boot(tab[f"eps = {float(k):g}"] - tab["eps = inf"]); pd_row[f"eps = {float(k):g}"] = f"{m:+.3f} [{l:+.3f}, {h:+.3f}]"
out.append(pd_row)
m, l, h = boot(tab["DAE"] - tab["IF"])
out.append({"Seed": "Paired diff DAE - IF [95% CI]", "DAE": f"{m:+.3f} [{l:+.3f}, {h:+.3f}]",
            "note": f"DAE > IF in {(tab['DAE'] > tab['IF']).sum()} of {len(tab)} seeds"})
TC2 = pd.DataFrame(out)
TC2.to_csv(os.path.join(C.TAB_DIR, "TC_2_per_seed_f1_paired.csv"), index=False)
print("\n[09] Table C.2"); print(TC2.fillna("").to_string(index=False))
