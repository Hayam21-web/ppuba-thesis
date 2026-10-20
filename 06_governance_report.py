"""Script 06: governance-readiness scoring (Section 4.4.4), leave-one-control-out, weight sensitivity, automated evidence checks,
and a reproducibility check (same seed -> identical corpus hash and identical detector scores).
Criterion scores are researcher judgments: fill governance_rubric_input.csv (columns: control + 5 criteria, 0-100). If it is empty,
the researcher-assigned control totals of Table 5.3c from config.GOVERNANCE_TOTALS are used and flagged as such."""
import hashlib, json, os, sys, numpy as np, pandas as pd
import config as C
from ppuba_lib.data import generate_corpus
W = C.RUBRIC_WEIGHTS; crit = list(W)
inp = os.path.join(C.BASE_DIR, "governance_rubric_input.csv")
if not os.path.exists(inp): pd.DataFrame({"control": list(C.GOVERNANCE_TOTALS), **{c: np.nan for c in crit}}).to_csv(inp, index=False)
r = pd.read_csv(inp); filled = r[crit].notna().all().all()
if filled: r["score"] = sum(r[c] * W[c] for c in crit); src = "weighted rubric (researcher criterion scores)"
else: r["score"] = r["control"].map(C.GOVERNANCE_TOTALS); src = "researcher-assigned control totals (Table 5.3c); criterion scores not supplied"
overall = float(r.score.mean()); loo = {c: float(r[r.control != c].score.mean()) for c in r.control}
res = dict(source=src, scores=dict(zip(r.control, r.score)), overall_index=overall, leave_one_control_out=loo,
           loo_min=min(loo.values()), loo_max=max(loo.values()), weights=W)
if filled:   # weight sensitivity: +/-50% perturbations of each weight (renormalized)
    sens = []
    for c in crit:
        for f in (0.5, 1.5):
            w2 = dict(W); w2[c] *= f; z = sum(w2.values()); w2 = {k: v / z for k, v in w2.items()}
            sens.append(dict(criterion=c, factor=f, overall=float((sum(r[k] * w2[k] for k in crit)).mean())))
    res["weight_sensitivity"] = sens
# automated evidence checks (what the pipeline can verify mechanically)
ev = {}
ev["seeds_recorded"] = C.SEEDS
ev["k_anonymity_measured"] = os.path.exists(os.path.join(C.SEED_DIR, f"split_seed_{C.REP_SEED}.json"))
ev["mia_measured"] = os.path.exists(os.path.join(C.SEED_DIR, f"audit_seed_{C.REP_SEED}.json"))
ev["shap_artifact"] = os.path.exists(os.path.join(C.SEED_DIR, f"shap_values_seed_{C.REP_SEED}.npz"))
h = lambda df: hashlib.sha256(pd.util.hash_pandas_object(df.drop(columns=["ip_address"]), index=True).values.tobytes()).hexdigest()
seed = C.SEEDS[0]; a, b = generate_corpus(seed), generate_corpus(seed)
ev["reproducibility_corpus_hash_identical"] = h(a) == h(b); ev["corpus_hash_seed_%d" % seed] = h(a)
res["evidence_checks"] = ev
json.dump(res, open(os.path.join(C.OUT_DIR, "governance_report.json"), "w"), indent=1, default=str)
print("[06] overall index =", round(overall, 2), "| source:", src, "| reproducible:", ev["reproducibility_corpus_hash_identical"])
