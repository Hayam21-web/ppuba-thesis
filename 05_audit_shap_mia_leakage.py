"""Script 05: MIA, re-identification, SHAP, and leakage checks (i)-(v) of Section 3.4 (reported in Appendix C)."""
import argparse, json, os, pickle, numpy as np, pandas as pd
import config as C
from ppuba_lib.data import load_processed
from ppuba_lib.models import fit_baselines
from ppuba_lib import audit as A
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS); ap.add_argument("--skip_leakage", action="store_true"); a = ap.parse_args()
for s in a.seeds:
    sp = load_processed(s); fitted = fit_baselines(sp, C.ALL_FEATURES, s)
    fl_models = pickle.load(open(os.path.join(C.SEED_DIR, f"fl_models_seed_{s}.pkl"), "rb"))
    out = dict(seed=s, mia=A.run_mia(sp, C.ALL_FEATURES, s, fl_models, fitted), reidentification=A.reid_summary(sp["anon"]))
    rows, sv, X = A.run_shap(sp, C.ALL_FEATURES, fitted["Random Forest"][0], s)
    out["shap"] = dict(n_explained=int(len(X)), n_positive=int((sp["test"].is_malicious == 1).sum()), ranking=rows)
    np.savez(os.path.join(C.SEED_DIR, f"shap_values_seed_{s}.npz"), shap=sv, X=X.values, features=np.array(C.ALL_FEATURES))
    if not a.skip_leakage:
        raw = pd.read_csv(os.path.join(C.DATA_DIR, f"raw_corpus_seed_{s}.csv")).rename(columns={"user_name": "user"})
        out["leakage"] = dict(
            i_no_user_in_multiple_partitions=True,   # asserted inside preprocess(); a violation aborts script 02
            ii_univariate_auc=A.univariate_auc(sp, C.NUMERIC_FEATURES),
            iii_ablation_without_sensitive_file_access_and_dept_risk_score=A.ablation_runs(sp, s),
            iv_dept_risk_score=A.dept_risk_checks(raw),
            v_shuffled_label_control=A.shuffled_control(sp, s))
    json.dump(out, open(os.path.join(C.SEED_DIR, f"audit_seed_{s}.json"), "w"), indent=1, default=float)
    print(f"[05] seed {s}: MIA(eps=5)={out['mia']['FL+DP eps=5.0']['mia_auc']:.3f}  top SHAP={[r['feature'] for r in rows[:3]]}")
