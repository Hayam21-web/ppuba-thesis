"""Script 08: sensitivity of every conclusion to the attack-intensity design choice (INTENSITY_SCALE x archetype z-shifts).
Reports detector F1 / PR-AUC / ROC-AUC and FL+DP (eps = inf and 5) for several intensity scales, so the thesis can state how
dependent the findings are on the author-defined generator (leakage/separability limitation, Section 3.4)."""
import argparse, json, os, numpy as np, pandas as pd
import config as C
from ppuba_lib.data import generate_corpus, preprocess
from ppuba_lib.models import fit_baselines, eval_models
from ppuba_lib.fl import run_sweep
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS[:3]); ap.add_argument("--scales", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0, 3.0]); a = ap.parse_args()
rows = []
for sc in a.scales:
    for s in a.seeds:
        sp = preprocess(s, generate_corpus(s, sc).assign(), save=False)
        fit = fit_baselines(sp, C.ALL_FEATURES, s); res, _ = eval_models(sp, C.ALL_FEATURES, fit)
        fl, *_ = run_sweep(sp, C.ALL_FEATURES, s, eps_grid=[float("inf"), C.REPORT_EPS])
        for n, m in {**res, "FedAvg eps=inf": fl["inf"], f"FL+DP eps={C.REPORT_EPS:g}": fl[str(C.REPORT_EPS)]}.items():
            rows.append(dict(intensity_scale=sc, seed=s, model=n, f1=m["f1"], pr_auc=m["pr_auc"], roc_auc=m["roc_auc"], precision=m["precision"], recall=m["recall"]))
    print(f"[08] scale {sc} done")
df = pd.DataFrame(rows); df.to_csv(os.path.join(C.TAB_DIR, "SENS_intensity_all_runs.csv"), index=False)
g = df.groupby(["intensity_scale", "model"])[["f1", "pr_auc", "roc_auc"]].agg(["mean", "std"]).round(3); g.to_csv(os.path.join(C.TAB_DIR, "SENS_intensity_summary.csv")); print(g)
