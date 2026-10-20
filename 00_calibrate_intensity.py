"""Script 00: a-priori calibration of the attack-intensity scale on SEPARATE calibration seeds (never used for evaluation).
Rule (fixed before any detector is trained): choose the smallest INTENSITY_SCALE on the grid for which the MEDIAN, over the 15 archetypes,
of the univariate ROC-AUC of the archetype's primary feature (attack-window days vs. all benign user-days) is >= TARGET_AUC.
No detector performance is used. The decision is written to calibrated_intensity.json and picked up by config.py."""
import json, os, sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import config as C
from ppuba_lib.data import generate_corpus
CAL_SEEDS = list(range(1001, 1009)); GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]; TARGET_AUC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
assert not set(CAL_SEEDS) & set(C.SEEDS)
primary = {n: next(iter(sp["fx"])) for n, sp in C.SCENARIOS.items()}; rows = []
for sc in GRID:
    per = {n: [] for n in C.SCENARIOS}
    for s in CAL_SEEDS:
        df = generate_corpus(s, sc); ben = df[df.malicious_user == 0]
        for n, f in primary.items():
            w = df[df.scenario == n]
            if len(w): per[n].append((roc_auc_score(np.r_[np.ones(len(w)), np.zeros(len(ben))], np.r_[w[f].values, ben[f].values]), len(w)))
    aucs = {n: float(np.average([a for a, _ in v], weights=[k for _, k in v])) for n, v in per.items() if v}
    rows.append(dict(scale=sc, median_primary_auc=float(np.median(list(aucs.values()))), min_primary_auc=min(aucs.values()), n_archetypes_observed=len(aucs), per_archetype=aucs))
    print(f"[00] scale {sc}: median primary-feature AUC = {rows[-1]['median_primary_auc']:.3f} (min {rows[-1]['min_primary_auc']:.3f})", flush=True)
ok = [r for r in rows if r["median_primary_auc"] >= TARGET_AUC]; chosen = (ok[0]["scale"] if ok else GRID[-1])
json.dump(dict(rule=f"smallest scale with median primary-feature univariate AUC >= {TARGET_AUC}", target_auc=TARGET_AUC, calibration_seeds=CAL_SEEDS, grid=rows, chosen_intensity_scale=chosen, target_met=bool(ok)),
          open(os.path.join(C.BASE_DIR, "calibrated_intensity.json"), "w"), indent=1)
print("[00] chosen INTENSITY_SCALE =", chosen, "| target met:", bool(ok))
