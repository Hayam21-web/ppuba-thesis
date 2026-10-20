"""Script 03: centralized baselines on every seed; saves metrics, per-record scores, time-to-detection and error cases."""
import argparse, json, os, time, numpy as np, pandas as pd
import config as C
from ppuba_lib.data import load_processed
from ppuba_lib.models import fit_baselines, eval_models
from ppuba_lib.metrics import time_to_detection
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS); a = ap.parse_args()
for s in a.seeds:
    t0 = time.time(); sp = load_processed(s); fitted = fit_baselines(sp, C.ALL_FEATURES, s)
    res, sc = eval_models(sp, C.ALL_FEATURES, fitted); train_time = time.time() - t0
    te = sp["test"]; ttd = {}; cases = []
    for n in res:
        ttd[n] = time_to_detection(te, sc[n]["test"], res[n]["threshold"])
        yp = (sc[n]["test"] >= res[n]["threshold"]).astype(int); e = te[(yp != te.is_malicious.values)].copy()
        e["detector"] = n; e["error_type"] = np.where(e.is_malicious == 1, "FN", "FP"); e["score"] = sc[n]["test"][e.index]; e["seed"] = s
        cases.append(e[["seed", "detector", "error_type", "user", "role", "day", "scenario", "attack_intensity", "score"] + C.NUMERIC_FEATURES])
    pd.concat(cases).to_csv(os.path.join(C.SEED_DIR, f"error_cases_baselines_seed_{s}.csv"), index=False)
    np.savez(os.path.join(C.SEED_DIR, f"scores_baselines_seed_{s}.npz"), y_val=sp["val"].is_malicious.values, y_test=te.is_malicious.values,
             **{f"{n}__{k}": v for n in sc for k, v in sc[n].items()})
    json.dump(dict(seed=s, metrics=res, time_to_detection=ttd, train_and_eval_seconds=train_time), open(os.path.join(C.SEED_DIR, f"baselines_seed_{s}.json"), "w"), indent=1, default=float)
    print(f"[03] seed {s}:", {n: round(r["f1"], 3) for n, r in res.items()}, f"({train_time:.0f}s)")
