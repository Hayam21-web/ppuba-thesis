"""Script 04: FL + DP sweep over the epsilon grid on every seed (+ client-scalability and composition-sensitivity runs)."""
import argparse, json, os, pickle, time, numpy as np, pandas as pd
import config as C
from ppuba_lib.data import load_processed
from ppuba_lib.fl import run_sweep
from ppuba_lib.metrics import time_to_detection
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS)
ap.add_argument("--no_extras", action="store_true"); a = ap.parse_args()
for s in a.seeds:
    sp = load_processed(s); t0 = time.time()
    res, models, scores, det, csz = run_sweep(sp, C.ALL_FEATURES, s)
    te = sp["test"]; key = str(C.REPORT_EPS); ttd = {k: time_to_detection(te, scores[k]["test"], res[k]["threshold"]) for k in res}
    cases = []
    for k in res:
        yp = (scores[k]["test"] >= res[k]["threshold"]).astype(int); e = te[yp != te.is_malicious.values].copy()
        e["detector"] = f"FL+DP eps={k}"; e["error_type"] = np.where(e.is_malicious == 1, "FN", "FP"); e["score"] = scores[k]["test"][e.index]; e["seed"] = s
        cases.append(e[["seed", "detector", "error_type", "user", "role", "day", "scenario", "attack_intensity", "score"] + C.NUMERIC_FEATURES])
    pd.concat(cases).to_csv(os.path.join(C.SEED_DIR, f"error_cases_fl_seed_{s}.csv"), index=False)
    with open(os.path.join(C.SEED_DIR, f"fl_models_seed_{s}.pkl"), "wb") as f: pickle.dump(models, f)
    np.savez(os.path.join(C.SEED_DIR, f"scores_fl_seed_{s}.npz"), y_val=sp["val"].is_malicious.values, y_test=te.is_malicious.values,
             **{f"eps_{k}__{n}": v[n] for k, v in scores.items() for n in v})
    out = dict(seed=s, metrics=res, details=det, client_sizes_and_positives=csz, time_to_detection=ttd, sweep_seconds=time.time() - t0)
    if not a.no_extras:
        sc = {}
        for n in C.SCALABILITY_CLIENTS:
            r, *_ = run_sweep(sp, C.ALL_FEATURES, s, eps_grid=[float("inf"), C.REPORT_EPS], n_clients=n); sc[str(n)] = r
        out["scalability"] = sc
        r, *_ = run_sweep(sp, C.ALL_FEATURES, s, composition="rounds"); out["composition_rounds_sensitivity"] = r
    json.dump(out, open(os.path.join(C.SEED_DIR, f"fl_seed_{s}.json"), "w"), indent=1, default=float)
    print(f"[04] seed {s}: F1 by eps =", {k: round(v["f1"], 3) for k, v in res.items()}, f"({time.time()-t0:.0f}s)")
