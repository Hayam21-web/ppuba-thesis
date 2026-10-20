"""Centralized detectors (script 03): Isolation Forest, Random Forest, Deep Autoencoder (+ centralized logistic regression reference)."""
import time, warnings, numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import config as C
from ppuba_lib.metrics import best_threshold, evaluate
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def fit_baselines(sp, feats, seed, y_override=None, models=("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR")):
    """Returns dict name -> dict(model, score_fn). y_override: shuffled labels for the control experiment."""
    tr = sp["train"]; Xtr = tr[feats].values; ytr = tr.is_malicious.values if y_override is None else y_override
    out = {}
    if "Isolation Forest" in models:
        m = IsolationForest(random_state=seed, **C.IF_PARAMS).fit(Xtr[ytr == 0]); out["Isolation Forest"] = (m, lambda X, m=m: -m.score_samples(X))
    if "Random Forest" in models:
        m = RandomForestClassifier(random_state=seed, **C.RF_PARAMS).fit(Xtr, ytr); out["Random Forest"] = (m, lambda X, m=m: m.predict_proba(X)[:, 1])
    if "Deep Autoencoder" in models:
        m = MLPRegressor(random_state=seed, **C.AE_PARAMS).fit(Xtr[ytr == 0], Xtr[ytr == 0])
        out["Deep Autoencoder"] = (m, lambda X, m=m: ((m.predict(X) - X) ** 2).mean(1))
    if "Centralized LR" in models:
        m = LogisticRegression(C=1 / (C.L2 * len(ytr)), class_weight={0: 1, 1: C.MINORITY_WEIGHT}, max_iter=500).fit(Xtr, ytr)
        out["Centralized LR"] = (m, lambda X, m=m: m.predict_proba(X)[:, 1])
    return out

def eval_models(sp, feats, fitted):
    va, te = sp["val"], sp["test"]; res, scores = {}, {}
    for name, (m, fn) in fitted.items():
        sva = fn(va[feats].values); t0 = time.perf_counter(); ste = fn(te[feats].values); dt = time.perf_counter() - t0
        thr = best_threshold(va.is_malicious.values, sva)
        r = evaluate(te.is_malicious.values, ste, thr); r["inference_seconds_test"] = dt
        r["throughput_records_per_s"] = len(te) / dt if dt > 0 else float("inf"); r["latency_ms_per_record"] = 1000 * dt / len(te)
        res[name] = r; scores[name] = dict(val=sva, test=ste)
    return res, scores
