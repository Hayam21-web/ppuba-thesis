import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

def best_threshold(y, s):
    """Threshold maximizing F1 on the validation precision-recall curve (Section 3.7/3.8)."""
    p, r, t = precision_recall_curve(y, s)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)[:-1]
    if len(t) == 0: return 0.5
    best = np.flatnonzero(f == f.max()); return float(t[best[len(best) // 2]])

def recall_at_fpr(y, s, level):
    fpr, tpr, _ = roc_curve(y, s); ok = fpr <= level
    return float(tpr[ok].max()) if ok.any() else 0.0

def evaluate(y, s, thr, low_fpr=(0.001, 0.01)):
    yp = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m = dict(precision=float(prec), recall=float(rec), f1=float(f1), pr_auc=float(average_precision_score(y, s)),
             roc_auc=float(roc_auc_score(y, s)), fpr=float(fp / (fp + tn)), fnr=float(fn / (fn + tp)),
             tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), threshold=float(thr), alerts=int(yp.sum()),
             false_alerts_per_1000_userdays=float(1000 * fp / len(y)))
    for l in low_fpr: m[f"recall_at_fpr_{l}"] = recall_at_fpr(y, s, l)
    return m

def time_to_detection(df, s, thr):
    """Days from attack-window start to first alert, per malicious user (NaN = never detected)."""
    out = []
    for u, g in df.assign(score=s).groupby("user"):
        if g.malicious_user.iloc[0] != 1: continue
        w = g[g.is_malicious == 1].sort_values("day"); hit = w[w.score >= thr]
        out.append(dict(user=u, scenario=w.scenario.iloc[0], role=w.role.iloc[0], window_days=len(w),
                        first_alert_day=(int(hit.day.iloc[0] - w.day.iloc[0]) if len(hit) else np.nan),
                        days_flagged=int(len(hit))))
    return out
