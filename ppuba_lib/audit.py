"""Privacy and explainability audit (script 05): MIA, re-identification, SHAP, leakage checks (i)-(v)."""
import numpy as np, pandas as pd, shap
from scipy import stats
from sklearn.metrics import roc_auc_score
import config as C
from ppuba_lib.fl import sigmoid, run_sweep
from ppuba_lib.models import fit_baselines, eval_models

def _logloss(p, y): return -(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))

def mia_auc(member_score, nonmember_score):
    """Attack score = higher => more likely member. AUC against membership labels."""
    y = np.r_[np.ones(len(member_score)), np.zeros(len(nonmember_score))]
    return float(roc_auc_score(y, np.r_[member_score, nonmember_score]))

def run_mia(sp, feats, seed, fl_models, fitted, repeats=10):
    tr, te = sp["train"], sp["test"]; rng = np.random.default_rng(seed + 30_000); out = {}
    Xtr, ytr, Xte, yte = tr[feats].values, tr.is_malicious.values, te[feats].values, te.is_malicious.values
    def run(score_fn):
        v = []
        for _ in range(repeats):
            mi = rng.choice(len(tr), C.MIA_SAMPLES, replace=False); ni = rng.choice(len(te), C.MIA_SAMPLES, replace=False)
            v.append(mia_auc(-score_fn(Xtr[mi], ytr[mi]), -score_fn(Xte[ni], yte[ni])))
        return dict(mia_auc=float(np.mean(v)), mia_auc_sd_over_draws=float(np.std(v, ddof=1)))
    for k, (w, b) in fl_models.items():
        out[f"FL+DP eps={k}"] = run(lambda X, y, w=w, b=b: _logloss(sigmoid(X @ w + b), y))
    for name, (m, fn) in fitted.items():
        if name in ("Random Forest", "Centralized LR"): out[name] = run(lambda X, y, m=m: _logloss(m.predict_proba(X)[:, 1], y))
        else: out[name] = run(lambda X, y, fn=fn: fn(X))              # IF / AE: anomaly score (lower = more 'normal' = more member-like)
    return out

def reid_summary(anon):
    return dict(k_anonymity_records=anon["k_anonymity_records"], k_anonymity_users=anon["k_anonymity_users"], l_diversity=anon["l_diversity"],
                max_reidentification_risk=anon["max_reid_risk_records"], mean_reidentification_risk=anon["mean_reid_risk_records"],
                unique_records=anon["unique_records"], groups=anon["n_groups"])

def run_shap(sp, feats, rf, seed):
    te = sp["test"]; rng = np.random.default_rng(seed + 40_000)
    pos = np.flatnonzero(te.is_malicious.values == 1); neg = rng.choice(np.flatnonzero(te.is_malicious.values == 0), C.SHAP_NEG_SAMPLE, replace=False)
    idx = np.r_[pos, neg]; X = te.iloc[idx][feats]
    sv = shap.TreeExplainer(rf).shap_values(X.values)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    ma = np.abs(sv).mean(0); order = np.argsort(-ma); rows = []
    for r, i in enumerate(order):
        x, s_ = X.values[:, i], sv[:, i]
        corr = float(stats.spearmanr(x, s_)[0]) if x.std() > 0 and s_.std() > 0 else 0.0
        rows.append(dict(rank=r + 1, feature=feats[i], mean_abs_shap=float(ma[i]), spearman_value_vs_shap=corr,
                         direction=("high value -> toward anomaly" if corr > 0.1 else "high value -> toward normal" if corr < -0.1 else "mixed/none")))
    return rows, sv, X

def univariate_auc(sp, feats):
    out = {}
    for part in ("train", "val", "test"):
        d = sp[part]; out[part] = {f: float(roc_auc_score(d.is_malicious, d[f])) for f in feats if d.is_malicious.nunique() == 2}
    return out

def dept_risk_checks(raw_like):
    """(iv) does dept_risk_score vary with scenario injection, or only with department / role?"""
    d = raw_like; r = {}
    r["within_user_sd_mean"] = float(d.groupby("user").dept_risk_score.std().mean())
    r["between_user_sd"] = float(d.groupby("user").dept_risk_score.mean().std())
    um = d.groupby("user").agg(m=("dept_risk_score", "mean"), mal=("malicious_user", "max"), role=("role", "first"))
    r["mean_dept_risk_malicious_users"] = float(um[um.mal == 1].m.mean()); r["mean_dept_risk_benign_users"] = float(um[um.mal == 0].m.mean())
    mu = d[d.malicious_user == 1]; a, b = mu[mu.is_malicious == 1].dept_risk_score, mu[mu.is_malicious == 0].dept_risk_score
    r["malicious_users_window_vs_nonwindow_mean_diff"] = float(a.mean() - b.mean()); r["window_vs_nonwindow_welch_p"] = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    groups = [g.m.values for _, g in um.groupby("role")]; r["anova_role_p"] = float(stats.f_oneway(*groups).pvalue)
    r["varies_with_scenario_injection"] = bool(r["window_vs_nonwindow_welch_p"] < 0.01)
    return r

def ablation_runs(sp, seed):
    feats = [f for f in C.ALL_FEATURES if f not in ("sensitive_file_access", "dept_risk_score")]
    fitted = fit_baselines(sp, feats, seed); res, _ = eval_models(sp, feats, fitted)
    fl, *_ = run_sweep(sp, feats, seed, eps_grid=[float("inf"), C.REPORT_EPS])
    for k, v in fl.items(): res[f"FL+DP eps={k}"] = v
    return res

def shuffled_control(sp, seed):
    rng = np.random.default_rng(seed + 50_000); s2 = {"train": sp["train"].copy(), "val": sp["val"].copy(), "test": sp["test"]}
    for p in ("train", "val"): s2[p]["is_malicious"] = rng.permutation(s2[p].is_malicious.values)
    fitted = fit_baselines(s2, C.ALL_FEATURES, seed, models=("Random Forest", "Centralized LR")); res, _ = eval_models(s2, C.ALL_FEATURES, fitted)
    fl, *_ = run_sweep(s2, C.ALL_FEATURES, seed, eps_grid=[float("inf")])
    res["FedAvg eps=inf"] = fl["inf"]; res["test_prevalence"] = float(sp["test"].is_malicious.mean()); return res
