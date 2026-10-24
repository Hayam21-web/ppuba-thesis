"""Script 10: external validation on ADFA-LD and on the Enron persons-of-interest data (Section 5.6.1, Table 5.5, Appendix B.3).

The same detectors, hyperparameters (config.py), validation-only threshold rule and seeds 42-46 as in Chapter 5 are applied,
without any re-tuning, to two public data sets of a different kind:

ADFA-LD (Creech & Hu, 2014) - Linux host system-call traces: 5,205 normal traces and 746 attack traces from six attack
  families (60 attack runs). Features (20): relative frequency of the 19 system calls that are most frequent in the
  normal training traces of the seed, plus log trace length. Group-aware split 60/15/25: every attack run (e.g. Adduser_3)
  and every normal trace lies in one partition only. The full pipeline is run: Isolation Forest, Random Forest, Deep
  Autoencoder, centralized LR, FedAvg + DP over the epsilon grid (10 virtual clients, ppuba_lib/fl.py unchanged), and the
  loss-based membership-inference attack (1,000 member / 1,000 non-member traces instead of 2,000, because the test
  partition holds about 1,490 traces).

Enron persons of interest (Udacity ud120 release of the Enron e-mail and payment data) - 144 employees after removing the
  two non-person entries, 18 persons of interest (POI). Only label-independent features are used: e-mail volume
  (to_messages, from_messages) and eight payment fields (signed log1p, missing = 0). The three POI-interaction counts
  (from_poi_to_this_person, from_this_person_to_poi, shared_receipt_with_poi) are excluded because they are computed from
  the POI labels themselves. Stratified split by person 60/15/25. Only the centralized detectors are run: with about 86
  training persons, ten clients would hold fewer records than one minibatch (B = 256), so the DP calibration of
  ppuba_lib/fl.py (sensitivity C/B) would not hold.

Data (not redistributed; place the files under external_data/):
  external_data/ADFA-LD.zip                  from https://github.com/verazuo/a-labelled-version-of-the-ADFA-LD-dataset
  external_data/final_project_dataset.pkl    from https://github.com/udacity/ud120-projects (final_project/)
  python 10_external_validation.py            -> outputs/tables/EXT_*.csv, outputs/per_seed/external_*.json
"""
import hashlib, io, json, os, pickle, re, zipfile
import numpy as np, pandas as pd
import config as C
from ppuba_lib.models import fit_baselines, eval_models
from ppuba_lib.fl import run_sweep
from ppuba_lib.audit import run_mia

EXT = os.path.join(C.BASE_DIR, "external_data")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"


def split_groups(groups, labels, seed, frac=(0.60, 0.15, 0.25)):
    """Assign whole groups to train/val/test, stratified by the group's class."""
    rng = np.random.default_rng(seed + 10_000); part = {}
    g = pd.DataFrame({"g": groups, "y": labels}).groupby("g").y.max()
    for cls in (0, 1):
        ids = rng.permutation(np.array(sorted(g[g == cls].index), dtype=object)); n = len(ids)
        a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
        for i, gid in enumerate(ids): part[gid] = "train" if i < a else ("val" if i < b else "test")
    return np.array([part[x] for x in groups])


def standardize(df, feats):
    tr = df[df.part == "train"]; mu, sd = tr[feats].mean(), tr[feats].std(ddof=0).replace(0, 1.0)
    df[feats] = (df[feats] - mu) / sd
    return {k: df[df.part == k].reset_index(drop=True) for k in ("train", "val", "test")}


# ------------------------------------------------------------------ ADFA-LD
def load_adfa(path):
    z = zipfile.ZipFile(path); rows = []
    for n in z.namelist():
        if not n.endswith(".txt") or "/" not in n: continue
        parts = n.split("/")
        if "Attack_Data_Master" in parts:
            fam_run = parts[parts.index("Attack_Data_Master") + 1]; label, group = 1, fam_run
            family = re.sub(r"_\d+$", "", fam_run)
        elif "Training_Data_Master" in parts or "Validation_Data_Master" in parts:
            label, group, family = 0, n, "normal"
        else:
            continue
        calls = np.array(z.read(n).decode().split(), dtype=int)
        if len(calls): rows.append((group, label, family, calls))
    return rows


def adfa_seed(rows, seed):
    groups = np.array([r[0] for r in rows], dtype=object); y = np.array([r[1] for r in rows])
    part = split_groups(groups, y, seed)
    # vocabulary: 19 most frequent system calls in the NORMAL TRAINING traces of this seed
    cnt = {}
    for (g, lab, fam, calls), p in zip(rows, part):
        if p == "train" and lab == 0:
            u, c = np.unique(calls, return_counts=True)
            for a, b in zip(u, c): cnt[a] = cnt.get(a, 0) + b
    vocab = [k for k, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:19]]
    feats = [f"sc_{v}" for v in vocab] + ["log_length"]
    X = np.zeros((len(rows), 20))
    for i, (_, _, _, calls) in enumerate(rows):
        for j, v in enumerate(vocab): X[i, j] = np.mean(calls == v)
        X[i, 19] = np.log(len(calls))
    rng = np.random.default_rng(seed + 20_000)
    df = pd.DataFrame(X, columns=feats); df["is_malicious"] = y; df["part"] = part; df["user"] = groups
    df["role"] = rng.choice(C.ROLES, len(df)); df["scenario"] = [r[2] for r in rows]
    return standardize(df, feats), feats


# ------------------------------------------------------------------ Enron POI
ENRON_FEATS = ["to_messages", "from_messages", "salary", "bonus", "total_payments", "exercised_stock_options",
               "restricted_stock", "expenses", "long_term_incentive", "deferred_income"]


def load_enron(path):
    d = pickle.load(open(path, "rb"))
    for k in ("TOTAL", "THE TRAVEL AGENCY IN THE PARK"): d.pop(k, None)
    df = pd.DataFrame.from_dict(d, orient="index")
    out = pd.DataFrame(index=df.index)
    for f in ENRON_FEATS:
        v = pd.to_numeric(df[f].replace("NaN", np.nan), errors="coerce").fillna(0.0)
        out[f] = np.sign(v) * np.log1p(np.abs(v))
    out["is_malicious"] = df["poi"].astype(int).values
    return out


def enron_seed(base, seed):
    df = base.copy(); df["user"] = df.index.values
    df["part"] = split_groups(df.user.values, df.is_malicious.values, seed); df["role"] = "n/a"
    return standardize(df.reset_index(drop=True), ENRON_FEATS)


# ------------------------------------------------------------------ run
if __name__ == "__main__":
    pa, pe = os.path.join(EXT, "ADFA-LD.zip"), os.path.join(EXT, "final_project_dataset.pkl")
    print("ADFA-LD.zip sha256:", sha(pa)); print("final_project_dataset.pkl sha256:", sha(pe))
    C.MIA_SAMPLES = 1000
    rows = load_adfa(pa); print(f"[10] ADFA-LD: {len(rows)} traces, {sum(r[1] for r in rows)} attacks, "
                                f"{len({r[0] for r in rows if r[1]})} attack runs")
    A = {"baseline": {}, "fl": {}, "mia": {}, "partition": {}}
    for s in C.SEEDS:
        sp, feats = adfa_seed(rows, s)
        fitted = fit_baselines(sp, feats, s); res, _ = eval_models(sp, feats, fitted)
        fres, fmodels, _, _, csz = run_sweep(sp, feats, s)
        assert min(n for n, _ in csz) >= C.BATCH_SIZE, "a client holds fewer records than one minibatch"
        A["baseline"][s] = res; A["fl"][s] = fres
        A["mia"][s] = run_mia(sp, feats, s, fmodels, fitted, repeats=10)
        A["partition"][s] = {k: dict(records=len(sp[k]), attacks=int(sp[k].is_malicious.sum())) for k in ("train", "val", "test")}
        A["partition"][s]["min_client_records"] = int(min(n for n, _ in csz))
        print(f"[10] ADFA seed {s}: " + ", ".join(f"{k} {v['f1']:.3f}" for k, v in res.items())
              + f" | FL eps=inf {fres['inf']['f1']:.3f}, eps=5 {fres['5.0']['f1']:.3f}")
    base = load_enron(pe); print(f"[10] Enron: {len(base)} persons, {int(base.is_malicious.sum())} POI")
    E = {"baseline": {}, "partition": {}}
    for s in C.SEEDS:
        sp = enron_seed(base, s); fitted = fit_baselines(sp, ENRON_FEATS, s); res, _ = eval_models(sp, ENRON_FEATS, fitted)
        E["baseline"][s] = res
        E["partition"][s] = {k: dict(persons=len(sp[k]), poi=int(sp[k].is_malicious.sum())) for k in ("train", "val", "test")}
        print(f"[10] Enron seed {s}: " + ", ".join(f"{k} {v['f1']:.3f}" for k, v in res.items()))
    json.dump(A, open(os.path.join(C.SEED_DIR, "external_adfa.json"), "w"), indent=1, default=float)
    json.dump(E, open(os.path.join(C.SEED_DIR, "external_enron.json"), "w"), indent=1, default=float)

    met = ["pr_auc", "precision", "recall", "f1", "roc_auc", "fpr"]
    rowsT = []
    for name in ("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR"):
        rowsT.append(["ADFA-LD", name] + [fmt([A["baseline"][s][name][m] for s in C.SEEDS]) for m in met])
    for k, lab in (("5.0", "FL + DP (eps = 5)"), ("inf", "FedAvg non-private (eps = inf)")):
        rowsT.append(["ADFA-LD", lab] + [fmt([A["fl"][s][k][m] for s in C.SEEDS]) for m in met])
    for name in ("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR"):
        rowsT.append(["Enron POI", name] + [fmt([E["baseline"][s][name][m] for s in C.SEEDS]) for m in met])
    pd.DataFrame(rowsT, columns=["Dataset", "Model", "PR-AUC", "Precision", "Recall", "F1", "ROC-AUC", "FPR"]).to_csv(
        os.path.join(C.TAB_DIR, "EXT_detection_performance.csv"), index=False)
    pu = []
    for k in ("0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "inf"):
        f = [A["fl"][s][k]["f1"] for s in C.SEEDS]; ref = np.mean([A["fl"][s]["inf"]["f1"] for s in C.SEEDS])
        pu.append([k, fmt(f), fmt([A["fl"][s][k]["pr_auc"] for s in C.SEEDS]), fmt([A["fl"][s][k]["roc_auc"] for s in C.SEEDS]),
                   round(100 * np.mean(f) / ref, 1) if ref > 0 else float("nan"),
                   fmt([A["mia"][s][f"FL+DP eps={k}"]["mia_auc"] for s in C.SEEDS])])
    pd.DataFrame(pu, columns=["epsilon", "F1", "PR-AUC", "ROC-AUC", "F1 retention vs eps=inf (%)", "MIA AUC"]).to_csv(
        os.path.join(C.TAB_DIR, "EXT_adfa_privacy_utility.csv"), index=False)
    mia_c = [[n, fmt([A["mia"][s][n]["mia_auc"] for s in C.SEEDS])] for n in ("Isolation Forest", "Random Forest", "Deep Autoencoder", "Centralized LR")]
    pd.DataFrame(mia_c, columns=["model", "MIA AUC"]).to_csv(os.path.join(C.TAB_DIR, "EXT_adfa_mia_centralized.csv"), index=False)
    part = [[ds, s, k, v[k]["records" if ds == "ADFA-LD" else "persons"], v[k]["attacks" if ds == "ADFA-LD" else "poi"]]
            for ds, P in (("ADFA-LD", A["partition"]), ("Enron POI", E["partition"])) for s, v in P.items() for k in ("train", "val", "test")]
    pd.DataFrame(part, columns=["dataset", "seed", "partition", "records", "positives"]).to_csv(os.path.join(C.TAB_DIR, "EXT_partitions.csv"), index=False)
    print("[10] wrote outputs/tables/EXT_detection_performance.csv, EXT_adfa_privacy_utility.csv, EXT_adfa_mia_centralized.csv, EXT_partitions.csv")
