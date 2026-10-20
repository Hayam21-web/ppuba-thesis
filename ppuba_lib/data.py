"""Corpus generation (script 01) and preprocessing/splitting (script 02)."""
import hashlib, hmac, os, pickle, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import config as C

def mask_identifier(val: str) -> str:
    return hmac.new(C.HMAC_KEY, val.encode(), hashlib.sha256).hexdigest()[:16]

def generate_corpus(seed: int, intensity_scale: float = None) -> pd.DataFrame:
    scale_g = C.INTENSITY_SCALE if intensity_scale is None else intensity_scale
    ss = np.random.SeedSequence(seed); r_meta, r_user = [np.random.default_rng(s) for s in ss.spawn(2)]
    roles = r_meta.choice(C.ROLES, C.N_USERS)
    dept = r_meta.integers(0, C.N_DEPARTMENTS, C.N_USERS)
    dept_risk = np.round(r_meta.uniform(*C.DEPT_RISK_RANGE, C.N_DEPARTMENTS), 2)
    mal = set(r_meta.choice(C.N_USERS, C.N_MALICIOUS, replace=False).tolist())
    names = list(C.SCENARIOS)
    d = np.arange(C.DAYS_PER_USER); weekend = np.isin(d % 7, (5, 6))
    dayf = np.where(weekend, C.WEEKEND_FACTOR, 1.0)
    frames = []
    for u in range(C.N_USERS):
        rng = np.random.default_rng(r_user.integers(2**63))
        base = np.array(C.ROLE_BASE[roles[u]]) * rng.lognormal(0, C.USER_HETEROGENEITY_SIGMA, 11)
        lam = np.tile(base, (C.DAYS_PER_USER, 1)); lam[:, [i for i in range(11) if i != 3]] *= dayf[:, None]
        scen, lab, win = "none", np.zeros(C.DAYS_PER_USER, int), None
        if u in mal:
            scen = names[int(rng.integers(len(names)))]; sp = C.SCENARIOS[scen]
            L = int(rng.integers(sp["window"][0], sp["window"][1] + 1)); st = int(rng.integers(20, C.DAYS_PER_USER - L))
            s = float(rng.uniform(*sp["intensity"])) * scale_g
            for f, z in sp["fx"].items():
                j = C.BASE_ORDER.index(f); mu = base[j]
                sd = np.sqrt(C.SESSION_SD_MIN ** 2 + (C.USER_HETEROGENEITY_SIGMA * mu) ** 2) if j == 3 else np.sqrt(mu + (C.USER_HETEROGENEITY_SIGMA * mu) ** 2)
                lam[st:st + L, j] = lam[st:st + L, j] + z * s * sd
            lab[st:st + L] = 1; win = (st, L, s)
        X = rng.poisson(np.clip(lam, 0, None)).astype(float)
        X[:, 3] = np.clip(rng.normal(lam[:, 3], C.SESSION_SD_MIN), 30, None)
        df = pd.DataFrame(X, columns=C.BASE_ORDER)
        df["dept_risk_score"] = np.clip(dept_risk[dept[u]] + rng.normal(0, C.DEPT_RISK_DAILY_JITTER, C.DAYS_PER_USER), 0.1, 1.0)
        df.insert(0, "user_name", f"user{u:04d}"); df.insert(1, "ip_address", f"10.{u // 250}.{u % 250}.{int(rng.integers(1, 250))}")
        df.insert(2, "file_path_root", f"/share/{roles[u].lower()}/u{u:04d}")
        df["day"] = d; df["role"] = roles[u]; df["department"] = int(dept[u]); df["is_malicious"] = lab
        df["malicious_user"] = int(u in mal); df["scenario"] = np.where(lab == 1, scen, "none")
        df["attack_start"] = win[0] if win else -1; df["attack_len"] = win[1] if win else 0; df["attack_intensity"] = win[2] if win else 0.0
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def kanon(df: pd.DataFrame, user_col="user") -> dict:
    t = df.copy(); t["tier"] = pd.qcut(t["dept_risk_score"], 3, labels=["Low", "Medium", "High"])
    g = t.groupby(["role", "tier"], observed=True)
    sizes = g.size(); k = int(sizes.min()); lmin = int(g["is_malicious"].nunique().min())
    users = g[user_col].nunique()
    return dict(k_anonymity_records=k, k_anonymity_users=int(users.min()), l_diversity=lmin, n_groups=int(len(sizes)),
                max_reid_risk_records=1.0 / k, mean_reid_risk_records=float((1.0 / sizes * sizes / sizes.sum()).sum()),
                unique_records=int((sizes == 1).sum()), quasi_identifiers=["role", "dept_risk_tier"])

def preprocess(seed: int, raw: pd.DataFrame = None, save: bool = True) -> dict:
    raw = pd.read_csv(os.path.join(C.DATA_DIR, f"raw_corpus_seed_{seed}.csv")) if raw is None else raw.copy()
    raw["user"] = raw["user_name"].map(mask_identifier)                     # pseudonymize user name
    raw["ip_token"] = raw["ip_address"].map(mask_identifier); raw["path_token"] = raw["file_path_root"].map(mask_identifier)
    raw = raw.drop(columns=["user_name", "ip_address", "file_path_root"])
    anon = kanon(raw)
    rng = np.random.default_rng(seed + 10_000)
    u = raw.groupby("user").agg(mal=("malicious_user", "max"), role=("role", "first")).reset_index()
    mal = u[u.mal == 1].user.to_numpy().copy(); rng.shuffle(mal)
    parts = {"train": list(mal[:C.MALICIOUS_SPLIT[0]]), "val": list(mal[C.MALICIOUS_SPLIT[0]:sum(C.MALICIOUS_SPLIT[:2])]),
             "test": list(mal[sum(C.MALICIOUS_SPLIT[:2]):])}
    nb = [C.SPLIT_USERS[i] - C.MALICIOUS_SPLIT[i] for i in range(3)]
    ben = u[u.mal == 0].copy(); ben["r"] = rng.random(len(ben))
    # stratified by role: assign users of each role proportionally to the three partitions
    ben = ben.sort_values(["role", "r"]); ben["rank_in_role"] = ben.groupby("role").cumcount() / ben.groupby("role")["user"].transform("count")
    frac = np.array(nb) / sum(nb); cut = np.cumsum(frac)
    ben["part"] = np.where(ben.rank_in_role < cut[0], "train", np.where(ben.rank_in_role < cut[1], "val", "test"))
    cnt = ben.part.value_counts().to_dict()                                   # fix rounding so totals are exactly 234/58/98
    for tgt, name in zip(nb, ("train", "val", "test")):
        while cnt.get(name, 0) > tgt:
            other = [n for n in ("train", "val", "test") if cnt.get(n, 0) < nb[("train", "val", "test").index(n)]][0]
            i = ben.index[ben.part == name][-1]; ben.loc[i, "part"] = other; cnt = ben.part.value_counts().to_dict()
    for name in ("train", "val", "test"): parts[name] += list(ben[ben.part == name].user)
    allu = [set(v) for v in parts.values()]
    assert not (allu[0] & allu[1] or allu[0] & allu[2] or allu[1] & allu[2]), "user overlap between partitions"
    for r in C.ROLES: raw[f"role_{r}"] = (raw["role"] == r).astype(float)
    sc = StandardScaler().fit(raw[raw.user.isin(parts["train"])][C.NUMERIC_FEATURES])
    raw[C.NUMERIC_FEATURES] = sc.transform(raw[C.NUMERIC_FEATURES])
    out = {k: raw[raw.user.isin(v)].reset_index(drop=True) for k, v in parts.items()}
    out["anon"] = anon; out["users"] = parts; out["seed"] = seed
    out["summary"] = {k: dict(users=len(parts[k]), records=len(out[k]), positives=int(out[k].is_malicious.sum()),
                              malicious_users=int(out[k].groupby("user").malicious_user.max().sum())) for k in ("train", "val", "test")}
    if save:
        with open(os.path.join(C.DATA_DIR, f"processed_seed_{seed}.pkl"), "wb") as f: pickle.dump(out, f)
    return out

def load_processed(seed: int) -> dict:
    with open(os.path.join(C.DATA_DIR, f"processed_seed_{seed}.pkl"), "rb") as f: return pickle.load(f)
