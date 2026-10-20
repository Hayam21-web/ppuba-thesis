"""Federated Averaging + DP-SGD-style logistic regression implemented directly in NumPy (Tables 3.5b, 4.4, 4.5)."""
import time, numpy as np
from scipy.stats import norm
import config as C
from ppuba_lib.metrics import best_threshold, evaluate

def _delta_gauss(eps, sigma):  # Balle & Wang (2018), analytic Gaussian mechanism, L2 sensitivity 1
    return norm.cdf(1 / (2 * sigma) - eps * sigma) - np.exp(eps) * norm.cdf(-1 / (2 * sigma) - eps * sigma)

def analytic_sigma(eps, delta):
    lo, hi = 1e-4, 1e7
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if _delta_gauss(eps, mid) > delta: lo = mid
        else: hi = mid
    return hi

def noise_plan(eps, composition=None, steps=None, rounds=None):
    comp = composition or C.COMPOSITION; R = rounds or C.ROUNDS; S = steps or C.LOCAL_STEPS
    T = R * S if comp == "steps" else R
    if np.isinf(eps): return dict(epsilon=eps, delta=0.0, composition=comp, n_compositions=T, eps_per_composition=float("inf"),
                                  delta_per_composition=0.0, noise_multiplier=0.0, noise_std_per_coordinate=0.0)
    e_i, d_i = eps / T, C.DELTA / T; sig = analytic_sigma(e_i, d_i)
    return dict(epsilon=eps, delta=C.DELTA, composition=comp, n_compositions=T, eps_per_composition=e_i, delta_per_composition=d_i,
                noise_multiplier=float(sig), noise_std_per_coordinate=float(sig * C.CLIP_C / C.BATCH_SIZE))

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def make_clients(train, feats, n_clients, seed):
    """Non-IID by role: each role's training users are split into SHARDS_PER_ROLE shards, shards dealt round-robin to clients."""
    rng = np.random.default_rng(seed + 20_000); shards = []
    for role in C.ROLES:
        us = rng.permutation(np.array(sorted(train[train.role == role].user.unique()), dtype=object))
        shards += [s for s in np.array_split(us, C.SHARDS_PER_ROLE) if len(s)]
    groups = [[] for _ in range(n_clients)]
    for i, s in enumerate(shards): groups[i % n_clients] += list(s)
    return [(train[train.user.isin(g)][feats].values, train[train.user.isin(g)].is_malicious.values) for g in groups if g]

def train_fedavg(clients, d, eps, seed, Xtr=None, ytr=None, composition=None, rounds=None):
    R = rounds or C.ROUNDS; plan = noise_plan(eps, composition, rounds=R); rng = np.random.default_rng(seed * 7919 + (0 if np.isinf(eps) else int(eps * 100)))
    sd = plan["noise_std_per_coordinate"]; w = np.zeros(d); b = 0.0; losses = []; t0 = time.time()
    ns = np.array([len(y) for _, y in clients], float); wts = ns / ns.sum()
    for r in range(R):
        ws, bs = [], []
        for X, y in clients:
            wl, bl = w.copy(), b; pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
            for _ in range(C.LOCAL_STEPS):
                k = min(len(pos), C.MIN_POS_PER_BATCH); k_neg = min(len(neg), C.BATCH_SIZE - k)
                idx = np.concatenate([rng.choice(pos, k, replace=False) if k else np.array([], int), rng.choice(neg, k_neg, replace=False)])
                xb, yb = X[idx], y[idx]; p = sigmoid(xb @ wl + bl); cw = np.where(yb == 1, C.MINORITY_WEIGHT, 1.0)
                g = ((p - yb) * cw)[:, None] * np.hstack([xb, np.ones((len(xb), 1))])            # per-example gradient (w, b)
                g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / C.CLIP_C)       # clip to L2 <= C
                gm = g.mean(0)
                if sd > 0: gm = gm + rng.normal(0, sd, gm.shape)                                    # Gaussian noise on the averaged gradient
                gm[:-1] += C.L2 * wl; wl = wl - C.LR * gm[:-1]; bl = bl - C.LR * gm[-1]
            ws.append(wl); bs.append(bl)
        w = sum(a * x for a, x in zip(wts, ws)); b = float(sum(a * x for a, x in zip(wts, bs)))
        if Xtr is not None:
            p = sigmoid(Xtr @ w + b); losses.append(float(-np.mean(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12))))
    return w, b, plan, losses, time.time() - t0

def run_sweep(sp, feats, seed, eps_grid=None, n_clients=None, composition=None):
    tr, va, te = sp["train"], sp["val"], sp["test"]; clients = make_clients(tr, feats, n_clients or C.N_CLIENTS, seed)
    Xtr, ytr = tr[feats].values, tr.is_malicious.values; Xva, yva, Xte, yte = va[feats].values, va.is_malicious.values, te[feats].values, te.is_malicious.values
    res, models, scores, det = {}, {}, {}, {}
    for eps in (eps_grid or C.EPSILON_GRID):
        key = "inf" if np.isinf(eps) else str(eps)
        w, b, plan, losses, dt = train_fedavg(clients, len(feats), eps, seed, Xtr, ytr, composition)
        sva, ste = sigmoid(Xva @ w + b), sigmoid(Xte @ w + b); thr = best_threshold(yva, sva)
        m = evaluate(yte, ste, thr); m["train_seconds"] = dt; m["seconds_per_round"] = dt / C.ROUNDS
        m["final_train_loss"] = losses[-1]; m["n_clients"] = len(clients)
        res[key] = m; models[key] = (w, b); scores[key] = dict(val=sva, test=ste); det[key] = dict(plan=plan, loss_curve=losses)
    return res, models, scores, det, [(len(y), int(y.sum())) for _, y in clients]
