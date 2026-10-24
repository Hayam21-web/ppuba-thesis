"""Implementation-consistency tests of the DP mechanism in ppuba_lib/fl.py (Appendix D, Table D.3).

dp_accounting_check.py verifies the ACCOUNTING (noise multipliers and composition). This script tests the CODE that the
pipeline actually runs (ppuba_lib.fl.noise_plan and ppuba_lib.fl.train_fedavg) against the specification in Table 3.4
and Section 3.5. It instruments the random-number generator used inside train_fedavg and runs small controlled cases:

  T1  budget split: eps/T and delta/T per noisy step with T = 50 rounds x 3 local steps = 150, so T * eps_step = eps
  T2  noise multipliers: noise_plan() reproduces sigma of Table 4.4 and noise SD = sigma * C / B
  T3  noise is drawn at every local step of every client (150 draws per client), with SD sigma*C/B, on all d+1 coordinates
  T4  every local step uses a minibatch of exactly B = 256 records drawn without replacement from the client's own records
  T5  per-example clipping: replacing one record by an extreme record (x 1e6, with or without the class weight of 30)
      changes the averaged clipped gradient by at most 2C/B (replace-one) and adding/removing a record bounds C/B
  T6  the class weight is applied BEFORE clipping (a positive record's clipped gradient norm is <= C, not 30 C)
  T7  eps = inf disables noise (non-private reference) and finite eps never does

It also lists, as findings, the operations that the code performs OUTSIDE the DP mechanism (Section 6.4).
These are software tests of one implementation; they are not a formal verification or an external audit.

    python verification/dp_implementation_check.py    (writes verification/dp_implementation_check.json; exit 1 on failure)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import ppuba_lib.fl as fl

REPORTED = {0.5: 1037.4, 1.0: 543.0, 2.0: 283.3, 5.0: 119.4, 10.0: 61.9, 20.0: 32.1}
results = []


def check(test, desc, ok, detail=""):
    results.append(dict(test=test, description=desc, passed=bool(ok), detail=detail))


class RecordingRNG:
    """Wraps numpy Generator; records every normal() and choice() call made by train_fedavg."""
    def __init__(self, seed):
        self.g = np.random.Generator(np.random.PCG64(seed)); self.normal_calls = []; self.choice_calls = []
    def normal(self, loc, scale, size):
        self.normal_calls.append((loc, scale, tuple(int(v) for v in np.atleast_1d(size)))); return self.g.normal(loc, scale, size)
    def choice(self, a, k, replace=True):
        out = self.g.choice(a, k, replace=replace); self.choice_calls.append((len(a), int(k), replace, np.asarray(out).copy())); return out
    def __getattr__(self, n): return getattr(self.g, n)


_real_rng = np.random.default_rng
def with_recorder(fn):
    holder = {}
    def fake(seed=None):
        holder["rng"] = RecordingRNG(seed); return holder["rng"]
    fl.np.random.default_rng = fake
    try: out = fn()
    finally: fl.np.random.default_rng = _real_rng
    return out, holder["rng"]


T = C.ROUNDS * C.LOCAL_STEPS; Bsz, Cn = C.BATCH_SIZE, C.CLIP_C
# ---- T1 / T2
for eps, rep in REPORTED.items():
    p = fl.noise_plan(eps)
    check("T1", f"eps={eps}: T={p['n_compositions']}, T*eps_step={p['n_compositions']*p['eps_per_composition']:.6g}, T*delta_step="
          f"{p['n_compositions']*p['delta_per_composition']:.3g}",
          p["n_compositions"] == 150 and abs(p["n_compositions"] * p["eps_per_composition"] - eps) < 1e-9
          and abs(p["n_compositions"] * p["delta_per_composition"] - C.DELTA) < 1e-15)
    check("T2", f"eps={eps}: sigma={p['noise_multiplier']:.1f} (Table 4.4: {rep}); SD={p['noise_std_per_coordinate']:.4f}",
          abs(round(p["noise_multiplier"], 1) - rep) < 0.05 and abs(p["noise_std_per_coordinate"] - p["noise_multiplier"] * Cn / Bsz) < 1e-12)
    # the solved sigma must satisfy the analytic-Gaussian condition at (eps/T, delta/T)
    check("T2", f"eps={eps}: analytic-Gaussian delta(sigma) <= delta/T",
          fl._delta_gauss(p["eps_per_composition"], p["noise_multiplier"]) <= p["delta_per_composition"] * (1 + 1e-6))

# ---- synthetic clients of realistic size (3,600 records, 20 features; a few positives)
rng0 = np.random.default_rng(7); d = len(C.ALL_FEATURES)
def client(n=3600, n_pos=8):
    X = rng0.normal(size=(n, d)); y = np.zeros(n, int); y[rng0.choice(n, n_pos, replace=False)] = 1; return X, y
clients = [client() for _ in range(3)]

# ---- T3 / T4 (eps = 5, full 50 rounds x 3 steps)
(_, _, plan, _, _), rec = with_recorder(lambda: fl.train_fedavg(clients, d, 5.0, seed=42))
n_draws = len(rec.normal_calls); sds = {round(c[1], 12) for c in rec.normal_calls}; shapes = {c[2] for c in rec.normal_calls}
check("T3", f"noise draws = {n_draws} (expected {T} per client x {len(clients)} clients = {T*len(clients)})", n_draws == T * len(clients))
check("T3", f"noise SD used = {sorted(sds)} (expected {plan['noise_std_per_coordinate']:.6f})",
      len(sds) == 1 and abs(next(iter(sds)) - plan["noise_std_per_coordinate"]) < 1e-12)
check("T3", f"noise shape = {shapes} (expected d+1 = {d+1}: weights and bias)", shapes == {(d + 1,)})
per_step = [rec.choice_calls[i][1] + rec.choice_calls[i + 1][1] for i in range(0, len(rec.choice_calls), 2)]
check("T4", f"batch size per local step: min {min(per_step)}, max {max(per_step)} (expected {Bsz}); steps = {len(per_step)}",
      set(per_step) == {Bsz} and len(per_step) == T * len(clients))
check("T4", "minibatches drawn without replacement", all(c[2] is False for c in rec.choice_calls))

# ---- T7
(_, _, p_inf, _, _), rec_inf = with_recorder(lambda: fl.train_fedavg(clients, d, float("inf"), seed=42))
check("T7", f"eps=inf: {len(rec_inf.normal_calls)} noise draws, noise SD {p_inf['noise_std_per_coordinate']}",
      len(rec_inf.normal_calls) == 0 and p_inf["noise_std_per_coordinate"] == 0.0)
check("T7", "all finite eps in the grid produce positive noise",
      all(fl.noise_plan(e)["noise_std_per_coordinate"] > 0 for e in C.EPSILON_GRID if np.isfinite(e)))

# ---- T5 / T6: sensitivity of one noise-free local step on a client whose dataset equals one minibatch (B = 256 records)
def one_step(X, y):
    old = (C.ROUNDS, C.LOCAL_STEPS)
    C.LOCAL_STEPS = 1
    try: w, b, *_ = fl.train_fedavg([(X, y)], d, float("inf"), seed=1, rounds=1)
    finally: C.ROUNDS, C.LOCAL_STEPS = old
    return np.append(w, b) / (-C.LR)          # starting from w = 0 the update is -LR * mean clipped gradient (L2 term is 0 at w = 0)

npos = C.MIN_POS_PER_BATCH
X0 = rng0.normal(size=(Bsz, d)); y0 = np.zeros(Bsz, int); y0[:npos] = 1
g0 = one_step(X0, y0)
worst = 0.0
for idx, flip in ((0, True), (0, False), (Bsz - 1, False)):       # positive record (class weight 30, with/without label flip), negative record
    for scale in (1e3, 1e6):
        X1, y1 = X0.copy(), y0.copy(); X1[idx] = scale * rng0.normal(size=d)
        if flip: y1[idx] = 0
        diff = float(np.linalg.norm(one_step(X1, y1) - g0)); worst = max(worst, diff)
check("T5", f"replace-one: max ||delta mean clipped gradient|| = {worst:.6f} <= 2C/B = {2*Cn/Bsz:.6f}", worst <= 2 * Cn / Bsz + 1e-12)
# T6: a single extreme positive record alone: its clipped contribution must be <= C (class weight inside the clip)
Xs = np.zeros((Bsz, d)); ys = np.zeros(Bsz, int); Xs[0] = 1e6; ys[0] = 1
gz = one_step(np.zeros((Bsz, d)), np.zeros(Bsz, int))          # all-zero negatives: each per-example gradient is 0.5 * e_bias (norm 0.5 < C)
single = float(np.linalg.norm(one_step(Xs, ys) * Bsz - gz * (Bsz - 1)))   # clipped gradient of the single extreme positive record
check("T6", f"clipped gradient norm of an extreme positive record = {single:.4f} <= C = {Cn} (would be up to 30C = {30*Cn} "
      "if the class weight were applied after clipping)", abs(single - Cn) < 1e-6)

# ---- Findings: operations outside the DP mechanism (not failures of the tests; disclosed in Section 6.4)
findings = [
    "StandardScaler mean and SD are fitted centrally on all training records (ppuba_lib/data.py, preprocess) without noise.",
    "Pseudonymization, the identity-aware split, the role-based client assignment and the k-anonymity/l-diversity "
    "statistics are computed centrally on the pooled corpus before training.",
    "Minibatches are stratified by label (up to 16 positives per batch); the dependence of the batch on the data is "
    "covered by the replace-one sensitivity 2C/B (Table D.2, replace-one columns), not by the add/remove calibration C/B.",
    "FedAvg weights are proportional to the true client record counts (ns / ns.sum()), which are not privatized.",
    "The training-loss curve (Figure 5.4) is computed each round on all training records without noise; it is a "
    "diagnostic output and not part of the protected release.",
    "Decision thresholds are selected on the validation users, whose records are not protected by DP.",
    "Noise is drawn with NumPy's PCG64 generator in floating point; no cryptographically secure RNG or protection "
    "against floating-point attacks (e.g., Mironov, 2012) is implemented.",
    "The noise generator is seeded from the published experiment seed (seed * 7919 + int(100 eps)), so the noise is "
    "reproducible by anyone who knows the seed; this is required for reproducibility of the thesis but would remove "
    "the protection in a deployment, where the noise must come from a secret, non-reproducible source.",
    "Protection is record-level (one user-day); a user contributes about 150 records and has no user-level guarantee.",
]
# ---- Illustration: user-level budget (worst case: all ~150 records of one user in every minibatch -> sensitivity 150C/B)
user_level = {}
try:
    import dp_accounting as dpa
    from dp_accounting import rdp
    for eps in REPORTED:
        ev = dpa.SelfComposedDpEvent(dpa.GaussianDpEvent(fl.noise_plan(eps)["noise_multiplier"] / C.DAYS_PER_USER), T)
        acc = rdp.RdpAccountant(); acc.compose(ev); user_level[str(eps)] = float(round(acc.get_epsilon(C.DELTA), 1))
except ImportError:
    user_level = "dp-accounting not installed (pip install -r verification/requirements.txt)"
out = dict(results=results, user_level_epsilon_worst_case_rdp=user_level, n_passed=sum(r["passed"] for r in results), n_tests=len(results), outside_dp_mechanism=findings,
           scope="Software tests of ppuba_lib/fl.py against Table 3.4; not a formal verification or an external audit.")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_implementation_check.json"), "w"), indent=1)
for r in results: print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['test']}  {r['description']}")
print(f"\n{out['n_passed']}/{out['n_tests']} implementation tests passed.")
print("Illustrative user-level epsilon (RDP, delta = 1e-5, worst case 150 records of one user per minibatch):", user_level)
print("Operations outside the DP mechanism:")
for f in findings: print(" -", f)
sys.exit(0 if out["n_passed"] == out["n_tests"] else 1)
