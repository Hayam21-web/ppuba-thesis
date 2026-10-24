"""Empirical privacy audit of the DP mechanism in ppuba_lib/fl.py (Appendix D, Table D.6).

Formal privacy accounting gives an UPPER bound on epsilon. An empirical audit gives a LOWER bound: an attacker tries to
tell apart two neighbouring datasets from the mechanism's output, and the attack's true- and false-positive rates bound
epsilon from below (Jagielski et al., 2020; Nasr et al., 2021). If an audit ever finds a lower bound ABOVE the theoretical
epsilon, the implementation leaks more than claimed. A lower bound close to the theoretical value shows that the audit is
strong enough to detect such an error.

Set-up (white-box, worst case, one noisy local step of the real training code):
  * the pipeline function ppuba_lib.fl.train_fedavg is run with one client, one round and one local step, starting from
    w = 0, on a client dataset of exactly B = 256 records (16 positives), so the minibatch is the whole dataset;
  * dataset D contains a record whose clipped gradient points in direction -u; the neighbour D' replaces it by a canary
    whose clipped gradient points in direction +u (replace-one adjacency, the strongest case, sensitivity 2C/B);
  * the attacker observes the released update and uses its projection on u as test statistic;
  * the noise is calibrated by the pipeline's own noise_plan(); per-step budgets of 0.5, 1, 2 and 4 (T = 1) and the
    per-step budgets of the thesis configuration (nominal eps = 5 and 20 split over T = 150 steps) are audited.

The threshold is chosen on one half of the trials and the lower bound is computed on the other half with one-sided
97.5% Clopper-Pearson limits: eps_lb = ln((TPR_low - delta) / FPR_high).

    python verification/dp_empirical_audit.py          (writes verification/dp_empirical_audit.json; ~10 min)
"""
import functools, json, os, sys
import numpy as np
from scipy.stats import beta, norm
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import ppuba_lib.fl as fl

B, CLIP, d = C.BATCH_SIZE, C.CLIP_C, len(C.ALL_FEATURES)
fl.noise_plan = functools.lru_cache(maxsize=None)(fl.noise_plan)     # caching only; the mechanism itself is unchanged
C.LOCAL_STEPS = 1


def theory_eps(sigma, delta, sens):
    """Exact epsilon of the Gaussian mechanism (std sigma in units of C/B) for sensitivity sens (in units of C/B)."""
    s = sigma / sens
    if fl._delta_gauss(1e-9, s) <= delta: return 0.0
    return brentq(lambda e: fl._delta_gauss(e, s) - delta, 1e-9, 100)


def cp_low(k, n, a=0.025): return 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
def cp_high(k, n, a=0.025): return 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)


def eps_lower_bound(s0, s1, delta):
    """Pick the threshold that maximizes the Clopper-Pearson-bounded estimate on the first half of the trials,
    then report the bounded estimate on the untouched second half (both test directions)."""
    h0, h1 = len(s0) // 2, len(s1) // 2
    a0, b0, a1, b1 = s0[:h0], s0[h0:], s1[:h1], s1[h1:]

    def bounded(x0, x1, t):
        k0, k1 = int((x0 > t).sum()), int((x1 > t).sum())
        lo, hi = cp_low(k1, len(x1)), cp_high(k0, len(x0))
        return np.log((lo - delta) / hi) if lo - delta > 0 else -np.inf

    best = (-np.inf, None, 1)
    for sign in (1, -1):
        x0, x1 = sign * a0, sign * a1
        for t in np.quantile(np.concatenate([x0, x1]), np.linspace(0.5, 0.9999, 300)):
            e = bounded(x0, x1, t)
            if e > best[0]: best = (e, t, sign)
    _, t, sign = best
    if t is None: return 0.0
    return max(0.0, float(bounded(sign * b0, sign * b1, t)))


def run(eps_arg, delta_total, n_trials, label):
    C.DELTA = delta_total; fl.noise_plan.cache_clear()
    rng = np.random.default_rng(12345)
    X = rng.normal(size=(B, d)) * 0.5; y = np.zeros(B, int); y[:C.MIN_POS_PER_BATCH] = 1
    u = np.zeros(d); u[0] = 1.0                                # canary direction in feature space
    X0, X1 = X.copy(), X.copy(); X0[-1] = -1e6 * u; X1[-1] = 1e6 * u   # last record: negative label, extreme features
    plan = fl.noise_plan(eps_arg, rounds=1)
    stats = []
    for Xd in (X0, X1):
        s = np.empty(n_trials)
        for k in range(n_trials):
            w, b, *_ = fl.train_fedavg([(Xd, y)], d, eps_arg, seed=k + (0 if Xd is X0 else 10_000_000), rounds=1)
            s[k] = -w[0] / C.LR                                # projection of the averaged noisy gradient on u
        stats.append(s)
    np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_stats", f"{label.replace(' ', '_').replace('=', '')}.npz"), D=stats[0], D_prime=stats[1])
    sigma, dstep = plan["noise_multiplier"], plan["delta_per_composition"]
    e_add = plan["eps_per_composition"]; e_rep = theory_eps(sigma, dstep, 2.0)
    lb = eps_lower_bound(stats[0], stats[1], dstep)
    sep = (stats[1].mean() - stats[0].mean()) / (CLIP / B); sd = np.concatenate([stats[0] - stats[0].mean(), stats[1] - stats[1].mean()]).std() / (CLIP / B)
    out = dict(config=label, sigma=sigma, delta_step=dstep, trials_per_dataset=n_trials,
               observed_mean_shift_in_C_over_B=float(sep), observed_noise_sd_in_C_over_B=float(sd),
               theoretical_eps_add_remove=e_add, theoretical_eps_replace_one=e_rep, empirical_eps_lower_bound=lb,
               consistent=bool(lb <= e_rep + 1e-9))
    print(f"{label:>34}: sigma={sigma:9.2f}  shift={sep:5.3f} C/B  sd={sd:8.2f}  eps_lb={lb:6.3f}  "
          f"theory add/remove={e_add:6.3f} replace-one={e_rep:6.3f}  {'OK' if out['consistent'] else 'VIOLATION'}")
    return out


os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_stats'), exist_ok=True)
res = []
for e in (0.5, 1.0, 2.0, 4.0):
    res.append(run(e, 1e-5, 200_000, f"one step calibrated to eps={e:g}"))
for nominal in (20.0, 5.0):
    res.append(run(nominal / 150, 1e-5 / 150, 500_000, f"thesis step (nominal eps={nominal:g})"))
out = dict(results=res, all_consistent=all(r["consistent"] for r in res),
           note="Empirical lower bounds; consistency means no implementation leak was detected. Not a formal proof.")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_empirical_audit.json"), "w"), indent=1)
print("ALL CONSISTENT" if out["all_consistent"] else "VIOLATION FOUND")
sys.exit(0 if out["all_consistent"] else 1)
