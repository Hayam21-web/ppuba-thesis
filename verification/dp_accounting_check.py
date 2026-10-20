"""Independent check of the differential-privacy accounting reported in Table 3.4 / Table 4.4.

Independent of the thesis pipeline: it re-derives the noise multipliers of the analytic Gaussian
mechanism (Balle & Wang, 2018, Theorem 8) for the per-step budget (eps/T, delta/T) and evaluates the
T-fold composition of the same Gaussian mechanism with two accountants of Google's open-source
dp-accounting library (Renyi DP and privacy-loss distribution).

    pip install dp-accounting scipy numpy
    python verification/dp_accounting_check.py      (reproduces Table D.2; writes verification/dpcheck.json)

The pipeline code it checks is ppuba_lib/fl.py (noise_plan, train_fedavg): the class weight multiplies the per-example
gradient BEFORE clipping, so the sensitivity of the averaged clipped gradient is C/B (2C/B under replace-one adjacency).
"""
import json
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import dp_accounting as dpa
from dp_accounting import rdp, pld

T, DELTA, C, B = 150, 1e-5, 3.0, 256          # 50 rounds x 3 local steps; clipping norm; minibatch size
REPORTED = {0.5: 1037.4, 1: 543.0, 2: 283.3, 5: 119.4, 10: 61.9, 20: 32.1}   # Table 4.4


def gaussian_delta(sigma, eps):
    """Exact privacy profile of the Gaussian mechanism with sensitivity 1 (Balle & Wang, 2018, Thm 8)."""
    return norm.cdf(1 / (2 * sigma) - eps * sigma) - np.exp(eps) * norm.cdf(-1 / (2 * sigma) - eps * sigma)


def analytic_sigma(eps, delta):
    return brentq(lambda s: gaussian_delta(s, eps) - delta, 1e-3, 1e7)


def compose(noise_multiplier):
    event = dpa.SelfComposedDpEvent(dpa.GaussianDpEvent(noise_multiplier), T)
    r = rdp.RdpAccountant(); r.compose(event)
    p = pld.PLDAccountant(); p.compose(event)
    return r.get_epsilon(DELTA), p.get_epsilon(DELTA)


rows = []
for eps, rep in REPORTED.items():
    sigma = analytic_sigma(eps / T, DELTA / T)
    rdp1, pld1 = compose(sigma)        # add/remove-one adjacency: sensitivity C/B
    rdp2, pld2 = compose(sigma / 2)    # replace-one adjacency: sensitivity 2C/B
    rows.append(dict(eps=f"{eps:g}", rep=f"{rep:,.1f}", sigma=sigma, sd=sigma * C / B,
                     rdp=rdp1, pld=pld1, rdp2=rdp2, pld2=pld2))

print(f"{'eps':>5} {'sigma rep':>10} {'sigma new':>10} {'noise SD':>9} {'RDP':>7} {'PLD':>7} {'RDP(r1)':>8} {'PLD(r1)':>8}")
for r in rows:
    print(f"{r['eps']:>5} {r['rep']:>10} {r['sigma']:>10,.1f} {r['sd']:>9.2f} {r['rdp']:>7.3f} {r['pld']:>7.3f} "
          f"{r['rdp2']:>8.3f} {r['pld2']:>8.3f}")
    assert abs(r['sigma'] - float(r['rep'].replace(',', ''))) < 0.1, "noise multiplier mismatch"
    assert max(r['rdp'], r['pld'], r['rdp2'], r['pld2']) < float(r['eps']), "accountant exceeds nominal epsilon"
print("All noise multipliers reproduced; all accountant values below the nominal budgets.")
import os
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpcheck.json"), "w"), indent=1)   # values of Table D.2
