"""Cross-check of the DP-SGD step in ppuba_lib/fl.py against Opacus (Appendix D, Table D.5).

The pipeline implements per-example clipping, averaging and Gaussian noise directly in NumPy. This script recomputes the
same local step with Opacus (Meta's open-source DP library: GradSampleModule for per-sample gradients and DPOptimizer
for clipping and noise) on identical minibatches and compares:

  C1  the clipped, averaged gradient (noise switched off) for 20 random minibatches drawn by the pipeline's own sampler,
      including extreme records and the class weight of 30, at the clipping norm C = 3.0 and B = 256;
  C2  the standard deviation of the Gaussian noise that Opacus adds for noise multiplier sigma (Table 4.4) compared
      with the pipeline's noise SD sigma * C / B, estimated from 2,000 noisy steps per epsilon.

Needs a separate environment:  pip install -r verification/requirements_opacus.txt   (torch 2.5.1, opacus 1.5.2)
    python verification/dp_opacus_crosscheck.py      (writes verification/dp_opacus_crosscheck.json; exit 1 on mismatch)
"""
import json, os, sys
import numpy as np
import torch
from opacus import GradSampleModule
from opacus.optimizers import DPOptimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import ppuba_lib.fl as fl

torch.set_default_dtype(torch.float64)
d = len(C.ALL_FEATURES); B, CLIP = C.BATCH_SIZE, C.CLIP_C
rng = np.random.default_rng(2026)


def pipeline_clipped_mean(xb, yb, w, b):
    """Exactly the arithmetic of ppuba_lib.fl.train_fedavg for one local step, without noise."""
    p = fl.sigmoid(xb @ w + b); cw = np.where(yb == 1, C.MINORITY_WEIGHT, 1.0)
    g = ((p - yb) * cw)[:, None] * np.hstack([xb, np.ones((len(xb), 1))])
    g = g / np.maximum(1.0, np.linalg.norm(g, axis=1, keepdims=True) / CLIP)
    return g.mean(0)


def opacus_step(xb, yb, w, b, sigma, seed):
    model = torch.nn.Linear(d, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(w).view(1, -1)); model.bias.copy_(torch.tensor([b]))
    gsm = GradSampleModule(model, loss_reduction="sum")
    opt = DPOptimizer(torch.optim.SGD(gsm.parameters(), lr=1.0), noise_multiplier=sigma, max_grad_norm=CLIP,
                      expected_batch_size=B, loss_reduction="sum")
    opt.generator = torch.Generator().manual_seed(seed)
    X = torch.tensor(xb); Y = torch.tensor(yb, dtype=torch.float64)
    cw = torch.where(Y == 1, torch.tensor(C.MINORITY_WEIGHT), torch.tensor(1.0))
    loss = (cw * torch.nn.functional.binary_cross_entropy_with_logits(gsm(X).view(-1), Y, reduction="none")).sum()
    opt.zero_grad(); loss.backward(); opt.pre_step()          # clip per sample, sum, add N(0, (sigma*C)^2) noise
    g = np.concatenate([model.weight.grad.view(-1).numpy(), model.bias.grad.view(-1).numpy()])
    return g / B                                               # with loss_reduction="sum" Opacus leaves the sum; average over B


results = {}
# ---- C1: deterministic part on minibatches built by the pipeline's sampler (16 positives + 240 negatives)
n = 3600; X = rng.normal(size=(n, d)); y = np.zeros(n, int); y[rng.choice(n, 40, replace=False)] = 1
X[rng.choice(n, 30, replace=False)] *= 1e3                          # extreme records that must be clipped
pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); maxdiff = 0.0; rel = 0.0
for t in range(20):
    idx = np.concatenate([rng.choice(pos, C.MIN_POS_PER_BATCH, replace=False), rng.choice(neg, B - C.MIN_POS_PER_BATCH, replace=False)])
    w = rng.normal(scale=0.5, size=d); b = float(rng.normal())
    a = pipeline_clipped_mean(X[idx], y[idx], w, b); o = opacus_step(X[idx], y[idx], w, b, 0.0, t)
    maxdiff = max(maxdiff, float(np.max(np.abs(a - o)))); rel = max(rel, float(np.linalg.norm(a - o) / np.linalg.norm(a)))
results["C1_clipped_mean"] = dict(minibatches=20, max_abs_difference=maxdiff, max_relative_difference=rel,
                                  passed=bool(rel < 1e-5),
                                  note="residual differences come from Opacus' clip factor C/(||g||+1e-6) versus C/||g||")

# ---- C2: noise standard deviation per epsilon
idx = np.concatenate([pos[:C.MIN_POS_PER_BATCH], neg[:B - C.MIN_POS_PER_BATCH]]); w = np.zeros(d); b = 0.0
base = opacus_step(X[idx], y[idx], w, b, 0.0, 0)
noise = {}
for eps in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    plan = fl.noise_plan(eps); sigma = plan["noise_multiplier"]
    draws = np.stack([opacus_step(X[idx], y[idx], w, b, sigma, 10_000 + k) - base for k in range(2000)])
    sd_opacus = float(draws.std(ddof=1)); sd_pipe = plan["noise_std_per_coordinate"]
    # 99.9% interval for an SD estimated from 2000 x 21 draws
    tol = 3.3 * sd_pipe / np.sqrt(2 * draws.size)
    noise[str(eps)] = dict(sigma=sigma, noise_sd_pipeline=sd_pipe, noise_sd_opacus=sd_opacus,
                           relative_difference=(sd_opacus - sd_pipe) / sd_pipe, passed=bool(abs(sd_opacus - sd_pipe) < tol))
results["C2_noise_sd"] = noise
results["versions"] = dict(torch=torch.__version__, opacus=__import__("opacus").__version__, numpy=np.__version__)
ok = results["C1_clipped_mean"]["passed"] and all(v["passed"] for v in noise.values())
results["all_passed"] = bool(ok)
json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dp_opacus_crosscheck.json"), "w"), indent=1)
print(f"C1 clipped mean: max abs diff {maxdiff:.2e}, max rel diff {rel:.2e} -> {'PASS' if results['C1_clipped_mean']['passed'] else 'FAIL'}")
for e, v in noise.items():
    print(f"C2 eps={e:>4}: sigma={v['sigma']:.1f} SD pipeline {v['noise_sd_pipeline']:.4f}  SD Opacus {v['noise_sd_opacus']:.4f}  "
          f"rel diff {100*v['relative_difference']:+.2f}% -> {'PASS' if v['passed'] else 'FAIL'}")
print("ALL PASSED" if ok else "MISMATCH FOUND")
sys.exit(0 if ok else 1)
