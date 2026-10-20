"""Script 02: pseudonymization, k-anonymity/l-diversity, identity-aware stratified split, scaling."""
import argparse, json, os
import config as C
from ppuba_lib.data import preprocess
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS); a = ap.parse_args()
for s in a.seeds:
    o = preprocess(s); json.dump({"summary": o["summary"], "anonymity": o["anon"]}, open(os.path.join(C.SEED_DIR, f"split_seed_{s}.json"), "w"), indent=2)
    print(f"[02] seed {s}:", o["summary"], "k(records)=", o["anon"]["k_anonymity_records"], "l=", o["anon"]["l_diversity"])
