"""Script 01: generate the seeded synthetic CERT-v6.2-schema corpus (one per seed)."""
import argparse, os
import config as C
from ppuba_lib.data import generate_corpus
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="*", default=C.SEEDS); a = ap.parse_args()
for s in a.seeds:
    df = generate_corpus(s); df.to_csv(os.path.join(C.DATA_DIR, f"raw_corpus_seed_{s}.csv"), index=False)
    print(f"[01] seed {s}: {len(df)} records, {df.malicious_user.sum()//C.DAYS_PER_USER} malicious users, {int(df.is_malicious.sum())} positive user-days")
