"""
PPUBA - single shared configuration (Table 4.5, Table B.2, Table 3.5b of the thesis).
Every value used by the pipeline lives here and is dumped to outputs/run_config.json for audit (NFR3/NFR4).
Generator design parameters were fixed a priori from the thesis description (Section 3.4 / Appendix B) and are NOT
tuned to any result; script 08 reports how sensitive the conclusions are to the attack-intensity choice.
"""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data"); OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures"); TAB_DIR = os.path.join(OUT_DIR, "tables"); SEED_DIR = os.path.join(OUT_DIR, "per_seed")
for _d in (DATA_DIR, OUT_DIR, FIG_DIR, TAB_DIR, SEED_DIR): os.makedirs(_d, exist_ok=True)

CORPUS_VERSION = "ppuba-synth-v1"; FEATURE_SCHEMA_VERSION = "f20-v1"
SEEDS = [42, 43, 44, 45, 46]; REP_SEED = 43

# ---------------- Corpus (Table 3.4 / B.2) ----------------
N_USERS = 400; DAYS_PER_USER = 150
ROLES = ["Admin", "Developer", "HR", "Finance", "Sales", "Engineering", "IT_Admin", "Executive"]
N_MALICIOUS = 10; MALICIOUS_SPLIT = (6, 2, 2)            # train / val / test
SPLIT_USERS = (240, 60, 100)
N_DEPARTMENTS = 12                                        # dept risk is a department attribute, independent of role and of labels
DEPT_RISK_RANGE = (0.1, 1.0); DEPT_RISK_DAILY_JITTER = 0.03
USER_HETEROGENEITY_SIGMA = 0.25                           # lognormal per-user, per-feature multiplier
WEEKEND_FACTOR = 0.15                                     # activity multiplier on days 5,6 of each week
SESSION_SD_MIN = 60.0
INTENSITY_SCALE = 1.0                                     # global multiplier on archetype intensity; overwritten by calibrated_intensity.json (script 00)
import json as _json
_cal = os.path.join(BASE_DIR, "calibrated_intensity.json")
if os.path.exists(_cal): INTENSITY_SCALE = _json.load(open(_cal))["chosen_intensity_scale"]

NUMERIC_FEATURES = ["failed_logins", "login_count", "afterhours_logins", "session_duration_min", "file_access_count",
                    "sensitive_file_access", "email_sent_count", "email_external_count", "web_upload_count",
                    "usb_connect_count", "print_count", "dept_risk_score"]
ROLE_FEATURES = [f"role_{r}" for r in ROLES]
ALL_FEATURES = NUMERIC_FEATURES + ROLE_FEATURES           # 12 + 8 = 20 inputs
COUNT_FEATURES = [f for f in NUMERIC_FEATURES if f not in ("session_duration_min", "dept_risk_score")]

# Role baseline daily means (Poisson means; session_duration in minutes)
ROLE_BASE = {
 #            failed login afterh  sess  file  sens  mail  mailx webup usb   print
 "Admin":       [0.6, 3.0, 0.3, 380, 40, 2.0, 12, 1.0, 1.0, 0.15, 2.0],
 "Developer":   [0.4, 3.2, 0.5, 420, 60, 1.0, 10, 0.6, 2.0, 0.20, 1.0],
 "HR":          [0.3, 2.8, 0.1, 360, 25, 3.0, 25, 2.0, 0.8, 0.05, 6.0],
 "Finance":     [0.3, 2.8, 0.2, 370, 30, 3.5, 22, 2.5, 0.8, 0.05, 5.0],
 "Sales":       [0.4, 3.0, 0.3, 350, 18, 0.5, 40, 6.0, 1.2, 0.10, 2.0],
 "Engineering": [0.4, 3.1, 0.4, 410, 55, 1.2, 9, 0.5, 1.8, 0.25, 1.0],
 "IT_Admin":    [0.8, 3.5, 0.8, 430, 45, 2.5, 8, 0.4, 1.5, 0.30, 1.0],
 "Executive":   [0.2, 2.5, 0.2, 330, 15, 1.5, 35, 4.0, 0.6, 0.05, 3.0]}
BASE_ORDER = ["failed_logins", "login_count", "afterhours_logins", "session_duration_min", "file_access_count",
              "sensitive_file_access", "email_sent_count", "email_external_count", "web_upload_count",
              "usb_connect_count", "print_count"]

# 15 scenario archetypes. Design principle (fixed a priori, not tuned to detector results): inside an attack window the mean of each
# affected feature is shifted by z * s benign standard deviations (benign SD of that feature for the user's role, including Poisson noise and
# per-user heterogeneity). Primary feature z = 4, secondary features z = 2.5, with intensity s ~ U(intensity range) * INTENSITY_SCALE.
SCENARIOS = {
 "credential_theft":         dict(window=(8, 14),  intensity=(0.75, 1.25), fx={"failed_logins": 4, "afterhours_logins": 2.5}),
 "brute_force_then_access":  dict(window=(8, 12),  intensity=(0.75, 1.25), fx={"failed_logins": 4, "sensitive_file_access": 2.5}),
 "data_exfil_usb":           dict(window=(10, 18), intensity=(0.75, 1.25), fx={"usb_connect_count": 4, "file_access_count": 2.5}),
 "data_exfil_email":         dict(window=(10, 18), intensity=(0.75, 1.25), fx={"email_external_count": 4, "email_sent_count": 2.5}),
 "cloud_upload":             dict(window=(10, 16), intensity=(0.75, 1.25), fx={"web_upload_count": 4, "file_access_count": 2.5}),
 "privilege_escalation":     dict(window=(10, 16), intensity=(0.75, 1.25), fx={"sensitive_file_access": 4, "failed_logins": 2.5}),
 "post_termination_access":  dict(window=(12, 20), intensity=(0.75, 1.25), fx={"afterhours_logins": 4, "login_count": 2.5}),
 "sabotage":                 dict(window=(8, 12),  intensity=(0.75, 1.25), fx={"file_access_count": 4, "sensitive_file_access": 2.5}),
 "print_exfiltration":       dict(window=(12, 20), intensity=(0.75, 1.25), fx={"print_count": 4, "sensitive_file_access": 2.5}),
 "mass_download":            dict(window=(8, 12),  intensity=(0.75, 1.25), fx={"file_access_count": 4, "web_upload_count": 2.5}),
 "slow_exfil_mixed":         dict(window=(14, 20), intensity=(0.75, 1.25), fx={"web_upload_count": 4, "email_external_count": 2.5, "usb_connect_count": 2.5}),
 "night_access_sensitive":   dict(window=(10, 16), intensity=(0.75, 1.25), fx={"afterhours_logins": 4, "sensitive_file_access": 2.5}),
 "shared_credential_abuse":  dict(window=(10, 16), intensity=(0.75, 1.25), fx={"failed_logins": 4, "login_count": 2.5, "afterhours_logins": 2.5}),
 "insider_recon":            dict(window=(12, 20), intensity=(0.75, 1.25), fx={"file_access_count": 4, "sensitive_file_access": 2.5, "session_duration_min": 2.5}),
 "external_leak_burst":      dict(window=(8, 14),  intensity=(0.75, 1.25), fx={"email_external_count": 4, "web_upload_count": 2.5, "usb_connect_count": 2.5}),
}
RNG_TYPE = "numpy.random.default_rng (PCG64), seeded with the experiment seed; sub-streams via SeedSequence(seed).spawn"
HMAC_KEY = b"NBU_PPUBA_THESIS_2026_KEYED_HASH"          # keyed-hash pseudonymization (replace for any real use)

# ---------------- Detectors (Table 4.5) ----------------
IF_PARAMS = dict(n_estimators=300, contamination=0.01, max_samples="auto")
RF_PARAMS = dict(n_estimators=400, max_depth=12, class_weight="balanced_subsample", n_jobs=-1)
AE_PARAMS = dict(hidden_layer_sizes=(12, 4, 12), activation="relu", solver="adam", learning_rate_init=0.005, max_iter=200)

# ---------------- Federated + DP (Table 4.4 / 4.5 / 3.5b) ----------------
N_CLIENTS = 10; ROUNDS = 50; LOCAL_STEPS = 3; LR = 0.05; L2 = 0.001
MINORITY_WEIGHT = 30.0; CLIP_C = 3.0; BATCH_SIZE = 256; MIN_POS_PER_BATCH = 16
EPSILON_GRID = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
DELTA = 1e-5; REPORT_EPS = 5.0
COMPOSITION = "steps"     # 'steps': basic composition over ROUNDS*LOCAL_STEPS noisy steps (Table 3.5b); 'rounds' = alt. sensitivity run
SHARDS_PER_ROLE = 3       # non-IID by role: each role's training users are split in 3 shards, distributed round-robin
UTILITY_RETENTION_RULE = 0.60
SCALABILITY_CLIENTS = [5, 10, 20]

# ---------------- Audit ----------------
MIA_SAMPLES = 2000; SHAP_NEG_SAMPLE = 300; LOW_FPR_LEVELS = [0.001, 0.01]
# Governance rubric (Section 4.4.4): criterion weights; the per-criterion scores are researcher-assigned.
RUBRIC_WEIGHTS = {"implementation": 0.30, "evidence": 0.25, "auditability": 0.20, "coverage": 0.15, "effectiveness": 0.10}
GOVERNANCE_TOTALS = {   # researcher-assigned control scores reported in Table 5.3c (used if governance_rubric_input.csv is empty)
 "Fixed random seeds and versioned pipeline": 95, "Identity-aware, leakage-safe split": 92,
 "Differential privacy (DP-SGD-style)": 91, "PII masking / pseudonymous user IDs": 88,
 "Federated learning (FedAvg)": 83, "SHAP explainability audit trail": 81,
 "Membership-inference / re-identification audit": 79, "k-anonymity / l-diversity quantification": 74}
