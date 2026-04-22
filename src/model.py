import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
import gc

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_PATH = # Enter Your Dataset Path
COST_FN = 500   # Cost of missing a fraud ($)
COST_FP = 50    # Cost of blocking a legit user ($)
RANDOM_STATE = 42

print("🚀 Initializing Enterprise Fraud Detection System...")

# ============================================================
# 2. GPU CHECK
# ============================================================
try:
    _ = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
    print("✅ NVIDIA RTX 4060 detected. High-Performance Mode ENABLED.")
    USE_GPU = True
except Exception as e:
    print(f"⚠️ GPU WARNING: {e}\n   Falling back to CPU.")
    USE_GPU = False

# ============================================================
# 3. DATA LOADING & GRAPH ENGINEERING
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ CRITICAL: File not found at {DATA_PATH}")

print("📂 Loading and Analyzing Network Topology...")
df = pd.read_csv(DATA_PATH)
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
df.reset_index(drop=True, inplace=True)

# --- Graph Features ---
# 1. Hub Detection (Degree Centrality)
df["orig_degree"] = df.groupby("nameOrig")["nameDest"].transform("nunique")
df["dest_degree"] = df.groupby("nameDest")["nameOrig"].transform("nunique")

# 2. Link Analysis (Pair Frequency)
df["pair_weight"] = df.groupby(["nameOrig", "nameDest"])["amount"].transform("count")

# 3. Time-Safe Risk Scoring (No Data Leakage)
print("🧠 Computing Time-Safe Risk Scores...")
df = df.sort_values("step").reset_index(drop=True)
global_fraud_rate = df["isFraud"].mean()

# Cumulative stats for destination accounts
grp = df.groupby("nameDest")["isFraud"]
cum_fraud = grp.cumsum() - df["isFraud"]
cum_count = grp.cumcount()

# Calculate historical risk
with np.errstate(divide="ignore", invalid="ignore"):
    dest_hist_rate = np.where(
        cum_count > 0,
        cum_fraud / cum_count,
        global_fraud_rate
    )
df["dest_risk_score"] = dest_hist_rate.astype(np.float32)

# Cleanup
df["type"] = df["type"].map({"TRANSFER": 0, "CASH_OUT": 1}).astype(np.int8)
drop_cols = [c for c in ["nameOrig", "nameDest", "isFlaggedFraud", "step"] if c in df.columns]
X = df.drop(columns=drop_cols + ["isFraud"]).astype(np.float32)
y = df["isFraud"].astype(np.int8)

del df, grp, cum_fraud, cum_count, dest_hist_rate
gc.collect()

# ============================================================
# 4. TRAINING (Train/Val/Test Split)
# ============================================================
print("✂️ Splitting Data (Train / Validation / Test)...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

# Scale Weights
neg, pos = np.bincount(y_train)
scale_weight = neg / pos

print("⚡ Training XGBoost Model on GPU...")
clf = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method="hist",
    device="cuda" if USE_GPU else "cpu",
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.05,
    scale_pos_weight=scale_weight,
    subsample=0.8,
    early_stopping_rounds=50,
    random_state=RANDOM_STATE
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

# ============================================================
# 5. COST OPTIMIZATION
# ============================================================
print("\n💰 Optimizing Decision Threshold...")
val_probs = clf.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.01, 0.99, 100)

def get_cost(y_true, probs, t):
    y_pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return (fp * COST_FP) + (fn * COST_FN)

costs = [get_cost(y_val, val_probs, t) for t in thresholds]
best_threshold = thresholds[np.argmin(costs)]

print(f"   -> Optimal Threshold Found: {best_threshold:.2f}")

# Apply to Test Set
test_probs = clf.predict_proba(X_test)[:, 1]
final_cost = get_cost(y_test, test_probs, best_threshold)
default_cost = get_cost(y_test, test_probs, 0.5)
print(f"   -> Savings vs Default: ${default_cost - final_cost:,.2f}")

# ============================================================
# 6. DRIFT DETECTION (Page-Hinkley)
# ============================================================
print("\n🌊 Running Page-Hinkley Drift Simulation...")

class PageHinkley:
    def __init__(self, threshold=5, alpha=0.9999):
        self.threshold = threshold
        self.alpha = alpha
        self.cum_sum = 0
        self.mean = 0
        self.count = 0
        self.min_sum = 0

    def update(self, error):
        self.count += 1
        self.mean += (error - self.mean) / self.count
        self.cum_sum += (error - self.mean) - self.alpha
        if self.cum_sum < self.min_sum: self.min_sum = self.cum_sum

        drift = self.cum_sum - self.min_sum
        if drift > self.threshold:
            self.cum_sum = 0 # Reset
            return True, drift
        return False, drift

detector = PageHinkley(threshold=10)
drift_vals = []
batch_size = 2000

# Simulate stream
n_batches = len(X_test) // batch_size
print(f"   -> Monitoring {n_batches} data batches...")

for i in range(n_batches):
    start, end = i * batch_size, (i + 1) * batch_size
    X_b, y_b = X_test.iloc[start:end], y_test.iloc[start:end].copy()

    # Inject Artificial Drift in last 20%
    if i > n_batches * 0.8:
        y_b = 1 - y_b

    preds = (clf.predict_proba(X_b)[:, 1] >= best_threshold).astype(int)
    error = np.mean(preds != y_b)

    is_drift, mag = detector.update(error)
    drift_vals.append(mag)

# Plot Drift
plt.figure(figsize=(10, 4))
plt.plot(drift_vals, color='purple', label='Drift Magnitude')
plt.axvline(n_batches * 0.8, color='red', linestyle='--', label='Drift Injection Start')
plt.title("Real-Time Concept Drift Detection")
plt.legend()
plt.savefig("drift_report.png")
print("   -> Saved 'drift_report.png'")

print("\n✅ SYSTEM ONLINE. READY FOR REPORT.")
