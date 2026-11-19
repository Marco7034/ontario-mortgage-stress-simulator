"""
ECON 354: Interpretable Machine Learning for Baseline Borrower Default Risk (PD_base)
& Ontario Macroeconomic Sensitivity
Phase 2: Linear Benchmark, RBF-SVM Tuning, Platt Calibration & Macro Stress Engine

Authors: Alex Babiak, Marco Puiatti
Scope: Ontario, Canada

Pipeline Structure (Cells 5-8)
------------------------------
1. Cell 5 (Partner's Track — Linear Econometric Benchmark & Naive Floor):
   - Loads the 6 verified Phase 1 microdata Parquet splits from `data/`.
   - Evaluates the trivial Naive Base-Rate predictor (constant training default rate
     0.21814, verifying ROC-AUC = 0.5000 and Brier Score ~= 0.1706).
   - Tunes L2-regularized Logistic Regression across C in [0.01, 0.1, 1.0, 10.0] on X_val,
     evaluates on X_test (N = 4,888), and outputs the 20-feature Economic Coefficient &
     Odds-Ratio table.
2. Cell 6 (Primary Track — Early-Stage SVM Progression, Tuning & Platt Calibration):
   - Contrasts Linear SVM (C=1.0) vs. Default RBF-SVM (C=1.0, gamma='scale') on X_val.
   - Runs a 16-point Validation grid search across C in [0.1, 1.0, 5.0, 10.0] and
     gamma in ['scale', 0.01, 0.05, 0.1] to quantify Bias-Variance trade-offs.
   - Wraps the optimal RBF-SVM in 3-fold Platt sigmoid calibration
     (`CalibratedClassifierCV(method="sigmoid", cv=3)`), persists `data/pd_test.npy`,
     and outputs the unified Out-of-Sample Model Comparison Table on X_test.
3. Cell 7 (Ontario Macroeconomic Stress-Testing Engine & Trivial Sanity Checks):
   - Loads `data/ontario_macro_series.parquet` (StatCan vectors v2062815 & v41690973).
   - Calibrates volatility-scaled macro elasticities (Beta_1, Beta_2) and applies
     log-odds macro shocks via `stress_test_portfolio()`.
   - Executes mandatory zero-shock identity, synthetic Ultra-Prime / High-Distress
     boundary checks, and strict monotonicity assertions.
   - Simulates the severe Ontario recession scenario (+3.0 pp Unemp, +2.5 pp CPI YoY),
     persists `data/pd_stressed.npy`, and plots the baseline vs. stressed KDE shift.
4. Cell 8 (Model Diagnostics, Calibration Reliability & Credit Tier Breakdown):
   - Plots the 2-panel Out-of-Sample ROC overlay and 10-decile Calibration Reliability
     Curve on X_test (N = 4,888).
   - Outputs Credit Grade (A-G) and Baseline Risk Quartile (Q1-Q4) stress breakdown
     tables reporting Count, Share (%), Actual Default (%), Mean PD_base (%),
     Mean PD_stressed (%), Absolute Jump (+pp), and Relative Risk Surge (%).
"""

try:
    from IPython.display import display
except ImportError:

    def display(obj):
        """Fallback display helper for non-IPython terminal execution."""
        print(obj)


# %%
# ==============================================================================
# 5. Partner's Track — Linear Econometric Benchmark (Logistic Regression) & Naive Floor
# ==============================================================================
# CELL 5: PARTNER'S TRACK — LOGISTIC REGRESSION BENCHMARK & NAIVE FLOOR
# ==============================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

# Configure consistent publication-grade plotting aesthetics & seed
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = (
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if "__file__" in globals()
    else os.path.abspath("data")
)
os.makedirs(DATA_DIR, exist_ok=True)


def resolve_data_path(filename):
    """Resolve Phase 1/2 artifact paths from data/ directory or working directory."""
    data_candidate = os.path.join(DATA_DIR, filename)
    if os.path.exists(data_candidate):
        return data_candidate
    return filename


def save_artifact_npy(filename, arr):
    """Persist NumPy array artifact inside DATA_DIR."""
    out_path = os.path.join(DATA_DIR, filename)
    np.save(out_path, arr)
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0, (
        f"Failed to serialize '{out_path}' to disk!"
    )
    return out_path


print("--- INITIATING CELL 5: NAIVE FLOOR & LOGISTIC REGRESSION BENCHMARK ---")

# 1. Load the 6 Verified Phase 1 Microdata Parquet Splits
X_train_sc = pd.read_parquet(resolve_data_path("X_train_processed.parquet"))
X_val_sc = pd.read_parquet(resolve_data_path("X_val_processed.parquet"))
X_test_sc = pd.read_parquet(resolve_data_path("X_test_processed.parquet"))
y_train = pd.read_parquet(resolve_data_path("y_train.parquet"))["loan_status"]
y_val = pd.read_parquet(resolve_data_path("y_val.parquet"))["loan_status"]
y_test = pd.read_parquet(resolve_data_path("y_test.parquet"))["loan_status"]

assert X_train_sc.shape == (22806, 20), f"Unexpected X_train shape: {X_train_sc.shape}"
assert X_val_sc.shape == (4887, 20), f"Unexpected X_val shape: {X_val_sc.shape}"
assert X_test_sc.shape == (4888, 20), f"Unexpected X_test shape: {X_test_sc.shape}"

# 2. Trivial Naive Base-Rate Predictor (Constant Training Default Incidence)
naive_base_rate = float(y_train.mean())
pd_test_naive = np.full(len(y_test), naive_base_rate, dtype=float)

naive_auc = float(roc_auc_score(y_test, pd_test_naive))
naive_brier = float(brier_score_loss(y_test, pd_test_naive))
naive_logloss = float(log_loss(y_test, pd_test_naive))
naive_acc = float(accuracy_score(y_test, (pd_test_naive >= 0.5).astype(int)))
naive_f1 = float(f1_score(y_test, (pd_test_naive >= 0.5).astype(int), zero_division=0))

assert abs(naive_auc - 0.5000) < 1e-6, f"Naive ROC-AUC must be 0.5000, got {naive_auc}"
assert abs(naive_brier - 0.1706) < 5e-4, f"Naive Brier must be ~0.1706, got {naive_brier}"
print(
    f"[OK] Naive Base-Rate Floor (p_hat = {naive_base_rate:.5f}): "
    f"Test ROC-AUC = {naive_auc:.4f}, Test Brier = {naive_brier:.4f}"
)

# 3. L2-Regularized Logistic Regression Validation Tuning (C in [0.01, 0.1, 1.0, 10.0])
c_grid_lr = [0.01, 0.1, 1.0, 10.0]
lr_models = {}
lr_tuning_rows = []

for c_val in c_grid_lr:
    lr_clf = LogisticRegression(C=c_val, max_iter=1000, random_state=RANDOM_SEED)
    lr_clf.fit(X_train_sc, y_train)
    lr_models[c_val] = lr_clf

    pd_tr_lr = lr_clf.predict_proba(X_train_sc)[:, 1]
    pd_va_lr = lr_clf.predict_proba(X_val_sc)[:, 1]

    lr_tuning_rows.append({
        "C (Inverse L2 Penalty)": c_val,
        "Train ROC-AUC (Score)": round(float(roc_auc_score(y_train, pd_tr_lr)), 4),
        "Validation ROC-AUC (Score)": round(float(roc_auc_score(y_val, pd_va_lr)), 4),
        "Validation Brier Score (MSE)": round(float(brier_score_loss(y_val, pd_va_lr)), 4),
        "Validation Log Loss (Nats)": round(float(log_loss(y_val, pd_va_lr)), 4),
    })

lr_tuning_df = pd.DataFrame(lr_tuning_rows)
print("\n--- LOGISTIC REGRESSION (L2) VALIDATION TUNING (Source: Microdata, Val N = 4,887) ---")
display(lr_tuning_df)

best_lr_row = lr_tuning_df.sort_values(
    by=["Validation ROC-AUC (Score)", "Validation Brier Score (MSE)"],
    ascending=[False, True],
).iloc[0]
best_lr_C = float(best_lr_row["C (Inverse L2 Penalty)"])
best_lr = lr_models[best_lr_C]
print(
    f"Selected Optimal Logistic Regression: C = {best_lr_C} "
    f"(Val ROC-AUC = {best_lr_row['Validation ROC-AUC (Score)']:.4f}, "
    f"Val Brier = {best_lr_row['Validation Brier Score (MSE)']:.4f})"
)

# 4. Evaluate Winning Logistic Regression on Held-Out Test Split (N = 4,888)
pd_test_lr = best_lr.predict_proba(X_test_sc)[:, 1]

lr_test_summary = pd.DataFrame([
    {
        "Benchmark Model (Test Split, N = 4,888)": "Naive Base-Rate Floor (Constant y_train Mean)",
        "ROC-AUC (Score)": f"{naive_auc:.4f}",
        "Brier Score (MSE)": f"{naive_brier:.4f}",
        "Log Loss (Nats)": f"{naive_logloss:.4f}",
        "Accuracy (%)": f"{naive_acc * 100:.2f}%",
        "F1-Score (Score)": f"{naive_f1:.4f}",
        "Mean Predicted PD (%)": f"{pd_test_naive.mean() * 100:.2f}%",
        "Actual Default Rate (%)": f"{y_test.mean() * 100:.2f}%",
    },
    {
        "Benchmark Model (Test Split, N = 4,888)": f"Tuned Logistic Regression (L2, C={best_lr_C:g})",
        "ROC-AUC (Score)": f"{roc_auc_score(y_test, pd_test_lr):.4f}",
        "Brier Score (MSE)": f"{brier_score_loss(y_test, pd_test_lr):.4f}",
        "Log Loss (Nats)": f"{log_loss(y_test, pd_test_lr):.4f}",
        "Accuracy (%)": f"{accuracy_score(y_test, (pd_test_lr >= 0.5).astype(int)) * 100:.2f}%",
        "F1-Score (Score)": f"{f1_score(y_test, (pd_test_lr >= 0.5).astype(int)):.4f}",
        "Mean Predicted PD (%)": f"{pd_test_lr.mean() * 100:.2f}%",
        "Actual Default Rate (%)": f"{y_test.mean() * 100:.2f}%",
    },
])
print("\n--- CELL 5 OUT-OF-SAMPLE TEST PERFORMANCE (Source: Borrower Microdata, Test N = 4,888) ---")
display(lr_test_summary)

# 5. Economic Coefficient & Odds-Ratio Interpretation Table (Sorted by |coef|)
coefs = best_lr.coef_[0]
coef_df = (
    pd.DataFrame({
        "Feature": X_train_sc.columns,
        "Log-Odds Coef (beta)": coefs,
        "Abs Coef |beta|": np.abs(coefs),
        "Odds Ratio exp(beta)": np.exp(coefs),
        "Pct Change in Odds (%)": (np.exp(coefs) - 1.0) * 100.0,
        "Economic Direction": np.where(
            coefs > 0, "Increases Default Odds (+)", "Reduces Default Odds (-)"
        ),
    })
    .sort_values(by="Abs Coef |beta|", ascending=False)
    .drop(columns=["Abs Coef |beta|"])
    .reset_index(drop=True)
)

coef_display = coef_df.copy()
coef_display["Log-Odds Coef (beta)"] = coef_display["Log-Odds Coef (beta)"].map(lambda v: f"{v:+.4f}")
coef_display["Odds Ratio exp(beta)"] = coef_display["Odds Ratio exp(beta)"].map(lambda v: f"{v:.4f}x")
coef_display["Pct Change in Odds (%)"] = coef_display["Pct Change in Odds (%)"].map(lambda v: f"{v:+.2f}%")

print(
    f"\n--- ECONOMIC COEFFICIENT & ODDS-RATIO TABLE (L2 Logistic Regression, C={best_lr_C:g}, "
    f"Intercept = {best_lr.intercept_[0]:+.4f}) ---"
)
print("Units: Log-odds change & multiplicative Odds Ratio per +1 SD (continuous/PCA) or 0->1 indicator shift.")
print("Source: Empirical Borrower Microdata (Training Split, N = 22,806).")
with pd.option_context("display.max_rows", 25, "display.max_columns", 10, "display.width", 140):
    display(coef_display)


# %%
# ==============================================================================
# 6. Primary Track — Early-Stage SVM Progression, Validation Tuning & Platt Calibration
# ==============================================================================
# CELL 6: PRIMARY TRACK — SVM PROGRESSION, VALIDATION TUNING & PLATT CALIBRATION
# ==============================================================================
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

print("\n--- INITIATING CELL 6: EARLY-STAGE SVM PROGRESSION & VALIDATION TUNING ---")

# 1. Early-Stage Progression: Linear SVM vs. Default RBF-SVM (Validation N = 4,887)
lin_svm = SVC(kernel="linear", C=1.0, cache_size=1000, random_state=RANDOM_SEED)
lin_svm.fit(X_train_sc, y_train)
lin_tr_auc = float(roc_auc_score(y_train, lin_svm.decision_function(X_train_sc)))
lin_val_auc = float(roc_auc_score(y_val, lin_svm.decision_function(X_val_sc)))

def_rbf_svm = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000, random_state=RANDOM_SEED)
def_rbf_svm.fit(X_train_sc, y_train)
def_tr_auc = float(roc_auc_score(y_train, def_rbf_svm.decision_function(X_train_sc)))
def_val_auc = float(roc_auc_score(y_val, def_rbf_svm.decision_function(X_val_sc)))

progression_df = pd.DataFrame([
    {
        "Early-Stage Model": "Linear SVM (C=1.0, Linear Kernel)",
        "Train ROC-AUC (Score)": f"{lin_tr_auc:.4f}",
        "Validation ROC-AUC (Score)": f"{lin_val_auc:.4f}",
        "Train-Val Gap (Overfit)": f"{lin_tr_auc - lin_val_auc:+.4f}",
        "Val AUC Gain vs. Linear": "+0.0000",
    },
    {
        "Early-Stage Model": "Default RBF-SVM (C=1.0, gamma='scale')",
        "Train ROC-AUC (Score)": f"{def_tr_auc:.4f}",
        "Validation ROC-AUC (Score)": f"{def_val_auc:.4f}",
        "Train-Val Gap (Overfit)": f"{def_tr_auc - def_val_auc:+.4f}",
        "Val AUC Gain vs. Linear": f"{def_val_auc - lin_val_auc:+.4f}",
    },
])
print("\n--- EARLY-STAGE SVM KERNEL PROGRESSION (Source: Microdata, Val N = 4,887) ---")
display(progression_df)

# 2. Validation Hyperparameter Grid Search: Bias-Variance Trade-off Table
C_grid_svm = [0.1, 1.0, 5.0, 10.0]
gamma_grid_svm = ["scale", 0.01, 0.05, 0.1]


def _eval_svm_pair(c_val, g_val):
    svm_cand = SVC(
        kernel="rbf",
        C=c_val,
        gamma=g_val,
        cache_size=1000,
        random_state=RANDOM_SEED,
    )
    svm_cand.fit(X_train_sc, y_train)
    tr_auc = float(roc_auc_score(y_train, svm_cand.decision_function(X_train_sc)))
    va_auc = float(roc_auc_score(y_val, svm_cand.decision_function(X_val_sc)))
    return {
        "C": c_val,
        "gamma_raw": g_val,
        "gamma": str(g_val),
        "Train ROC-AUC (Score)": tr_auc,
        "Validation ROC-AUC (Score)": va_auc,
        "Overfitting Gap (Train - Val)": tr_auc - va_auc,
    }


grid_results = Parallel(n_jobs=-1)(
    delayed(_eval_svm_pair)(c_val, g_val)
    for c_val in C_grid_svm
    for g_val in gamma_grid_svm
)
svm_grid_df = (
    pd.DataFrame(grid_results)
    .sort_values(by="Validation ROC-AUC (Score)", ascending=False)
    .reset_index(drop=True)
)

svm_grid_display = svm_grid_df.drop(columns=["gamma_raw"]).copy()
for col in ["Train ROC-AUC (Score)", "Validation ROC-AUC (Score)", "Overfitting Gap (Train - Val)"]:
    fmt = "{:+.4f}" if "Gap" in col else "{:.4f}"
    svm_grid_display[col] = svm_grid_display[col].map(lambda v, f=fmt: f.format(v))

print(
    "\n--- RBF-SVM VALIDATION GRID SEARCH: BIAS-VARIANCE TRADE-OFF "
    "(Source: Microdata, Train N = 22,806 | Val N = 4,887) ---"
)
with pd.option_context("display.max_rows", 20, "display.max_columns", 10, "display.width", 140):
    display(svm_grid_display)

best_svm_row = svm_grid_df.iloc[0]
best_svm_C = float(best_svm_row["C"])
best_svm_gamma = best_svm_row["gamma_raw"]
print(
    f"Selected Optimal RBF-SVM: C = {best_svm_C:g}, gamma = {best_svm_gamma} "
    f"(Val ROC-AUC = {best_svm_row['Validation ROC-AUC (Score)']:.4f}, "
    f"Overfit Gap = {best_svm_row['Overfitting Gap (Train - Val)']:+.4f})"
)

# 3. 3-Fold Platt Sigmoid Calibration on Optimal RBF-SVM
optimal_rbf_svm = SVC(
    kernel="rbf",
    C=best_svm_C,
    gamma=best_svm_gamma,
    cache_size=1000,
    random_state=RANDOM_SEED,
)
calibrated_svm = CalibratedClassifierCV(
    estimator=optimal_rbf_svm,
    method="sigmoid",
    cv=3,
    n_jobs=-1,
)
print(
    f"\nFitting Calibrated RBF-SVM (C={best_svm_C:g}, gamma={best_svm_gamma}) "
    "with 3-Fold Platt Sigmoid Calibration on X_train_sc..."
)
calibrated_svm.fit(X_train_sc, y_train)

pd_train = calibrated_svm.predict_proba(X_train_sc)[:, 1]
pd_val = calibrated_svm.predict_proba(X_val_sc)[:, 1]
pd_test = calibrated_svm.predict_proba(X_test_sc)[:, 1]
y_pred_prob = pd_test

pd_test_path = save_artifact_npy("pd_test.npy", pd_test)
print(
    f"[OK] Serialized '{pd_test_path}' ({len(pd_test):,d} test records, "
    f"Mean PD_base = {pd_test.mean() * 100:.2f}%)."
)


# 4. Unified Out-of-Sample Test Set Model Comparison Table (N = 4,888)
def _summarize_model(label, probs, y_true):
    preds = (probs >= 0.5).astype(int)
    return {
        "Model (Out-of-Sample Test Split, N = 4,888)": label,
        "ROC-AUC (Score)": f"{roc_auc_score(y_true, probs):.4f}",
        "Brier Score (MSE)": f"{brier_score_loss(y_true, probs):.4f}",
        "Log Loss (Nats)": f"{log_loss(y_true, probs):.4f}",
        "Accuracy (%)": f"{accuracy_score(y_true, preds) * 100:.2f}%",
        "F1-Score (Score)": f"{f1_score(y_true, preds, zero_division=0):.4f}",
        "Mean Predicted PD (%)": f"{probs.mean() * 100:.2f}%",
        "Actual Default Rate (%)": f"{y_true.mean() * 100:.2f}%",
    }


unified_test_df = pd.DataFrame([
    _summarize_model("1. Naive Base-Rate Predictor (Constant y_train Mean)", pd_test_naive, y_test),
    _summarize_model(f"2. Tuned Logistic Regression (L2, C={best_lr_C:g})", pd_test_lr, y_test),
    _summarize_model(
        f"3. Calibrated RBF-SVM (Platt Sigmoid, C={best_svm_C:g}, gamma={best_svm_gamma})",
        pd_test,
        y_test,
    ),
])
print(
    "\n--- UNIFIED OUT-OF-SAMPLE MODEL COMPARISON "
    "(Source: Borrower Credit Microdata, Test Split N = 4,888) ---"
)
with pd.option_context("display.max_columns", 10, "display.width", 160):
    display(unified_test_df)


# %%
# ==============================================================================
# 7. Ontario Macroeconomic Stress-Testing Engine & Trivial Sanity Checks
# ==============================================================================
# CELL 7: ONTARIO MACROECONOMIC STRESS-TESTING ENGINE & TRIVIAL SANITY CHECKS
# ==============================================================================
print("\n--- INITIATING CELL 7: ONTARIO MACROECONOMIC STRESS-TESTING ENGINE ---")

# 1. Load Verified StatCan Ontario Macro Series & Calibrated Baseline PDs
df_macro = pd.read_parquet(resolve_data_path("ontario_macro_series.parquet"))
if "pd_test" not in locals():
    pd_test = np.load(resolve_data_path("pd_test.npy"))

# 2. Calibrate Volatility-Scaled Ontario Macro Elasticities
hist_unemp_sd = float(df_macro["Ontario_Unemployment_Rate"].std())
hist_inf_sd = float(df_macro["Ontario_CPI_YoY"].std())

beta_unemployment = 0.35 / max(hist_unemp_sd, 0.1)
beta_inflation = 0.15 / max(hist_inf_sd, 0.1)

print(
    f"Historical StatCan Std Devs -> Unemp: {hist_unemp_sd:.4f} pp | "
    f"CPI YoY: {hist_inf_sd:.4f} pp"
)
print(
    f"Calibrated Unemployment Sensitivity (Beta_1): {beta_unemployment:.4f} "
    "log-odds per +1 pp Unemp"
)
print(
    f"Calibrated Inflation Sensitivity    (Beta_2): {beta_inflation:.4f} "
    "log-odds per +1 pp CPI YoY"
)


# 3. Econometric Logistic-Link Stress-Testing Function
def stress_test_portfolio(unemployment_shift, inflation_shift, base_pds):
    """Map Ontario macro shocks (percentage points) into default probabilities:
    logit(PD_stressed) = logit(PD_base) + beta_1 * Delta_Unemp + beta_2 * Delta_CPI
    """
    eps = 1e-12
    p_arr = np.asarray(base_pds, dtype=float)
    safe_pds = np.clip(p_arr, eps, 1.0 - eps)
    logit_pds = np.log(safe_pds / (1.0 - safe_pds))
    stressed_logits = (
        logit_pds
        + (beta_unemployment * unemployment_shift)
        + (beta_inflation * inflation_shift)
    )
    bounded_logits = np.clip(stressed_logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-bounded_logits))


# 4. Mandatory Trivial / Known-Case Sanity Check Assertions
# Check A: Zero-shock identity
pd_zero_shock = stress_test_portfolio(0.0, 0.0, pd_test)
max_zero_diff = float(np.max(np.abs(pd_zero_shock - pd_test)))
assert np.allclose(pd_zero_shock, pd_test, atol=1e-7), (
    f"Zero-shock identity failed! Max deviation = {max_zero_diff:.2e}"
)

# Check B: Synthetic Ultra-Prime & High-Distress Profile Boundary + Monotonicity
ultra_prime = pd.DataFrame([{col: 0.0 for col in X_train_sc.columns}])
ultra_prime["person_age"] = 0.5
ultra_prime["loan_int_rate"] = -1.5
ultra_prime["person_home_ownership_OWN"] = 1.0
ultra_prime["loan_intent_VENTURE"] = 1.0
ultra_prime["PC1_Financial_Scale"] = -1.2
ultra_prime["PC2_Tenure_Stability"] = 2.0
ultra_prime["PC3_Leverage_Burden"] = -0.8

high_distress = pd.DataFrame([{col: 0.0 for col in X_train_sc.columns}])
high_distress["person_age"] = -0.3
high_distress["loan_int_rate"] = 1.4
high_distress["person_home_ownership_RENT"] = 1.0
high_distress["loan_intent_MEDICAL"] = 1.0
high_distress["loan_grade_D"] = 1.0
high_distress["cb_person_default_on_file_Y"] = 1.0
high_distress["PC1_Financial_Scale"] = 1.0
high_distress["PC2_Tenure_Stability"] = -1.2
high_distress["PC3_Leverage_Burden"] = 0.5

pd_prime_base = float(calibrated_svm.predict_proba(ultra_prime)[0, 1])
pd_distress_base = float(calibrated_svm.predict_proba(high_distress)[0, 1])

assert pd_prime_base < 0.05, f"Ultra-Prime PD_base >= 0.05: {pd_prime_base:.4f}"
assert pd_distress_base > 0.70, f"High-Distress PD_base <= 0.70: {pd_distress_base:.4f}"

pd_prime_1pp = float(stress_test_portfolio(1.0, 0.0, pd_prime_base))
pd_prime_3pp = float(stress_test_portfolio(3.0, 0.0, pd_prime_base))
pd_distress_1pp = float(stress_test_portfolio(1.0, 0.0, pd_distress_base))
pd_distress_3pp = float(stress_test_portfolio(3.0, 0.0, pd_distress_base))

assert pd_prime_3pp > pd_prime_1pp > pd_prime_base, "Ultra-Prime monotonicity failed!"
assert pd_distress_3pp > pd_distress_1pp > pd_distress_base, "High-Distress monotonicity failed!"

pd_test_unemp_1pp = stress_test_portfolio(1.0, 0.0, pd_test)
pd_test_unemp_3pp = stress_test_portfolio(3.0, 0.0, pd_test)
assert bool((pd_test_unemp_1pp > pd_test).all()), "Portfolio +1pp unemp monotonicity failed!"
assert bool((pd_test_unemp_3pp > pd_test_unemp_1pp).all()), "Portfolio +3pp unemp monotonicity failed!"

sanity_df = pd.DataFrame([
    {
        "Sanity Check": "1. Zero-Shock Identity (Delta Unemp=0, Delta CPI=0)",
        "Observed Result": f"Max |PD(0,0) - PD_base| = {max_zero_diff:.2e}",
        "Acceptance Criterion": "< 1.0e-07",
        "Status": "PASS",
    },
    {
        "Sanity Check": "2. Synthetic Ultra-Prime Borrower Baseline PD",
        "Observed Result": f"PD_base = {pd_prime_base * 100:.2f}% -> +3pp Unemp: {pd_prime_3pp * 100:.2f}%",
        "Acceptance Criterion": "PD_base < 5.00% & Monotonic",
        "Status": "PASS",
    },
    {
        "Sanity Check": "3. Synthetic High-Distress Borrower Baseline PD",
        "Observed Result": f"PD_base = {pd_distress_base * 100:.2f}% -> +3pp Unemp: {pd_distress_3pp * 100:.2f}%",
        "Acceptance Criterion": "PD_base > 70.00% & Monotonic",
        "Status": "PASS",
    },
    {
        "Sanity Check": "4. Test Portfolio Strict Monotonicity (+1pp -> +3pp Unemp)",
        "Observed Result": "100% of 4,888 borrowers strictly increase",
        "Acceptance Criterion": "All PD(+3pp) > PD(+1pp) > PD_base",
        "Status": "PASS",
    },
])
print("\n--- MANDATORY TRIVIAL & MONOTONICITY SANITY CHECK AUDIT ---")
display(sanity_df)

# 5. Simulate Severe Ontario Recession Scenario (+3.0 pp Unemp, +2.5 pp YoY CPI)
unemp_shock = 3.0
inf_shock = 2.5
pd_stressed = stress_test_portfolio(unemp_shock, inf_shock, pd_test)

pd_stressed_path = save_artifact_npy("pd_stressed.npy", pd_stressed)
print(
    f"\n[OK] Serialized '{pd_stressed_path}' ({len(pd_stressed):,d} test records)."
)

# 6. Publication-Grade Distribution Shift Figure (with Explicit Units & Sources)
fig, ax = plt.subplots(figsize=(10, 5.2))
sns.kdeplot(
    pd_test,
    fill=True,
    clip=(0.0, 1.0),
    color="#2b5c8f",
    alpha=0.45,
    linewidth=2,
    ax=ax,
    label=f"Baseline Portfolio (Mean PD = {pd_test.mean() * 100:.2f}%)",
)
sns.kdeplot(
    pd_stressed,
    fill=True,
    clip=(0.0, 1.0),
    color="#d95f02",
    alpha=0.45,
    linewidth=2,
    ax=ax,
    label=(
        f"Ontario Recession Shock (+{unemp_shock:.1f} pp Unemp, +{inf_shock:.1f} pp CPI YoY; "
        f"Mean PD = {pd_stressed.mean() * 100:.2f}%)"
    ),
)

ax.axvline(
    pd_test.mean(),
    color="#2b5c8f",
    linestyle="--",
    lw=1.8,
    label=f"Baseline Mean ({pd_test.mean() * 100:.2f}%)",
)
ax.axvline(
    pd_stressed.mean(),
    color="#d95f02",
    linestyle="--",
    lw=1.8,
    label=f"Stressed Mean ({pd_stressed.mean() * 100:.2f}%)",
)

ax.set_title(
    "Ontario Macroeconomic Stress Test: Portfolio Probability of Default ($PD$) Shift",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Probability of Default ($PD$, Proportion in [0, 1])", fontweight="bold")
ax.set_ylabel("Kernel Density Estimate (Density)", fontweight="bold")
ax.set_xlim(-0.02, 1.02)
ax.legend(loc="upper right", frameon=True)
fig.text(
    0.5,
    -0.02,
    "Source: Borrower Credit Risk Microdata (Held-Out Test Split, N = 4,888) & "
    "Statistics Canada WDS API (Ontario Unemployment v2062815 & CPI v41690973, 2017–2026).",
    ha="center",
    fontsize=9,
    style="italic",
)
plt.tight_layout()
plt.show()

print("\n--- MACRO STRESS ENGINE SUMMARY (TEST PORTFOLIO, N = 4,888) ---")
print(f"Mean Baseline PD (PD_base):      {pd_test.mean() * 100:.2f}%")
print(f"Mean Stressed PD (PD_stressed):  {pd_stressed.mean() * 100:.2f}%")
print(f"Average Absolute PD Jump:       +{(pd_stressed.mean() - pd_test.mean()) * 100:.2f} pp")
print(
    f"Average Relative Risk Surge:    +"
    f"{((pd_stressed.mean() - pd_test.mean()) / pd_test.mean()) * 100:.2f}%"
)


# %%
# ==============================================================================
# 8. Model Diagnostics, Calibration Reliability Curve & Credit Tier Stress Breakdown
# ==============================================================================
# CELL 8: MODEL DIAGNOSTICS, CALIBRATION RELIABILITY & CREDIT TIER BREAKDOWN
# ==============================================================================
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

print("\n--- INITIATING CELL 8: TEST DIAGNOSTICS & CREDIT TIER STRESS BREAKDOWN ---")

# 1. Two-Panel Diagnostic Figure on Held-Out Test Split (N = 4,888)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# Panel A: ROC Curve Overlay (RBF-SVM vs. Logistic Regression vs. Naive Floor)
fpr_svm, tpr_svm, _ = roc_curve(y_test, pd_test)
fpr_lr, tpr_lr, _ = roc_curve(y_test, pd_test_lr)
auc_svm = float(roc_auc_score(y_test, pd_test))
auc_lr = float(roc_auc_score(y_test, pd_test_lr))

axes[0].plot(
    fpr_svm,
    tpr_svm,
    color="#2b5c8f",
    lw=2.2,
    label=f"Calibrated RBF-SVM (AUC = {auc_svm:.4f})",
)
axes[0].plot(
    fpr_lr,
    tpr_lr,
    color="#d95f02",
    lw=2.0,
    linestyle="--",
    label=f"Tuned Logistic Regression (AUC = {auc_lr:.4f})",
)
axes[0].plot(
    [0, 1],
    [0, 1],
    color="#666666",
    lw=1.3,
    linestyle=":",
    label="Naive Base-Rate Floor (AUC = 0.5000)",
)
axes[0].set_title(
    "Panel A: Out-of-Sample ROC Curve Comparison (Test N = 4,888)",
    fontweight="bold",
    fontsize=11,
)
axes[0].set_xlabel("False Positive Rate (1 - Specificity, Proportion)", fontweight="bold")
axes[0].set_ylabel("True Positive Rate (Sensitivity, Proportion)", fontweight="bold")
axes[0].legend(loc="lower right", frameon=True)
axes[0].grid(alpha=0.3)

# Panel B: 10-Decile Calibration Reliability Curve
prob_true_svm, prob_pred_svm = calibration_curve(
    y_test, pd_test, n_bins=10, strategy="quantile"
)
prob_true_lr, prob_pred_lr = calibration_curve(
    y_test, pd_test_lr, n_bins=10, strategy="quantile"
)
brier_svm = float(brier_score_loss(y_test, pd_test))
brier_lr = float(brier_score_loss(y_test, pd_test_lr))

axes[1].plot(
    prob_pred_svm,
    prob_true_svm,
    marker="o",
    color="#2b5c8f",
    lw=2.2,
    label=f"Calibrated RBF-SVM (Brier = {brier_svm:.4f})",
)
axes[1].plot(
    prob_pred_lr,
    prob_true_lr,
    marker="s",
    color="#d95f02",
    lw=1.9,
    linestyle="--",
    label=f"Tuned Logistic Regression (Brier = {brier_lr:.4f})",
)
axes[1].plot(
    [0, 1],
    [0, 1],
    color="#666666",
    lw=1.3,
    linestyle=":",
    label="Perfect Calibration (45-Degree Line)",
)
axes[1].set_title(
    "Panel B: Probability Calibration Curve (10 Decile Bins, Test N = 4,888)",
    fontweight="bold",
    fontsize=11,
)
axes[1].set_xlabel("Mean Predicted Probability PD_base (Proportion in [0, 1])", fontweight="bold")
axes[1].set_ylabel("Observed Empirical Default Frequency (Proportion)", fontweight="bold")
axes[1].legend(loc="upper left", frameon=True)
axes[1].grid(alpha=0.3)

fig.text(
    0.5,
    -0.02,
    "Source: Held-Out Test Split (N = 4,888), Empirical Borrower Credit Risk Microdata "
    "(3-Fold Platt Sigmoid Calibration on Optimal RBF-SVM).",
    ha="center",
    fontsize=9,
    style="italic",
)
plt.tight_layout()
plt.show()

# 2. Credit-Grade Stress Sensitivity Breakdown (Grades A-G from One-Hot Encodings)
grade_labels = np.full(len(X_test_sc), "Grade A (Prime Ref)", dtype=object)
for g_letter in ["B", "C", "D", "E", "F", "G"]:
    col_name = f"loan_grade_{g_letter}"
    if col_name in X_test_sc.columns:
        mask = X_test_sc[col_name].to_numpy() == 1
        grade_labels[mask] = f"Grade {g_letter}"

tier_eval_df = pd.DataFrame({
    "Credit Grade": grade_labels,
    "Actual Default": y_test.to_numpy(dtype=float),
    "PD_base": pd_test,
    "PD_stressed": pd_stressed,
})

grade_order = [
    "Grade A (Prime Ref)",
    "Grade B",
    "Grade C",
    "Grade D",
    "Grade E",
    "Grade F",
    "Grade G",
]
grade_summary = (
    tier_eval_df.groupby("Credit Grade", observed=False)
    .agg(
        Count=("Actual Default", "count"),
        Actual_Default_Rate=("Actual Default", "mean"),
        Mean_PD_base=("PD_base", "mean"),
        Mean_PD_stressed=("PD_stressed", "mean"),
    )
    .reindex([g for g in grade_order if g in tier_eval_df["Credit Grade"].unique()])
    .reset_index()
)
grade_summary["Share (%)"] = grade_summary["Count"] / len(tier_eval_df) * 100.0
grade_summary["Jump (+pp)"] = (
    grade_summary["Mean_PD_stressed"] - grade_summary["Mean_PD_base"]
) * 100.0
grade_summary["Relative Surge (%)"] = (
    (grade_summary["Mean_PD_stressed"] - grade_summary["Mean_PD_base"])
    / grade_summary["Mean_PD_base"]
) * 100.0

grade_display = pd.DataFrame({
    "Credit Grade": grade_summary["Credit Grade"],
    "Borrowers (N)": grade_summary["Count"].map(lambda v: f"{v:,d}"),
    "Portfolio Share (%)": grade_summary["Share (%)"].map(lambda v: f"{v:.2f}%"),
    "Actual Default (%)": grade_summary["Actual_Default_Rate"].map(lambda v: f"{v * 100:.2f}%"),
    "Mean PD_base (%)": grade_summary["Mean_PD_base"].map(lambda v: f"{v * 100:.2f}%"),
    "Mean PD_stressed (%)": grade_summary["Mean_PD_stressed"].map(lambda v: f"{v * 100:.2f}%"),
    "Absolute Jump (+pp)": grade_summary["Jump (+pp)"].map(lambda v: f"+{v:.2f} pp"),
    "Relative Risk Surge (%)": grade_summary["Relative Surge (%)"].map(lambda v: f"+{v:.2f}%"),
})

print(
    "\n--- CREDIT GRADE STRESS SENSITIVITY TABLE "
    "(Source: Test Split N = 4,888 + StatCan Shock +3.0 pp Unemp / +2.5 pp CPI YoY) ---"
)
with pd.option_context("display.max_columns", 10, "display.width", 160):
    display(grade_display)

# 3. Baseline Risk-Quartile Stress Sensitivity Breakdown (Q1-Q4)
tier_eval_df["Risk Quartile"] = pd.qcut(
    tier_eval_df["PD_base"],
    q=4,
    labels=[
        "Q1: Ultra-Prime (Lowest Risk)",
        "Q2: Prime / Moderate",
        "Q3: Near-Prime / Leveraged",
        "Q4: High Distress (Highest Risk)",
    ],
)

quartile_summary = (
    tier_eval_df.groupby("Risk Quartile", observed=False)
    .agg(
        Count=("Actual Default", "count"),
        Actual_Default_Rate=("Actual Default", "mean"),
        Mean_PD_base=("PD_base", "mean"),
        Mean_PD_stressed=("PD_stressed", "mean"),
    )
    .reset_index()
)
quartile_summary["Share (%)"] = quartile_summary["Count"] / len(tier_eval_df) * 100.0
quartile_summary["Jump (+pp)"] = (
    quartile_summary["Mean_PD_stressed"] - quartile_summary["Mean_PD_base"]
) * 100.0
quartile_summary["Relative Surge (%)"] = (
    (quartile_summary["Mean_PD_stressed"] - quartile_summary["Mean_PD_base"])
    / quartile_summary["Mean_PD_base"]
) * 100.0

quartile_display = pd.DataFrame({
    "Baseline Risk Quartile": quartile_summary["Risk Quartile"],
    "Borrowers (N)": quartile_summary["Count"].map(lambda v: f"{v:,d}"),
    "Portfolio Share (%)": quartile_summary["Share (%)"].map(lambda v: f"{v:.2f}%"),
    "Actual Default (%)": quartile_summary["Actual_Default_Rate"].map(lambda v: f"{v * 100:.2f}%"),
    "Mean PD_base (%)": quartile_summary["Mean_PD_base"].map(lambda v: f"{v * 100:.2f}%"),
    "Mean PD_stressed (%)": quartile_summary["Mean_PD_stressed"].map(lambda v: f"{v * 100:.2f}%"),
    "Absolute Jump (+pp)": quartile_summary["Jump (+pp)"].map(lambda v: f"+{v:.2f} pp"),
    "Relative Risk Surge (%)": quartile_summary["Relative Surge (%)"].map(lambda v: f"+{v:.2f}%"),
})

print(
    "\n--- BASELINE RISK QUARTILE STRESS SENSITIVITY TABLE "
    "(Source: Test Split N = 4,888 + StatCan Shock +3.0 pp Unemp / +2.5 pp CPI YoY) ---"
)
with pd.option_context("display.max_columns", 10, "display.width", 160):
    display(quartile_display)

peak_grade = grade_summary.loc[grade_summary["Jump (+pp)"].idxmax()]
peak_quartile = quartile_summary.loc[quartile_summary["Jump (+pp)"].idxmax()]
print(
    f"\nKey Economic Takeaway: Peak absolute risk expansion occurs in {peak_grade['Credit Grade']} "
    f"(+{peak_grade['Jump (+pp)']:.2f} pp) and {peak_quartile['Risk Quartile']} "
    f"(+{peak_quartile['Jump (+pp)']:.2f} pp), where borrowers sit near the steepest curvature "
    "of the logistic link function."
)
