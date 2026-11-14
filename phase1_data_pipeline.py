"""
ECON 354: Interpretable Machine Learning for Baseline Borrower Default Risk (PD_base)
& Ontario Macroeconomic Sensitivity
Phase 1: Real-Data Pipeline, Leakage-Free Preprocessing & Dimensionality Reduction

Authors: Alex Babiak, Marco Puiatti
Scope: Ontario, Canada

Research Question
-----------------
"How can machine learning be used to estimate individual borrower baseline default risk
(PD_base), and how sensitive is that baseline risk distribution to Ontario macroeconomic
shocks in unemployment and inflation?"

Datasets Used & Justifications (100% Real Data — Zero Synthetic Data)
---------------------------------------------------------------------
1. Individual Borrower Microdata (`credit_risk_dataset.csv` — 32,581 records, 12 variables):
   - Source: Programmatically fetched via `urllib.request` from
     https://raw.githubusercontent.com/LeHongNgoc3820/Project_Credit_Risk/main/credit_risk_dataset.csv
   - Why it is used: Institutional Canadian mortgage loan-level files (e.g., CMHC or
     Schedule I banks) are restricted under PIPEDA privacy regulations and proprietary
     trade secrecy. This 32,581-record empirical credit dataset serves as a transparent
     micro-level proxy capturing authentic borrower balance-sheet relationships
     (`person_income`, `loan_amnt`, `loan_percent_income`), credit and employment
     stability (`cb_person_cred_hist_length`, `person_emp_length`), loan pricing/risk
     grades (`loan_int_rate`, `loan_grade`), and observed binary default outcomes
     (`loan_status`, ~21.82% default incidence). Using real microdata preserves genuine
     right-skewed financial tails, missingness patterns, and feature multicollinearity.

2. Ontario Macroeconomic Time-Series (Statistics Canada Web Data Service API):
   - Source: Programmatically queried via POST request to StatCan's public WDS API
     (https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods)
     for Ontario Monthly Unemployment Rate (`v2062815`, Table 14-10-0287-01) and
     Ontario All-Items CPI (`v41690973`, Table 18-10-0004-01), computing 12-month
     Year-over-Year inflation (`Ontario_CPI_YoY`).
   - Why it is used: Cross-sectional borrower data is "point-in-time" and blind to
     business-cycle fluctuations. Pairing individual baseline default probabilities
     (PD_base) with official Ontario labor-market and inflation series allows us to test
     how regional macroeconomic shocks perturb baseline borrower risk distributions
     (used strictly as environmental stress parameters on PD_base, not for insurance
     pricing).

Explicit Model Outcomes
-----------------------
1. Primary Binary Outcome (y_i in {0, 1}): Observed borrower default indicator
   (`loan_status`, where 1 = default and 0 = performing; 21.82% population default
   incidence).
2. Primary Continuous Outcome (PD_{base, i} in [0, 1]): Calibrated individual baseline
   Probability of Default generated via Platt sigmoid scaling on the RBF-SVM decision
   function f(x_i) (benchmarked against Logistic Regression).
3. Secondary Stress Outcome (PD_{stressed, i} in [0, 1]): Macro-adjusted borrower
   default probability resulting from shifting baseline log-odds by Ontario unemployment
   and CPI inflation shocks.
4. Out-of-Sample Evaluation Outcomes (Test Set, N = 4,888): Discriminatory power
   (ROC-AUC, F1-Score, Accuracy) and probability calibration (Brier Score, Log Loss,
   Mean Predicted PD vs. Actual Default Rate).

Explicit Model Parameters
-------------------------
1. Raw Applicant Inputs (x_raw): 11 borrower predictors (5 collinear financial variables,
   2 continuous demographic/rate controls, and 4 categorical indicators expanded via
   one-hot encoding into 23 features, then compressed to 20 post-PCA features).
2. Preprocessing Parameters: Training-set medians (m_train) for missing value imputation
   (`person_emp_length`, `loan_int_rate`), training-set Tukey 1.5 x IQR fences
   ([Q1 - 1.5*IQR, Q3 + 1.5*IQR]) for outlier capping (`person_income`, `loan_amnt`),
   and training-set `StandardScaler` moments (mu_train, sigma_train).
3. Dimensionality Reduction Parameters: PCA eigenvector projection matrix W in R^{5 x 3}
   capturing 81.90% of continuous financial variance across three orthogonal risk factors
   (`PC1_Financial_Scale`, `PC2_Tenure_Stability`, `PC3_Leverage_Burden`).
4. Classifier & Macro Sensitivity Parameters: SVM soft-margin regularization (C), RBF
   kernel bandwidth (gamma), Platt scaling parameters (A, B), and Ontario macro
   elasticities (beta_1, beta_2).
"""

try:
    from IPython.display import display
except ImportError:

    def display(obj):
        """Fallback display helper for non-IPython terminal execution."""
        print(obj)


# %%
# ==============================================================================
# 1. Environment Setup & Real-Data Ingestion (Borrower Microdata + StatCan WDS API)
# ==============================================================================
# We consolidate all Phase 1 libraries and ingest our two real-world datasets:
# 1. Borrower Microdata (`credit_risk_dataset.csv`): 32,581 empirical borrower records
#    fetched via `urllib.request`, serving as a transparent micro-level proxy given
#    PIPEDA and proprietary restrictions on institutional Canadian mortgage files.
# 2. Ontario Macroeconomic Time-Series: Programmatically queried via Statistics Canada's
#    public WDS API for Ontario's monthly Unemployment Rate (`v2062815`, Table
#    14-10-0287-01) and All-Items CPI (`v41690973`, Table 18-10-0004-01), computing
#    12-month YoY inflation (`Ontario_CPI_YoY`) and persisting `ontario_macro_series.parquet`
#    for Phase 3 macro-sensitivity testing of PD_base.
# ==============================================================================
# CELL 1: ENVIRONMENT SETUP & REAL-DATA INGESTION (100% REAL DATA)
# ==============================================================================
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure publication-grade plotting aesthetics
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

# ------------------------------------------------------------------------------
# STEP 1: INGEST REAL BORROWER MICRODATA (32,581 RECORDS, 12 VARIABLES)
# Proxy for PIPEDA-restricted Canadian institutional mortgage files
# ------------------------------------------------------------------------------
print("--- STEP 1: INGESTING REAL MICRO CREDIT RISK DATASET ---")
micro_url = (
    "https://raw.githubusercontent.com/LeHongNgoc3820/Project_Credit_Risk/main/"
    "credit_risk_dataset.csv"
)
micro_dest = "credit_risk_dataset.csv"

try:
    urllib.request.urlretrieve(micro_url, micro_dest)
    print("Credit risk dataset downloaded successfully.")
except Exception as e:
    if os.path.exists(micro_dest) and os.path.getsize(micro_dest) > 0:
        print(
            f"Remote download unavailable ({e}); using existing local '{micro_dest}'."
        )
    else:
        raise RuntimeError(
            f"Critical failure downloading micro credit data: {e}"
        ) from e

df_raw = pd.read_csv(micro_dest)
assert df_raw.shape == (32581, 12), f"Unexpected microdata shape: {df_raw.shape}"

print(f"Raw Microdata Shape: {df_raw.shape}")
print("\n--- Microdata Missing Values ---")
print(df_raw.isnull().sum())
print("\n--- Microdata Information ---")
df_raw.info()

# ------------------------------------------------------------------------------
# STEP 2: PROGRAMMATIC STATISTICS CANADA WDS API EXTRACTION (ONTARIO MACRO)
# Environmental stress series for Phase 3 macro-sensitivity testing of PD_base
# ------------------------------------------------------------------------------
print("\n--- STEP 2: PROGRAMMATIC STATISTICS CANADA WDS API FETCH (100% REAL DATA) ---")
statcan_url = (
    "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
)
headers = {"Content-Type": "application/json"}

# Vector 2062815: Ontario Monthly Unemployment Rate (Table 14-10-0287-01)
# Vector 41690973: Ontario All-Items Consumer Price Index (Table 18-10-0004-01)
payload = [
    {"vectorId": 2062815, "latestN": 120},
    {"vectorId": 41690973, "latestN": 120},
]
macro_dest = "ontario_macro_series.parquet"

try:
    response = requests.post(statcan_url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    api_data = response.json()

    series_list = []
    for series in api_data:
        # Parse nested StatCan JSON hierarchy: series["object"]["vectorId"]
        obj = series["object"]
        vec_id = int(obj["vectorId"])
        points = obj["vectorDataPoint"]
        for pt in points:
            if pt.get("value") is not None:
                series_list.append({
                    "vectorId": vec_id,
                    "refPer": pt["refPer"],
                    "value": float(pt["value"]),
                })

    df_macro_raw = pd.DataFrame(series_list)

    # Pivot wide on reference period (refPer) and clean axis metadata
    df_macro = df_macro_raw.pivot(
        index="refPer", columns="vectorId", values="value"
    ).reset_index()
    df_macro.columns.name = None
    df_macro.rename(
        columns={
            2062815: "Ontario_Unemployment_Rate",
            41690973: "Ontario_CPI",
        },
        inplace=True,
    )

    df_macro["Date"] = pd.to_datetime(df_macro["refPer"], format="%Y-%m-%d")
    df_macro = df_macro.sort_values("Date").reset_index(drop=True)

    # Drop any unaligned trailing/leading release months prior to YoY differencing
    df_macro.dropna(subset=["Ontario_Unemployment_Rate", "Ontario_CPI"], inplace=True)
    df_macro.reset_index(drop=True, inplace=True)

    # Compute Year-over-Year (YoY) CPI inflation without deprecated implicit padding
    df_macro["Ontario_CPI_YoY"] = (
        df_macro["Ontario_CPI"].pct_change(periods=12, fill_method=None) * 100.0
    )

    # Drop initial 12 differencing NaNs and verify zero remaining nulls
    df_macro.dropna(subset=["Ontario_CPI_YoY"], inplace=True)
    df_macro.reset_index(drop=True, inplace=True)
    assert df_macro.isnull().sum().sum() == 0, "NaNs detected in df_macro!"

    # Serialize clean Ontario macro time-series for Phase 3 PD_base sensitivity testing
    df_macro.to_parquet(macro_dest, index=False)

    print(
        "\nStatCan Real Macro Data parsed and serialized to "
        "'ontario_macro_series.parquet' for Phase 3 PD_base macro-sensitivity testing."
    )
    print(f"Macrodata Shape: {df_macro.shape}")
    display(df_macro.head())
    display(df_macro.tail())
except Exception as e:
    if os.path.exists(macro_dest) and os.path.getsize(macro_dest) > 0:
        print(
            f"StatCan API unavailable ({e}); loading cached real StatCan series from '{macro_dest}'."
        )
        df_macro = pd.read_parquet(macro_dest)
        assert df_macro.isnull().sum().sum() == 0, "NaNs detected in cached df_macro!"
        print(f"Macrodata Shape: {df_macro.shape}")
        display(df_macro.head())
        display(df_macro.tail())
    else:
        print("StatCan Request failed. Raw Exception detail:", e)
        if "response" in locals() and hasattr(response, "text"):
            print("Server Response Text:", response.text)
        raise RuntimeError(
            "Macro data fetch failed. Terminating to preserve 100% Real Data constraint."
        ) from e


# %%
# ==============================================================================
# 2. Leakage-Free Exploratory Data Analysis & Preprocessing
# ==============================================================================
# We perform out-of-sample disciplined preprocessing on the real borrower microdata:
# 1. Median Imputation: Missing values in `person_emp_length` and `loan_int_rate` are
#    imputed using medians fitted strictly on the 70% stratified training index (`train_idx`).
# 2. Tukey's 1.5 x IQR Outlier Capping: Monetary variables (`person_income`, `loan_amnt`)
#    are Winsorized at [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] using training-split quartiles.
#    Note that `loan_amnt` is capped and retained strictly for leverage ratio calculation
#    and PCA compression (not as an unscaled exposure input).
# 3. Feature Engineering & One-Hot Encoding: We recompute
#    `loan_percent_income = loan_amnt / person_income`, one-hot encode categorical variables
#    (`drop_first=True`, `dtype=int`), and render the Pearson correlation heatmap across
#    the 5 continuous financial features.
# ==============================================================================
# CELL 2: LEAKAGE-FREE EXPLORATORY DATA ANALYSIS & PREPROCESSING
# ==============================================================================
df_clean = df_raw.copy()

# Identify the exact 70% stratified training index so median and IQR statistics
# are estimated strictly on training observations (zero pre-split leakage).
train_idx, _ = train_test_split(
    df_clean.index,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=df_clean["loan_status"],
)

# ------------------------------------------------------------------------------
# STEP 1: IMPUTING MISSING MICRODATA (TRAINING-ONLY ROBUST MEDIANS)
# ------------------------------------------------------------------------------
print("--- STEP 1: IMPUTING MISSING MICRODATA ---")
person_emp_length_med = df_clean.loc[train_idx, "person_emp_length"].median()
loan_int_rate_med = df_clean.loc[train_idx, "loan_int_rate"].median()

df_clean["person_emp_length"] = df_clean["person_emp_length"].fillna(
    person_emp_length_med
)
df_clean["loan_int_rate"] = df_clean["loan_int_rate"].fillna(loan_int_rate_med)

print(f"Imputed person_emp_length with median: {person_emp_length_med:.1f} years")
print(f"Imputed loan_int_rate with median:     {loan_int_rate_med:.2f}%")
print(f"Remaining nulls in dataset:            {df_clean.isnull().sum().sum()}")

# ------------------------------------------------------------------------------
# STEP 2: TUKEY 1.5x IQR OUTLIER CAPPING (TRAINING-ONLY FENCES)
# loan_amnt is capped and retained strictly for leverage ratio calculation and PCA
# ------------------------------------------------------------------------------
print("\n--- STEP 2: TUKEY 1.5x IQR OUTLIER CAPPING ---")
for col in ["person_income", "loan_amnt"]:
    q25 = df_clean.loc[train_idx, col].quantile(0.25)
    q75 = df_clean.loc[train_idx, col].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    num_capped_lower = int((df_clean[col] < lower_bound).sum())
    num_capped_upper = int((df_clean[col] > upper_bound).sum())

    print(
        f"{col:15s} | IQR Bounds: [${lower_bound:10.2f}, ${upper_bound:10.2f}] "
        f"| Capped Below: {num_capped_lower:4d} | Capped Above: {num_capped_upper:4d}"
    )
    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

# ------------------------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING (RECALCULATE LEVERAGE RATIO)
# ------------------------------------------------------------------------------
print("\n--- STEP 3: FEATURE ENGINEERING ---")
df_clean["loan_percent_income"] = (
    df_clean["loan_amnt"] / df_clean["person_income"]
)
print(
    "Recalculated loan_percent_income summary: "
    f"min={df_clean['loan_percent_income'].min():.4f}, "
    f"max={df_clean['loan_percent_income'].max():.4f}, "
    f"mean={df_clean['loan_percent_income'].mean():.4f}"
)

# ------------------------------------------------------------------------------
# STEP 4: CATEGORICAL ONE-HOT ENCODING (DROP_FIRST=TRUE, DTYPE=INT)
# ------------------------------------------------------------------------------
print("\n--- STEP 4: CATEGORICAL ONE-HOT ENCODING ---")
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
df_clean = pd.get_dummies(
    df_clean, columns=categorical_cols, drop_first=True, dtype=int
)
print(f"Post-encoding dataset shape: {df_clean.shape}")

# ------------------------------------------------------------------------------
# STEP 5: MULTICOLLINEARITY DIAGNOSTICS (PEARSON CORRELATION HEATMAP)
# ------------------------------------------------------------------------------
print("\n--- STEP 5: MULTICOLLINEARITY CORRELATION HEATMAP ---")
continuous_financials = [
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_emp_length",
]

corr_matrix = df_clean.loc[train_idx, continuous_financials].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation Coefficient"},
)
plt.title(
    "Correlation Matrix: Continuous Financial Features (Pre-PCA)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
# ==============================================================================
# 3. Stratified Partitioning, Leakage-Free Scaling & PCA
# ==============================================================================
# We partition the dataset into 70% Train (N = 22,806), 15% Validation (N = 4,887), and
# 15% Test (N = 4,888) splits using `stratify=y` and `random_state=42` to preserve the
# exact population default rate (~21.82%). `StandardScaler` is fitted exclusively on
# `X_train_raw` and applied via `.transform()` to Validation and Test across both the 5
# PCA financial variables and the 2 remaining continuous features (`person_age`,
# `loan_int_rate`). Finally, we fit `PCA(n_components=3)` on the scaled financial subset,
# verify >80% cumulative explained variance, inspect the orthogonal loading matrix
# (`PC1_Financial_Scale`, `PC2_Tenure_Stability`, `PC3_Leverage_Burden`), replace the raw
# collinear variables across all splits, and serialize the 6 core modeling Parquet splits
# to disk.
# ==============================================================================
# CELL 3: STRATIFIED PARTITIONING, LEAKAGE-FREE SCALING & PCA
# ==============================================================================
print("--- STEP 1: STRATIFIED PARTITION SPLIT (70% TRAIN / 15% VAL / 15% TEST) ---")
X = df_clean.drop(columns=["loan_status"])
y = df_clean["loan_status"]

X_train_raw, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(
    f"Split Sizes -> Train: {X_train_raw.shape[0]:,}, "
    f"Val: {X_val_raw.shape[0]:,}, Test: {X_test_raw.shape[0]:,}"
)

# 1. Collinear financial features targeted for PCA compression
pca_target_cols = [
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_emp_length",
]

# 2. Remaining continuous features scaled to preserve SVM RBF Euclidean geometry
other_num_cols = ["person_age", "loan_int_rate"]

scaler_pca = StandardScaler()
scaler_other = StandardScaler()

X_train_sc = X_train_raw.copy()
X_val_sc = X_val_raw.copy()
X_test_sc = X_test_raw.copy()

# Cast continuous columns to float prior to in-place scaled assignment
for df_split in [X_train_sc, X_val_sc, X_test_sc]:
    df_split[pca_target_cols + other_num_cols] = df_split[
        pca_target_cols + other_num_cols
    ].astype(float)

# Fit StandardScalers EXCLUSIVELY on X_train_raw to prevent data leakage
X_train_sc[pca_target_cols] = scaler_pca.fit_transform(X_train_raw[pca_target_cols])
X_val_sc[pca_target_cols] = scaler_pca.transform(X_val_raw[pca_target_cols])
X_test_sc[pca_target_cols] = scaler_pca.transform(X_test_raw[pca_target_cols])

X_train_sc[other_num_cols] = scaler_other.fit_transform(X_train_raw[other_num_cols])
X_val_sc[other_num_cols] = scaler_other.transform(X_val_raw[other_num_cols])
X_test_sc[other_num_cols] = scaler_other.transform(X_test_raw[other_num_cols])

# ------------------------------------------------------------------------------
# STEP 2: PRINCIPAL COMPONENT ANALYSIS (PCA) DECOMPOSITION
# ------------------------------------------------------------------------------
print("\n--- STEP 2: PRINCIPAL COMPONENT ANALYSIS (PCA) FIT ---")
pca = PCA(n_components=3, random_state=RANDOM_SEED)
pca.fit(X_train_sc[pca_target_cols])

exp_var = pca.explained_variance_ratio_
cum_var = np.cumsum(exp_var)
print(
    f"Explained Variance by Component: "
    f"PC1={exp_var[0]*100:.2f}%, PC2={exp_var[1]*100:.2f}%, PC3={exp_var[2]*100:.2f}%"
)
print(
    f"Cumulative Explained Variance (3 PCs): {cum_var[-1]*100:.2f}% (>80% Mandate Met)"
)
assert cum_var[-1] >= 0.80, (
    f"PCA explained variance below 80% threshold: {cum_var[-1]:.4f}"
)

# Render Scree Plot and Cumulative Explained Variance Curve
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.bar(
    range(1, 4),
    exp_var,
    color="skyblue",
    alpha=0.85,
    edgecolor="black",
    label="Individual Variance",
)
ax2 = ax1.twinx()
ax2.plot(
    range(1, 4),
    cum_var,
    color="darkred",
    marker="o",
    lw=2.5,
    label="Cumulative Variance",
)
ax2.axhline(0.80, color="gray", linestyle="--", lw=1.2, label="80% Threshold")

ax1.set_xlabel("Principal Components", fontweight="bold")
ax1.set_ylabel("Individual Explained Variance Ratio", fontweight="bold")
ax2.set_ylabel("Cumulative Explained Variance Ratio", fontweight="bold")
ax1.set_xticks(range(1, 4))
ax1.set_xticklabels([
    "PC1 (Financial Scale)",
    "PC2 (Tenure & Stability)",
    "PC3 (Leverage Burden)",
])
ax2.set_ylim(0.0, 1.05)
plt.title(
    f"PCA Scree Plot & Cumulative Explained Variance ({cum_var[-1]*100:.2f}% Total)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
plt.tight_layout()
plt.show()

# Display Feature Loading Matrix
pc_cols = ["PC1_Financial_Scale", "PC2_Tenure_Stability", "PC3_Leverage_Burden"]
loadings = pd.DataFrame(
    pca.components_.T,
    columns=pc_cols,
    index=pca_target_cols,
)
print("\n--- PCA Feature Loading Matrix ---")
display(loadings.round(4))

# Replace raw collinear features with the 3 orthogonal Principal Components
for df_target in [X_train_sc, X_val_sc, X_test_sc]:
    transformed = pca.transform(df_target[pca_target_cols])
    df_target["PC1_Financial_Scale"] = transformed[:, 0]
    df_target["PC2_Tenure_Stability"] = transformed[:, 1]
    df_target["PC3_Leverage_Burden"] = transformed[:, 2]
    df_target.drop(columns=pca_target_cols, inplace=True)

# Serialize the 6 core microdata modeling splits to Parquet
X_train_sc.to_parquet("X_train_processed.parquet")
X_val_sc.to_parquet("X_val_processed.parquet")
X_test_sc.to_parquet("X_test_processed.parquet")

y_train.to_frame().to_parquet("y_train.parquet")
y_val.to_frame().to_parquet("y_val.parquet")
y_test.to_frame().to_parquet("y_test.parquet")

print(
    "\nPCA transformation complete. 6 microdata modeling Parquet splits written to disk "
    "(ontario_macro_series.parquet persisted in Cell 1)."
)


# %%
# ==============================================================================
# 4. Phase 1 System Integrity Audit & Artifact Verification
# ==============================================================================
# We conclude Phase 1 by executing vectorized integrity assertions across all feature
# partitions, target vectors, and the Ontario macroeconomic sensitivity series for our
# PD_base modeling pipeline:
# 1. Verify zero NaN values and 100% finite numerical entries via
#    `np.isfinite(df.select_dtypes(include=np.number).to_numpy()).all()`.
# 2. Assert target default-rate homogeneity (~21.82%) across the 70/15/15 stratified
#    splits.
# 3. Confirm on-disk persistence of all 7 Parquet deliverables (6 microdata modeling
#    splits + 1 Ontario macro sensitivity series).
# ==============================================================================
# CELL 4: PHASE 1 SYSTEM INTEGRITY AUDIT & ARTIFACT VERIFICATION
# ==============================================================================
print("--- INITIATING PHASE 1 SYSTEM INTEGRITY & ARTIFACT AUDIT ---")

splits = [
    ("Train", X_train_sc, y_train),
    ("Validation", X_val_sc, y_val),
    ("Test", X_test_sc, y_test),
]

pop_default_rate = float(y.mean())
verification_rows = []

# 1. Vectorized Integrity & Target Homogeneity Assertions
for name, X_sub, y_sub in splits:
    nan_cnt_X = int(X_sub.isna().sum().sum())
    nan_cnt_y = int(y_sub.isna().sum())

    # Vectorized NumPy finite check (avoids Pandas FutureWarnings)
    all_finite_X = bool(
        np.isfinite(X_sub.select_dtypes(include=np.number).to_numpy()).all()
    )
    all_finite_y = bool(np.isfinite(y_sub.to_numpy()).all())

    split_default_rate = float(y_sub.mean())

    assert nan_cnt_X == 0, f"Audit failed: {nan_cnt_X} NaNs in {name} features!"
    assert nan_cnt_y == 0, f"Audit failed: {nan_cnt_y} NaNs in {name} target!"
    assert all_finite_X, f"Audit failed: Non-finite values in {name} features!"
    assert all_finite_y, f"Audit failed: Non-finite values in {name} target!"

    # Assert stratified target homogeneity (<0.1% deviation from population incidence)
    assert abs(split_default_rate - pop_default_rate) < 1e-3, (
        f"Stratification drift in {name}: {split_default_rate:.4f} vs"
        f" {pop_default_rate:.4f}"
    )

    verification_rows.append({
        "Dataset / Split": f"{name} Split (Microdata)",
        "Feature Matrix Shape": str(X_sub.shape),
        "Target Shape": str(y_sub.shape),
        "Null Count (X / y)": f"{nan_cnt_X} / {nan_cnt_y}",
        "All Finite (np.isfinite)": "PASS (100% Finite)",
        "Default Incidence / Macro Summary": f"{split_default_rate * 100:.3f}%",
    })

# Verify Ontario Macro Time-Series integrity (for Phase 3 PD_base sensitivity)
macro_nans = int(df_macro.isna().sum().sum())
macro_finite = bool(
    np.isfinite(df_macro.select_dtypes(include=np.number).to_numpy()).all()
)
assert macro_nans == 0, f"Audit failed: {macro_nans} NaNs in df_macro!"
assert macro_finite, "Audit failed: Non-finite values in df_macro!"

verification_rows.append({
    "Dataset / Split": "Ontario Macro Sensitivity Series (StatCan)",
    "Feature Matrix Shape": str(df_macro.shape),
    "Target Shape": "N/A (Macro Stress Inputs)",
    "Null Count (X / y)": f"{macro_nans} / 0",
    "All Finite (np.isfinite)": "PASS (100% Finite)",
    "Default Incidence / Macro Summary": (
        f"Unemp: {df_macro['Ontario_Unemployment_Rate'].mean():.2f}% | "
        f"CPI YoY: {df_macro['Ontario_CPI_YoY'].mean():.2f}%"
    ),
})

# 2. Verify Disk Persistence of all 7 Required Parquet Artifacts
saved_files = [
    "X_train_processed.parquet",
    "X_val_processed.parquet",
    "X_test_processed.parquet",
    "y_train.parquet",
    "y_val.parquet",
    "y_test.parquet",
    "ontario_macro_series.parquet",
]
for fname in saved_files:
    assert os.path.exists(fname) and os.path.getsize(fname) > 0, (
        f"Audit failed: Required artifact '{fname}' missing or empty on disk!"
    )

# 3. Render Comprehensive Verification Summary Table
verification_df = pd.DataFrame(verification_rows)
print("\n--- PHASE 1 BASELINE PD & MACRO SENSITIVITY ARTIFACT VERIFICATION ---")
display(verification_df)

print(
    "\nVerified 7/7 Parquet Artifacts Persisted on Disk (6 Microdata Splits + 1"
    " Macro Series):"
)
for fname in saved_files:
    print(f"  [OK] {fname:30s} ({os.path.getsize(fname):,d} bytes)")
