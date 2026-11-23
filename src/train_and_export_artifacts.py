"""
Clean training and artifact-export script for the interactive dashboard.

Fits the exact Phase 1 preprocessing pipeline (training-only medians, Tukey 1.5x IQR
caps, StandardScalers, and 3-component PCA) and Phase 2 models (Tuned Logistic
Regression C=10.0 and 3-Fold Platt-Calibrated RBF-SVM C=10.0, gamma=0.05), then
persists lightweight, dashboard-ready artifacts into `models/` and `data/`.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_SEED = 42
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    # ------------------------------------------------------------------
    # 1. Fit Phase 1 Preprocessor on Raw Microdata (Strict Train-Only)
    # ------------------------------------------------------------------
    raw_csv = os.path.join(DATA_DIR, "credit_risk_dataset.csv")
    df_raw = pd.read_csv(raw_csv)

    train_idx, _ = train_test_split(
        df_raw.index,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df_raw["loan_status"],
    )

    train_medians = {
        "person_emp_length": float(df_raw.loc[train_idx, "person_emp_length"].median()),
        "loan_int_rate": float(df_raw.loc[train_idx, "loan_int_rate"].median()),
    }

    df_clean = df_raw.copy()
    df_clean["person_emp_length"] = df_clean["person_emp_length"].fillna(
        train_medians["person_emp_length"]
    )
    df_clean["loan_int_rate"] = df_clean["loan_int_rate"].fillna(
        train_medians["loan_int_rate"]
    )

    iqr_bounds = {}
    for col in ["person_income", "loan_amnt"]:
        q25 = float(df_clean.loc[train_idx, col].quantile(0.25))
        q75 = float(df_clean.loc[train_idx, col].quantile(0.75))
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        iqr_bounds[col] = {"lower": lower, "upper": upper}
        df_clean[col] = np.clip(df_clean[col], lower, upper)

    df_clean["loan_percent_income"] = df_clean["loan_amnt"] / df_clean["person_income"]

    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    df_encoded = pd.get_dummies(
        df_clean, columns=categorical_cols, drop_first=True, dtype=int
    )

    X = df_encoded.drop(columns=["loan_status"])
    y = df_encoded["loan_status"]

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    pca_target_cols = [
        "person_income",
        "loan_amnt",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "person_emp_length",
    ]
    other_num_cols = ["person_age", "loan_int_rate"]

    scaler_pca = StandardScaler().fit(X_train_raw[pca_target_cols].astype(float))
    scaler_other = StandardScaler().fit(X_train_raw[other_num_cols].astype(float))

    X_train_sc = X_train_raw.copy()
    X_train_sc[pca_target_cols] = scaler_pca.transform(X_train_raw[pca_target_cols].astype(float))
    X_train_sc[other_num_cols] = scaler_other.transform(X_train_raw[other_num_cols].astype(float))

    pca = PCA(n_components=3, random_state=RANDOM_SEED).fit(X_train_sc[pca_target_cols])
    pc_names = ["PC1_Financial_Scale", "PC2_Tenure_Stability", "PC3_Leverage_Burden"]

    pc_vals = pca.transform(X_train_sc[pca_target_cols])
    for idx, col in enumerate(pc_names):
        X_train_sc[col] = pc_vals[:, idx]
    X_train_sc.drop(columns=pca_target_cols, inplace=True)

    # Verify exact column alignment with persisted Phase 1 Parquet
    X_train_disk = pd.read_parquet(os.path.join(DATA_DIR, "X_train_processed.parquet"))
    assert list(X_train_sc.columns) == list(X_train_disk.columns), "Feature column mismatch!"
    assert np.allclose(X_train_sc.to_numpy(), X_train_disk.to_numpy(), atol=1e-10), (
        "Preprocessed matrix mismatch vs X_train_processed.parquet!"
    )

    preprocessor = {
        "train_medians": train_medians,
        "iqr_bounds": iqr_bounds,
        "categorical_cols": categorical_cols,
        "pca_target_cols": pca_target_cols,
        "other_num_cols": other_num_cols,
        "pc_names": pc_names,
        "scaler_pca": scaler_pca,
        "scaler_other": scaler_other,
        "pca": pca,
        "feature_columns": list(X_train_sc.columns),
    }
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"), compress=3)

    # ------------------------------------------------------------------
    # 2. Fit & Export Phase 2 Models (Logistic Regression & Calibrated RBF-SVM)
    # ------------------------------------------------------------------
    X_test_disk = pd.read_parquet(os.path.join(DATA_DIR, "X_test_processed.parquet"))
    y_train_disk = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet"))["loan_status"]

    lr_model = LogisticRegression(C=10.0, max_iter=1000, random_state=RANDOM_SEED)
    lr_model.fit(X_train_disk, y_train_disk)
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"), compress=3)

    pd_test_lr = lr_model.predict_proba(X_test_disk)[:, 1]
    np.save(os.path.join(DATA_DIR, "pd_test_lr.npy"), pd_test_lr)

    opt_svm = SVC(kernel="rbf", C=10.0, gamma=0.05, cache_size=1000, random_state=RANDOM_SEED)
    cal_svm = CalibratedClassifierCV(estimator=opt_svm, method="sigmoid", cv=3, n_jobs=-1)
    cal_svm.fit(X_train_disk, y_train_disk)
    joblib.dump(cal_svm, os.path.join(MODELS_DIR, "calibrated_rbf_svm.joblib"), compress=3)

    pd_test_svm = cal_svm.predict_proba(X_test_disk)[:, 1]
    pd_test_saved = np.load(os.path.join(DATA_DIR, "pd_test.npy"))
    assert np.allclose(pd_test_svm, pd_test_saved, atol=1e-10), (
        "Calibrated RBF-SVM output mismatch vs data/pd_test.npy!"
    )

    # ------------------------------------------------------------------
    # 3. Export StatCan Macro Configuration JSON
    # ------------------------------------------------------------------
    df_macro = pd.read_parquet(os.path.join(DATA_DIR, "ontario_macro_series.parquet"))
    hist_unemp_sd = float(df_macro["Ontario_Unemployment_Rate"].std())
    hist_inf_sd = float(df_macro["Ontario_CPI_YoY"].std())
    macro_config = {
        "beta_unemployment": float(0.35 / max(hist_unemp_sd, 0.1)),
        "beta_inflation": float(0.15 / max(hist_inf_sd, 0.1)),
        "hist_unemp_sd": hist_unemp_sd,
        "hist_inf_sd": hist_inf_sd,
        "macro_obs_count": int(len(df_macro)),
        "macro_date_min": str(df_macro["Date"].min().date()),
        "macro_date_max": str(df_macro["Date"].max().date()),
        "latest_unemp_rate": float(df_macro["Ontario_Unemployment_Rate"].iloc[-1]),
        "latest_cpi_yoy": float(df_macro["Ontario_CPI_YoY"].iloc[-1]),
        "test_portfolio_size": int(len(X_test_disk)),
    }
    with open(os.path.join(MODELS_DIR, "macro_config.json"), "w", encoding="utf-8") as f:
        json.dump(macro_config, f, indent=2)

    print("Exported dashboard artifacts to models/:")
    for fname in sorted(os.listdir(MODELS_DIR)):
        fpath = os.path.join(MODELS_DIR, fname)
        print(f"  [OK] {fname:30s} ({os.path.getsize(fpath):,d} bytes)")


if __name__ == "__main__":
    main()
