"""
Dashboard-ready inference and Ontario macroeconomic stress-testing engine.

Provides zero-side-effect functions for an interactive UI (e.g. Streamlit, Dash,
or FastAPI) to:
1. Load pre-trained models, preprocessors, and macro configuration once.
2. Transform raw applicant inputs into the 20-feature PCA-orthogonalized space.
3. Score baseline default probabilities (PD_base) with both Calibrated RBF-SVM and
   Tuned Logistic Regression.
4. Dynamically apply Ontario unemployment and CPI inflation shocks in log-odds space
   for either a single borrower or the full 4,888-borrower test portfolio.
"""

import json
import os
from typing import Any, Dict, Union

import joblib
import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_dashboard_bundle(base_dir: str = DEFAULT_BASE_DIR) -> Dict[str, Any]:
    """Load all persisted preprocessors, models, macro parameters, and test arrays."""
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    with open(os.path.join(models_dir, "macro_config.json"), "r", encoding="utf-8") as f:
        macro_config = json.load(f)

    return {
        "preprocessor": joblib.load(os.path.join(models_dir, "preprocessor.joblib")),
        "logistic_regression": joblib.load(
            os.path.join(models_dir, "logistic_regression.joblib")
        ),
        "calibrated_rbf_svm": joblib.load(
            os.path.join(models_dir, "calibrated_rbf_svm.joblib")
        ),
        "macro_config": macro_config,
        "df_macro": pd.read_parquet(os.path.join(data_dir, "ontario_macro_series.parquet")),
        "X_test": pd.read_parquet(os.path.join(data_dir, "X_test_processed.parquet")),
        "y_test": pd.read_parquet(os.path.join(data_dir, "y_test.parquet"))["loan_status"],
        "pd_test_svm": np.load(os.path.join(data_dir, "pd_test.npy")),
        "pd_test_lr": np.load(os.path.join(data_dir, "pd_test_lr.npy")),
    }


def preprocess_borrower(
    raw_input: Union[Dict[str, Any], pd.DataFrame],
    preprocessor: Dict[str, Any],
) -> pd.DataFrame:
    """Transform raw borrower attributes into the 20-feature model input matrix."""
    df = pd.DataFrame([raw_input]) if isinstance(raw_input, dict) else raw_input.copy()

    # 1. Training-median imputation
    for col, med_val in preprocessor["train_medians"].items():
        if col not in df.columns:
            df[col] = med_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med_val)

    # 2. Tukey 1.5x IQR Winsorization & leverage ratio recalculation
    for col, bounds in preprocessor["iqr_bounds"].items():
        df[col] = np.clip(df[col].astype(float), bounds["lower"], bounds["upper"])
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]

    # 3. Standardize continuous variables & project 5 financials onto 3 PCA factors
    pca_cols = preprocessor["pca_target_cols"]
    other_cols = preprocessor["other_num_cols"]
    df[pca_cols] = preprocessor["scaler_pca"].transform(df[pca_cols].astype(float))
    df[other_cols] = preprocessor["scaler_other"].transform(df[other_cols].astype(float))

    pc_vals = preprocessor["pca"].transform(df[pca_cols])
    for idx, pc_name in enumerate(preprocessor["pc_names"]):
        df[pc_name] = pc_vals[:, idx]
    df.drop(columns=pca_cols, inplace=True)

    # 4. One-hot encode categoricals and align strictly to training feature columns
    df = pd.get_dummies(
        df, columns=preprocessor["categorical_cols"], drop_first=False, dtype=int
    )
    feature_cols = preprocessor["feature_columns"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    return df[feature_cols].astype(float)


def apply_macro_shock(
    base_pds: Union[float, np.ndarray],
    delta_unemp_pp: float,
    delta_cpi_pp: float,
    beta_unemp: float,
    beta_cpi: float,
) -> Union[float, np.ndarray]:
    """Shift baseline PDs in log-odds space using StatCan-calibrated elasticities."""
    eps = 1e-12
    is_scalar = np.ndim(base_pds) == 0
    p_arr = np.asarray(base_pds, dtype=float)
    safe = np.clip(p_arr, eps, 1.0 - eps)
    logits = np.log(safe / (1.0 - safe))
    stressed_logits = logits + (beta_unemp * delta_unemp_pp) + (beta_cpi * delta_cpi_pp)
    stressed = 1.0 / (1.0 + np.exp(-np.clip(stressed_logits, -30.0, 30.0)))
    return float(stressed) if is_scalar else stressed


def simulate_borrower_stress(
    raw_borrower: Dict[str, Any],
    delta_unemp_pp: float,
    delta_cpi_pp: float,
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """End-to-end single-borrower scoring + macro stress simulation for UI forms."""
    X_feat = preprocess_borrower(raw_borrower, bundle["preprocessor"])
    pd_base_svm = float(bundle["calibrated_rbf_svm"].predict_proba(X_feat)[0, 1])
    pd_base_lr = float(bundle["logistic_regression"].predict_proba(X_feat)[0, 1])

    cfg = bundle["macro_config"]
    pd_stressed_svm = float(
        apply_macro_shock(
            pd_base_svm,
            delta_unemp_pp,
            delta_cpi_pp,
            cfg["beta_unemployment"],
            cfg["beta_inflation"],
        )
    )
    pd_stressed_lr = float(
        apply_macro_shock(
            pd_base_lr,
            delta_unemp_pp,
            delta_cpi_pp,
            cfg["beta_unemployment"],
            cfg["beta_inflation"],
        )
    )

    return {
        "pd_base_svm": pd_base_svm,
        "pd_stressed_svm": pd_stressed_svm,
        "jump_pp_svm": (pd_stressed_svm - pd_base_svm) * 100.0,
        "pd_base_lr": pd_base_lr,
        "pd_stressed_lr": pd_stressed_lr,
        "jump_pp_lr": (pd_stressed_lr - pd_base_lr) * 100.0,
        "pca_scores": {
            pc: float(X_feat[pc].iloc[0]) for pc in bundle["preprocessor"]["pc_names"]
        },
    }


def simulate_portfolio_stress(
    delta_unemp_pp: float,
    delta_cpi_pp: float,
    bundle: Dict[str, Any],
    model: str = "svm",
) -> Dict[str, Any]:
    """Vectorized portfolio macro stress simulation across all 4,888 test borrowers."""
    pd_base = bundle["pd_test_svm"] if model == "svm" else bundle["pd_test_lr"]
    cfg = bundle["macro_config"]
    pd_stressed = apply_macro_shock(
        pd_base,
        delta_unemp_pp,
        delta_cpi_pp,
        cfg["beta_unemployment"],
        cfg["beta_inflation"],
    )

    quartiles = pd.qcut(
        bundle["pd_test_svm"],
        q=4,
        labels=["Q1: Ultra-Prime", "Q2: Prime", "Q3: Near-Prime", "Q4: High Distress"],
    )
    tier_df = pd.DataFrame({
        "Quartile": quartiles,
        "PD_base": pd_base * 100.0,
        "PD_stressed": pd_stressed * 100.0,
    })
    summary = (
        tier_df.groupby("Quartile", observed=False)
        .agg(
            Borrowers=("PD_base", "count"),
            Mean_PD_Base_Pct=("PD_base", "mean"),
            Mean_PD_Stressed_Pct=("PD_stressed", "mean"),
        )
        .reset_index()
    )
    summary["Jump_pp"] = summary["Mean_PD_Stressed_Pct"] - summary["Mean_PD_Base_Pct"]

    mean_base = float(pd_base.mean() * 100.0)
    mean_stressed = float(pd_stressed.mean() * 100.0)
    return {
        "pd_base": pd_base,
        "pd_stressed": pd_stressed,
        "mean_pd_base_pct": mean_base,
        "mean_pd_stressed_pct": mean_stressed,
        "abs_jump_pp": mean_stressed - mean_base,
        "rel_surge_pct": ((mean_stressed - mean_base) / mean_base) * 100.0,
        "quartile_breakdown": summary,
    }
