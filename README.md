# Ontario Mortgage Stress Simulator (ECON 354)

**Authors:** Alex Babiak, Marco Puiatti  
**Scope:** Ontario, Canada  

Interpretable machine learning pipeline for estimating individual borrower baseline default risk ($PD_{\text{base}}$) and testing portfolio sensitivity to Ontario macroeconomic shocks in unemployment and CPI inflation.

- **Model Results & Benchmark Comparison:** See [results/svm_results.md](results/svm_results.md) for side-by-side Test, Validation, and Train metrics comparing Calibrated RBF-SVM (`C=10.0, gamma=0.05`) against Tuned Logistic Regression (`L2, C=10.0`) and the Naive Base-Rate floor.

## Repository Structure

```text
ontario-mortgage-stress-simulator/
├── ECON354_Term_Project.ipynb                  # Full end-to-end term project notebook (Phase 1 & Phase 2)
├── src/
│   ├── phase1_data_pipeline.py                 # Phase 1: Real-data ingestion, preprocessing, PCA & integrity audit
│   ├── phase2_modeling_and_stress.py           # Phase 2: Logistic benchmark, SVM tuning/calibration, macro stress & diagnostics
│   ├── train_and_export_artifacts.py           # Exports fitted preprocessor, models & macro config for dashboard use
│   └── stress_engine.py                        # Zero-side-effect inference & macro stress API for interactive dashboards
├── models/
│   ├── preprocessor.joblib                     # Fitted medians, Tukey IQR bounds, StandardScalers & 3-PC PCA
│   ├── logistic_regression.joblib              # Tuned L2 Logistic Regression (C=10.0)
│   ├── calibrated_rbf_svm.joblib               # 3-Fold Platt-Calibrated RBF-SVM (C=10.0, gamma=0.05)
│   └── macro_config.json                       # StatCan calibrated elasticities (Beta_1, Beta_2) & metadata
├── dashboard/
│   ├── POWERBI_BUILD_GUIDE.md                  # Step-by-step Power BI Desktop setup, What-If sliders & DAX
│   ├── DASHBOARD_LAYOUT_SPEC.md                # Visual layout & wireframe specification (2 pages)
│   ├── MODEL_EXPLANATION.md                    # Plain-English & technical model guide for dashboard/report
│   └── data/                                   # Self-contained CSVs for Power BI (test portfolio, macro, metrics)
├── presentation_graphics/                      # Slide-ordered high-DPI white-background PNGs for PowerPoint
├── results/
│   ├── svm_results.md                          # Comprehensive SVM vs. Logistic Regression benchmark report
│   ├── model_comparison_metrics.csv            # Train/Val/Test classification & calibration metrics
│   ├── svm_validation_grid_search.csv          # 16-point RBF-SVM C × gamma grid search metrics
│   ├── kernel_family_comparison.csv            # Linear, Quadratic, Cubic, Sigmoid, RBF & Logit comparison
│   ├── efficiency_ranking.csv                  # Compute cost, parameter count & Pareto efficiency ranking
│   ├── calibration_deciles_comparison.csv      # 10-decile calibration reliability comparison
│   ├── credit_tier_stress_comparison.csv       # Credit grade & risk quartile macro stress breakdown
│   ├── kernel_family_comparison.png            # Bar chart comparing all 6 kernels/models on Test set
│   ├── kernel_boundary_grid.png                # 2×2 white-background decision boundary grid
│   ├── efficiency_pareto_ranking.png           # Compute cost vs. Test ROC-AUC Pareto frontier & ranking
│   ├── decision_boundary_comparison.png        # Logistic Regression vs. RBF-SVM boundary comparison
│   └── model_classification_comparison.png     # Test metrics & classification error reduction charts
└── data/
    ├── credit_risk_dataset.csv                 # Raw borrower microdata (32,581 records, 12 variables)
    ├── ontario_macro_series.parquet            # StatCan Ontario monthly unemployment & YoY CPI (107 months)
    ├── X_train_processed.parquet               # Training feature matrix (22,806 × 20)
    ├── X_val_processed.parquet                 # Validation feature matrix (4,887 × 20)
    ├── X_test_processed.parquet                # Test feature matrix (4,888 × 20)
    ├── y_train.parquet                         # Training binary default target (22,806)
    ├── y_val.parquet                           # Validation binary default target (4,887)
    ├── y_test.parquet                          # Test binary default target (4,888)
    ├── pd_test.npy                             # Calibrated baseline test PD estimates (RBF-SVM, 4,888)
    ├── pd_test_lr.npy                          # Baseline test PD estimates (Logistic Regression, 4,888)
    └── pd_stressed.npy                         # Macro-stressed test PD estimates (+3.0 pp Unemp, +2.5 pp CPI YoY)
```

## Running the Pipeline

```bash
python3 src/phase1_data_pipeline.py
python3 src/phase2_modeling_and_stress.py
python3 src/train_and_export_artifacts.py
```

## Interactive Dashboard Integration (`src/stress_engine.py`)

```python
from src.stress_engine import load_dashboard_bundle, simulate_borrower_stress, simulate_portfolio_stress

bundle = load_dashboard_bundle()  # cache once in Streamlit/Dash

# 1. Score a single custom borrower + macro shock
borrower_result = simulate_borrower_stress(
    {
        "person_age": 31, "person_income": 75000, "person_home_ownership": "RENT",
        "person_emp_length": 4.0, "loan_intent": "PERSONAL", "loan_grade": "C",
        "loan_amnt": 18000, "loan_int_rate": 12.5,
        "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 6,
    },
    delta_unemp_pp=3.0, delta_cpi_pp=2.5, bundle=bundle,
)

# 2. Vectorized 4,888-borrower portfolio stress test
portfolio_result = simulate_portfolio_stress(delta_unemp_pp=3.0, delta_cpi_pp=2.5, bundle=bundle, model="svm")
```
