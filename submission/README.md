# ECON 354 Term Project — Final Submission Package

**Title:** Estimating Baseline Borrower Default Risk ($PD_{\text{base}}$) & Ontario Macroeconomic Stress Sensitivity  
**Authors:** Alex Babiak, Marco Puiatti  

## Contents

```text
submission/
├── ECON354_Final_Project.ipynb      # Single end-to-end executed notebook covering the full project
├── README.md                        # This guide
├── requirements.txt                 # Python dependencies (`pip install -r requirements.txt`)
└── data/
    ├── credit_risk_dataset.csv      # Simulated credit-bureau borrower microdata proxy (32,581 records)
    ├── ontario_macro_series.parquet # Real Statistics Canada Ontario Unemployment & CPI YoY (107 months, 2017–2026)
    ├── X_train_processed.parquet    # Leakage-free Train features (22,806 × 20)
    ├── X_val_processed.parquet      # Leakage-free Validation features (4,887 × 20)
    ├── X_test_processed.parquet     # Leakage-free Test features (4,888 × 20)
    ├── y_train.parquet              # Train default indicator (22,806)
    ├── y_val.parquet                # Validation default indicator (4,887)
    ├── y_test.parquet               # Test default indicator (4,888)
    ├── pd_test.npy                  # Calibrated RBF-SVM baseline test PDs (4,888)
    ├── pd_test_lr.npy               # Tuned Logistic Regression baseline test PDs (4,888)
    └── pd_stressed.npy              # Ontario recession-stressed test PDs (+3.0 pp Unemp, +2.5 pp CPI YoY)
```

## How to Run

1. Open `ECON354_Final_Project.ipynb` in Jupyter Notebook, JupyterLab, or VS Code (with `submission/` as the working directory).
2. All outputs, tables, and figures are already saved in the notebook. To re-run from scratch, click **Run All** (requires `pandas`, `numpy`, `scikit-learn`, `pyarrow`, `matplotlib`, `seaborn`, `requests`, and `joblib`).
3. If run offline, Cell 1 automatically falls back to the cached files inside `data/` so the entire notebook runs end-to-end without requiring internet access.
