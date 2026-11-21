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
│   └── phase2_modeling_and_stress.py           # Phase 2: Logistic benchmark, SVM tuning/calibration, macro stress & diagnostics
├── results/
│   ├── svm_results.md                          # Comprehensive SVM vs. Logistic Regression benchmark report
│   ├── model_comparison_metrics.csv            # Train/Val/Test classification & calibration metrics
│   ├── svm_validation_grid_search.csv          # 16-point RBF-SVM C × gamma grid search metrics
│   ├── kernel_family_comparison.csv            # Linear, Quadratic, Cubic, Sigmoid, RBF & Logit comparison
│   ├── calibration_deciles_comparison.csv      # 10-decile calibration reliability comparison
│   ├── credit_tier_stress_comparison.csv       # Credit grade & risk quartile macro stress breakdown
│   ├── kernel_family_comparison.png            # Bar chart comparing all 6 kernels/models on Test set
│   ├── kernel_boundary_grid.png                # 2×2 white-background decision boundary grid
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
    ├── pd_test.npy                             # Calibrated baseline test PD estimates (4,888)
    └── pd_stressed.npy                         # Macro-stressed test PD estimates (+3.0 pp Unemp, +2.5 pp CPI YoY)
```

## Running the Pipeline

```bash
python3 src/phase1_data_pipeline.py
python3 src/phase2_modeling_and_stress.py
```
