# Ontario Mortgage Stress Simulator (ECON 354)

**Authors:** Alex Babiak, Marco Puiatti  
**Scope:** Ontario, Canada  

Interpretable machine learning pipeline for estimating individual borrower baseline default risk ($PD_{\text{base}}$) and testing portfolio sensitivity to Ontario macroeconomic shocks in unemployment and CPI inflation.

## Repository Structure

```text
ontario-mortgage-stress-simulator/
├── ECON354_Term_Project.ipynb          # Full end-to-end term project notebook
├── src/
│   └── phase1_data_pipeline.py         # Phase 1: Real-data ingestion, preprocessing, PCA & integrity audit
└── data/
    ├── credit_risk_dataset.csv         # Raw borrower microdata (32,581 records, 12 variables)
    ├── ontario_macro_series.parquet    # StatCan Ontario monthly unemployment & YoY CPI (107 months)
    ├── X_train_processed.parquet       # Training feature matrix (22,806 × 20)
    ├── X_val_processed.parquet         # Validation feature matrix (4,887 × 20)
    ├── X_test_processed.parquet        # Test feature matrix (4,888 × 20)
    ├── y_train.parquet                 # Training binary default target (22,806)
    ├── y_val.parquet                   # Validation binary default target (4,887)
    └── y_test.parquet                  # Test binary default target (4,888)
```

## Running Phase 1

```bash
python3 src/phase1_data_pipeline.py
```
