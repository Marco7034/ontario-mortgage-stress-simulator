# Power BI Desktop Build Guide — Ontario Mortgage Stress Simulator

This folder (`dashboard/`) is **100% self-contained**: every CSV file needed for Power BI is in `dashboard/data/`. No Python runtime or external database connection is required in Power BI—all macroeconomic stress calculations run **live in DAX** using pre-scored baseline probabilities and log-odds.

---

## 1. Data Files in `dashboard/data/`

| File | Rows | Purpose in Power BI |
| :--- | :---: | :--- |
| `powerbi_test_portfolio.csv` | `4,888` | Held-out test borrowers with human-readable attributes, Risk Quartiles, and pre-scored `PD_base` / `Logit_PD_base` for both **RBF-SVM** and **Logistic Regression**. |
| `powerbi_macro_series.csv` | `107` | Monthly Statistics Canada Ontario Unemployment Rate (`v2062815`) and YoY CPI Inflation (`v41690973`) from `2017-11-01` to `2026-09-01`. |
| `model_comparison_metrics.csv` | `9` | Train / Validation / Test performance metrics (`ROC-AUC`, `Brier Score`, `Log Loss`, `Accuracy`, `Precision`, `Recall`, `F1`) for Naive, Logistic Regression, and RBF-SVM. |
| `efficiency_ranking.csv` | `6` | Compute cost (`Fit_Time_s`), effective parameters, and Pareto Efficiency Score ranking across all tested models. |
| `calibration_deciles_comparison.csv` | `10` | 10-decile calibration reliability data (`Predicted PD` vs. `Observed Default %`) for RBF-SVM and Logistic Regression. |
| `credit_tier_stress_comparison.csv` | `11` | Pre-computed `+3.0 pp Unemp / +2.5 pp CPI` stress breakdown across Credit Grades (`A–G`) and Quartiles (`Q1–Q4`). |

---

## 2. Step-by-Step Import Instructions

1. Open **Power BI Desktop** (blank report).
2. Click **Home $\to$ Get data $\to$ Text/CSV** and import all 6 CSV files from `dashboard/data/`.
3. In **Power Query** (or **Data view**), verify column data types:
   - `powerbi_macro_series[Date]` $\to$ **Date**
   - `powerbi_test_portfolio[PD_base_SVM]`, `[PD_base_Logit]`, `[PD_stressed_Recession_SVM]`, `[PD_stressed_Recession_Logit]`, `[loan_percent_income]` $\to$ **Decimal Number** (format as **Percentage, 2 decimals** in the Report view).
   - `powerbi_test_portfolio[Logit_PD_base_SVM]`, `[Logit_PD_base_Logit]` $\to$ **Decimal Number**.
   - `powerbi_test_portfolio[person_income]`, `[loan_amnt]` $\to$ **Currency / Fixed Decimal**.
4. **Relationships:** Keep the tables independent (no relationships needed between `powerbi_test_portfolio` and `powerbi_macro_series`, since `powerbi_test_portfolio` is cross-sectional and receives macro shocks via the What-If sliders).

---

## 3. Create the Two Interactive What-If Macro Sliders

Go to **Modeling $\to$ New parameter $\to$ Numeric range** and create two parameters (check **"Add slicer to this page"**):

1. **Parameter 1 — Unemployment Shock:**
   - **Name:** `Unemp Shock`
   - **Data type:** `Decimal number`
   - **Minimum:** `-2.0`
   - **Maximum:** `5.0`
   - **Increment:** `0.5`
   - **Default:** `3.0` *(Ontario severe recession scenario)*

2. **Parameter 2 — Inflation Shock:**
   - **Name:** `CPI Shock`
   - **Data type:** `Decimal number`
   - **Minimum:** `-2.0`
   - **Maximum:** `5.0`
   - **Increment:** `0.5`
   - **Default:** `2.5` *(Ontario severe recession scenario)*

Power BI will automatically generate two disconnected parameter tables (`Unemp Shock` and `CPI Shock`) with measures `'Unemp Shock'[Unemp Shock Value]` and `'CPI Shock'[CPI Shock Value]`.

---

## 4. Paste These Exact DAX Measures

Create a new measure table (or place these measures inside `powerbi_test_portfolio` via **Modeling $\to$ New measure**).

### A. Primary Track — Calibrated RBF-SVM Live Stress Measures

```dax
Mean PD Base (SVM) =
AVERAGE('powerbi_test_portfolio'[PD_base_SVM])
```

```dax
Mean PD Stressed (SVM) =
VAR Beta1 = 0.213231
VAR Beta2 = 0.084338
VAR DeltaU = 'Unemp Shock'[Unemp Shock Value]
VAR DeltaC = 'CPI Shock'[CPI Shock Value]
RETURN
AVERAGEX(
    'powerbi_test_portfolio',
    VAR z = 'powerbi_test_portfolio'[Logit_PD_base_SVM] + (Beta1 * DeltaU) + (Beta2 * DeltaC)
    VAR z_safe = MIN(30, MAX(-30, z))
    RETURN DIVIDE(1, 1 + EXP(-z_safe))
)
```

```dax
PD Jump pp (SVM) =
([Mean PD Stressed (SVM)] - [Mean PD Base (SVM)]) * 100
```

```dax
Relative Surge (SVM) =
DIVIDE(
    [Mean PD Stressed (SVM)] - [Mean PD Base (SVM)],
    [Mean PD Base (SVM)]
)
```

```dax
Expected Defaults Stressed (SVM) =
VAR Beta1 = 0.213231
VAR Beta2 = 0.084338
VAR DeltaU = 'Unemp Shock'[Unemp Shock Value]
VAR DeltaC = 'CPI Shock'[CPI Shock Value]
RETURN
SUMX(
    'powerbi_test_portfolio',
    VAR z = 'powerbi_test_portfolio'[Logit_PD_base_SVM] + (Beta1 * DeltaU) + (Beta2 * DeltaC)
    VAR z_safe = MIN(30, MAX(-30, z))
    RETURN DIVIDE(1, 1 + EXP(-z_safe))
)
```

### B. Benchmark Track — Tuned Logistic Regression Live Stress Measures

```dax
Mean PD Base (Logit) =
AVERAGE('powerbi_test_portfolio'[PD_base_Logit])
```

```dax
Mean PD Stressed (Logit) =
VAR Beta1 = 0.213231
VAR Beta2 = 0.084338
VAR DeltaU = 'Unemp Shock'[Unemp Shock Value]
VAR DeltaC = 'CPI Shock'[CPI Shock Value]
RETURN
AVERAGEX(
    'powerbi_test_portfolio',
    VAR z = 'powerbi_test_portfolio'[Logit_PD_base_Logit] + (Beta1 * DeltaU) + (Beta2 * DeltaC)
    VAR z_safe = MIN(30, MAX(-30, z))
    RETURN DIVIDE(1, 1 + EXP(-z_safe))
)
```

```dax
PD Jump pp (Logit) =
([Mean PD Stressed (Logit)] - [Mean PD Base (Logit)]) * 100
```

### C. Portfolio Summary Measures

```dax
Borrower Count = COUNTROWS('powerbi_test_portfolio')
```

```dax
Actual Default Rate = AVERAGE('powerbi_test_portfolio'[actual_default])
```

```dax
Combined Macro Log-Odds Shift =
(0.213231 * 'Unemp Shock'[Unemp Shock Value]) + (0.084338 * 'CPI Shock'[CPI Shock Value])
```

---

## 5. Verification Checklist (Confirm Your DAX Matches Python)

Set the two sliders to **`Unemp Shock = 3.0`** and **`CPI Shock = 2.5`** (with no borrower filters selected) and confirm your KPI cards display these exact numbers:

| Measure | Expected Value |
| :--- | :---: |
| `Borrower Count` | `4,888` |
| `Actual Default Rate` | `21.83%` |
| `Mean PD Base (SVM)` | `22.20%` |
| `Mean PD Stressed (SVM)` | `30.37%` |
| `PD Jump pp (SVM)` | `+8.17` |
| `Relative Surge (SVM)` | `+36.80%` |
| `Mean PD Base (Logit)` | `22.07%` |
| `Mean PD Stressed (Logit)` | `32.30%` |
| `PD Jump pp (Logit)` | `+10.24` |

Set both sliders to **`0.0`** and verify `Mean PD Stressed (SVM)` equals `22.20%` (`PD Jump pp = 0.00`).
