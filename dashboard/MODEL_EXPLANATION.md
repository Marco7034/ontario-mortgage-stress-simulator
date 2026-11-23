# Model & Methodology Explanation (Dashboard Reference)

This document explains the data, preprocessing, classification models, and macroeconomic stress-testing engine powering the Power BI dashboard.

---

## 1. Datasets Used

1. **Borrower Microdata (`credit_risk_dataset.csv` — 32,581 records, 12 variables):**
   - An externally published simulated credit-bureau dataset (`laotse/credit-risk-dataset`) used as a proxy for Canadian household borrowers, since actual Canadian bank and CMHC loan-level mortgage files are restricted under PIPEDA privacy laws and proprietary rules.
   - Partitioned via stratified random sampling (`random_state=42`) into **70% Train (`22,806`)**, **15% Validation (`4,887`)**, and **15% Test (`4,888`)**, preserving the exact `~21.82%` population default rate across every split.
   - The Power BI dashboard uses the **4,888 held-out Test borrowers** (`powerbi_test_portfolio.csv`) so every metric reflects true out-of-sample performance.

2. **Ontario Macroeconomic Series (`powerbi_macro_series.csv` — 107 months, 2017–2026):**
   - **100% official government data** queried programmatically from Statistics Canada's Web Data Service (WDS) API:
     - **Ontario Monthly Unemployment Rate** (Vector `v2062815`, Table `14-10-0287-01`; mean `6.57%`, historical SD = `1.6414 pp`).
     - **Ontario All-Items CPI Year-over-Year Inflation** (derived from Vector `v41690973`, Table `18-10-0004-01`; mean `2.90%`, historical SD = `1.7786 pp`).

---

## 2. Phase 1: Leakage-Free Preprocessing & PCA

All cleaning statistics are fitted **strictly on the 70% training split** before transforming validation and test borrowers:
- **Median Imputation:** Missing `person_emp_length` (`4.0` years) and `loan_int_rate` (`10.99%`) filled with training medians.
- **Tukey $1.5 \times \text{IQR}$ Winsorization:** Extreme high outliers in `person_income` (`\$141,300.38` cap) and `loan_amnt` (`\$23,125.00` cap) are capped so extreme values do not distort distance calculations.
- **Standardization & PCA:** The 5 correlated financial variables (`person_income`, `loan_amnt`, `loan_percent_income`, `cb_person_cred_hist_length`, `person_emp_length`) are standardized and compressed into **3 orthogonal Principal Components** explaining **`81.90%`** of their combined variance:
  - `PC1_Financial_Scale` (`32.20%`): Overall loan size and borrowing footprint.
  - `PC2_Tenure_Stability` (`29.74%`): Higher income + longer employment/credit tenure + lower debt-to-income ratio.
  - `PC3_Leverage_Burden` (`19.96%`): Longer credit/employment history paired with lower income and smaller loans.
- Together with `person_age`, `loan_int_rate`, and 15 one-hot indicators (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`), each borrower enters the models as a **20-feature vector**.

---

## 3. Stage 1: Baseline Default Classification ($PD_{\text{base}}$)

The project pairs two complementary models evaluated on the same `4,888` test borrowers:

### A. Tuned Logistic Regression (`L2, C = 10.0`) — Institutional Benchmark
- **How it works:** Computes a linear weighted sum of the 20 features in log-odds space and maps it through the logistic sigmoid function.
- **Test Performance:** `ROC-AUC = 0.8700`, `Brier Score = 0.1032`, `Accuracy = 86.54%`, `F1 = 0.6420`.
- **Why it is included:** Ranks **#1 in Efficiency (`0.1719`)**, retaining `96.18%` of peak AUC with only **21 transparent weights** in **`0.13 seconds`** (`1.0×` compute). At institutional scale (millions of bank mortgages), Logistic Regression provides speed, stability, and direct regulatory explainability.

### B. Calibrated RBF-SVM (`C = 10.0, gamma = 0.05`, 3-Fold Platt Sigmoid) — Primary Non-Linear Model
- **How it works:** Uses a Gaussian Radial Basis Function (RBF) kernel to measure similarity to boundary-defining "support vector" borrowers, forming curved, non-linear boundaries that capture compounding interactions across leverage, interest rates, and stability. Wrapping the SVM in **3-fold Platt sigmoid calibration** converts raw geometric margins into true calibrated probabilities $PD_{\text{base}} \in (0, 1)$.
- **Test Performance:** `ROC-AUC = 0.9013` (`+0.0313` over Logit), `Brier Score = 0.0747` (`-27.6%` lower error), `Accuracy = 90.86%` (`-32.1%` fewer wrong predictions), `F1 = 0.7624`, and `1.59 pp` mean 10-decile calibration error (vs. `2.51 pp` for Logit).
- **Why it is included:** Captures **`99.64%`** of the peak AUC achieved by a slower Cubic Polynomial SVM (`0.9046`) while training **`3.1×` faster** (`8.15s` vs. `25.58s`). For our regional stress-testing analysis, RBF-SVM provides the sharpest individual $PD_{\text{base}}$ calibration, especially in the high-risk and ultra-prime tails.

---

## 4. Stage 2: Dynamic Ontario Macroeconomic Stress Engine ($PD_{\text{stressed}}$)

Because cross-sectional borrower data lacks time stamps, the model applies macroeconomic shocks dynamically in **log-odds (logit) space**:

1. **Volatility-Scaled Elasticities:** Using Statistics Canada's historical Ontario standard deviations ($\sigma_{\text{Unemp}} = 1.6414\text{ pp}$, $\sigma_{\text{CPI}} = 1.7786\text{ pp}$) and calibrated 1-SD shock weights (`0.35` for unemployment; `0.15` for inflation, encoding that job loss impacts default roughly `2.5×` more strongly than inflation per percentage point):
   $$\beta_1 = \frac{0.35}{1.6414} = \mathbf{0.2132\text{ per +1 pp Unemp}}, \qquad \beta_2 = \frac{0.15}{1.7786} = \mathbf{0.0843\text{ per +1 pp CPI YoY}}$$
2. **Logit Shock Equation (Computed Live in Power BI DAX):**
   $$\text{logit}(PD_{\text{stressed}, i}) = \ln\!\left(\frac{PD_{\text{base}, i}}{1 - PD_{\text{base}, i}}\right) + \beta_1 \Delta\text{Unemp} + \beta_2 \Delta\text{CPI}$$
   $$PD_{\text{stressed}, i} = \frac{1}{1 + \exp\!\left(-\text{logit}(PD_{\text{stressed}, i})\right)}$$
3. **Key Economic Finding (`+3.0 pp` Unemp, `+2.5 pp` CPI YoY):**
   - Portfolio mean default probability rises from **`22.20%` to `30.37%`** (`+8.17 pp`, or `+36.80%` relative surge).
   - **Near-Prime borrowers (`Q3`, baseline PD `8.79%–17.01%`)** suffer the largest absolute jump (**`+11.57 pp`**, from `11.26%` to `22.83%`) because they sit on the financial margin near the steepest slope of the logistic curve.
   - **Ultra-Prime borrowers (`Q1`)** stay low risk (`2.53%` $\to$ `5.68%`, `+3.15 pp`), while **High-Distress / Grade G** borrowers are already near default saturation (`93.80%` $\to$ `96.76%`).
