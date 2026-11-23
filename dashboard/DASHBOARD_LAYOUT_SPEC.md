# Power BI Dashboard Layout & Visual Specification

*(Note: Keep SVM decision-boundary plots out of the Power BI dashboard—those belong in your PowerPoint slides. The dashboard focuses on interactive macroeconomic stress simulation, borrower segment breakdowns, and clean model-performance comparisons.)*

---

## Color Palette & Styling

- **Canvas Background:** `#F8FAFC` (very soft off-white) with `#FFFFFF` white cards and thin `#E2E8F0` borders.
- **Primary (RBF-SVM / Baseline):** `#1F4E79` (Deep Navy)
- **Stressed / Comparison (Macro Shock / Logistic Regression):** `#D95F02` (Warm Burnt Orange)
- **Accent / Positive / Neutral:** `#0B6E4F` (Forest Green) and `#64748B` (Slate Gray)
- **Font:** Segoe UI (Power BI default) — Bold titles (`12–14 pt`), clean data labels (`9–10 pt`), zero 3D effects.

---

## Page 1: Ontario Macroeconomic Stress Simulator (Main Interactive Page)

```text
+---------------------------------------------------------------------------------------------------+
| TITLE: Ontario Household Credit Stress Simulator (Test Portfolio, N = 4,888)                      |
+---------------------------------------------------------------------------------------------------+
| [Slider: Unemp Shock (pp): +3.0] | [Slider: CPI Shock (pp): +2.5] | [Grade] | [Ownership] | [Inc] |
+---------------------------------------------------------------------------------------------------+
|  [KPI 1]                 [KPI 2]                  [KPI 3]                  [KPI 4]                |
|  Baseline Mean PD        Stressed Mean PD         Absolute Risk Jump       Expected Defaults      |
|  22.20%                  30.37%                   +8.17 pp (+36.8%)        1,484 (vs 1,085 base)  |
+---------------------------------------------------------------------------------------------------+
|                                                |                                                  |
|  CHART 1 (Clustered Column Chart)              |  CHART 2 (Clustered Column Chart)                |
|  Baseline vs. Stressed PD by Risk Quartile     |  Baseline vs. Stressed PD by Credit Grade (A-G)  |
|  X: Risk_Quartile (Q1 -> Q4)                   |  X: loan_grade (A -> G)                          |
|  Y: [Mean PD Base (SVM)], [Mean PD Stressed]   |  Y: [Mean PD Base (SVM)], [Mean PD Stressed]     |
|  (Highlights Q3 Near-Prime +11.57 pp jump)     |  (Highlights Grade C/E/F risk surge)             |
|                                                |                                                  |
+------------------------------------------------+--------------------------------------------------+
|                                                                                                   |
|  CHART 3 (Dual-Line Time-Series Chart — StatCan Context, 2017–2026)                               |
|  X: powerbi_macro_series[Date]                                                                    |
|  Y: Ontario_Unemployment_Rate (%) [Navy]  &  Ontario_CPI_YoY (%) [Orange]                         |
|  Subtitle: "Statistics Canada WDS API (v2062815 & v41690973) used to calibrate Beta_1 & Beta_2"   |
+---------------------------------------------------------------------------------------------------+
```

### Page 1 Visual Details
1. **Top Control Bar (Slicers):**
   - `Unemp Shock` slider (`-2.0` to `+5.0 pp`) & `CPI Shock` slider (`-2.0` to `+5.0 pp`).
   - Dropdown slicers from `powerbi_test_portfolio`: `loan_grade`, `person_home_ownership`, `income_bracket`, `loan_intent`.
2. **4 KPI Cards:**
   - Card 1: `Mean PD Base (SVM)` (`22.20%`)
   - Card 2: `Mean PD Stressed (SVM)` (`30.37%`)
   - Card 3: `PD Jump pp (SVM)` (`+8.17 pp`, with callout or subtitle showing `Relative Surge (SVM)` `+36.80%`)
   - Card 4: `Expected Defaults Stressed (SVM)` (`1,484` borrowers)
3. **Chart 1 (Left) — Risk Quartile Stress Impact (`Clustered Column Chart`):**
   - Axis: `powerbi_test_portfolio[Risk_Quartile]` (`Q1: Ultra-Prime` $\to$ `Q4: High Distress`).
   - Values: `Mean PD Base (SVM)` (`#1F4E79`) and `Mean PD Stressed (SVM)` (`#D95F02`).
   - *Key takeaway visible:* `Q3: Near-Prime` shows the largest percentage-point jump (`11.26%` $\to$ `22.83%`, `+11.57 pp`).
4. **Chart 2 (Right) — Credit Grade Stress Impact (`Clustered Column Chart`):**
   - Axis: `powerbi_test_portfolio[loan_grade]` (`A` through `G`).
   - Values: `Mean PD Base (SVM)` (`#1F4E79`) and `Mean PD Stressed (SVM)` (`#D95F02`).
5. **Chart 3 (Bottom) — Ontario Macroeconomic Context (`Line Chart`):**
   - Data: `powerbi_macro_series` (`Date` on X-axis; `Ontario_Unemployment_Rate` and `Ontario_CPI_YoY` on Y-axis).
   - Gives the viewer real historical context (COVID-19 unemployment spike to `14.2%` in 2020; 2022 inflation peak at `8.13%`) justifying why `+3.0 pp` unemployment and `+2.5 pp` inflation represent a realistic severe stress test.

---

## Page 2: Model Comparison & Institutional Efficiency (RBF-SVM vs. Logistic Regression)

```text
+---------------------------------------------------------------------------------------------------+
| TITLE: Model Comparison: Accuracy vs. Institutional Efficiency (Held-Out Test Split, N = 4,888)   |
+---------------------------------------------------------------------------------------------------+
|  TABLE 1: Out-of-Sample Benchmark Comparison (Filter: model_comparison_metrics[Split] = "Test")   |
|  Naive Base-Rate (0.5000 AUC) | Logistic Regression (0.8700 AUC) | RBF-SVM (0.9013 AUC)           |
+---------------------------------------------------------------------------------------------------+
|                                                |                                                  |
|  CHART 4 (Line Chart — Calibration Curve)      |  CHART 5 (Clustered Bar — Compute vs. Value)     |
|  Data: calibration_deciles_comparison          |  Data: efficiency_ranking                        |
|  X: Decile_Bin (1 to 10)                       |  Y: Model                                        |
|  Y: SVM_Mean_Predicted_PD_Pct vs Observed,     |  X: Efficiency_Score (or Test_ROC_AUC & Fit_s)   |
|     LR_Mean_Predicted_PD_Pct vs Observed       |  Shows Logit (#1 Value, 0.13s) vs RBF (#2, 8.1s) |
|                                                |                                                  |
+------------------------------------------------+--------------------------------------------------+
|                                                                                                   |
|  CHART 6 (Clustered Column Chart — Live Model Stress Comparison)                                  |
|  X: powerbi_test_portfolio[Risk_Quartile]                                                         |
|  Y: [Mean PD Stressed (SVM)] vs [Mean PD Stressed (Logit)]                                        |
|  Shows how both models respond when you move the Unemp / CPI sliders on Page 1 (sync slicers!)    |
+---------------------------------------------------------------------------------------------------+
```

### Page 2 Visual Details
1. **Table 1 (Top):** Pull directly from `model_comparison_metrics.csv` filtered to `Split = "Test"`: shows `ROC_AUC`, `Brier_Score`, `Log_Loss`, `Accuracy_Pct`, `Precision_Pct`, `Recall_Pct`, and `F1_Score`.
2. **Chart 4 (Left — 10-Decile Calibration Check):**
   - From `calibration_deciles_comparison.csv`: `Decile_Bin` (`1..10`) on X-axis, plotting `SVM_Observed_Default_Pct` vs. `SVM_Mean_Predicted_PD_Pct` (and `LR_Mean_Predicted_PD_Pct`). Visually proves RBF-SVM tracks actual default rates across all 10 deciles (`1.59 pp` mean error vs. `2.51 pp` for Logistic Regression).
3. **Chart 5 (Right — Institutional Efficiency Ranking):**
   - From `efficiency_ranking.csv`: Horizontal bar chart sorted by `Efficiency_Rank`, showing why **Logistic Regression (`0.1719`, `0.13s`, `21` weights)** wins for institutional-scale speed/transparency while **RBF-SVM (`0.1026`, `8.15s`)** wins for non-linear accuracy (`99.64%` of peak AUC).
4. **Chart 6 (Bottom — Live Stress Comparison Across Models):**
   - Sync the `Unemp Shock` and `CPI Shock` slicers from Page 1 to Page 2 (**View $\to$ Sync slicers**), and plot `Mean PD Stressed (SVM)` vs. `Mean PD Stressed (Logit)` by `Risk_Quartile`.
