# Mental Health Treatment-Seeking Prediction
### DSC680 Capstone Project 1 — Bellevue University MSDS

---

<div align="center">
<table>
<tr>
<td width="50%"><img src="figures/fig08_roc_pr_calibration.png" alt="ROC PR Calibration"/></td>
<td width="50%"><img src="figures/fig11_shap_beeswarm.png" alt="SHAP beeswarm"/></td>
</tr>
<tr>
<td align="center"><sub>ROC · Precision-Recall · Calibration curves</sub></td>
<td align="center"><sub>SHAP — which factors drive treatment-seeking most</sub></td>
</tr>
<tr>
<td width="50%"><img src="figures/fig09_odds_ratios.png" alt="Odds ratios"/></td>
<td width="50%"><img src="figures/fig13_fairness_audit.png" alt="Fairness audit"/></td>
</tr>
<tr>
<td align="center"><sub>Logistic regression odds ratios per feature</sub></td>
<td align="center"><sub>Fairness audit across demographic subgroups</sub></td>
</tr>
</table>
</div>

---

## What This Project Does

Predicts whether a tech worker is likely to seek mental health treatment based on their workplace environment — helping HR teams understand which organizational factors actually drive people to get help.

Built on the OSMI 2016 tech-workplace mental health survey (1,259 respondents). Four classifiers were compared; Logistic Regression won for its interpretability. SHAP analysis ranks the top predictors. A fairness audit checks the model performs equally across age and gender groups.

## Key Findings

| Finding | Detail |
|---|---|
| Strongest predictor | Work interference — when mental health affects job performance, people act |
| Second strongest | Family history of mental illness |
| Employer lever | Benefits availability directly increases treatment likelihood |
| Fairness | No significant performance gaps across gender or age groups |

## Model Results

| Model | AUC | Winner |
|---|---|---|
| **Logistic Regression** | **0.723** | ✅ |
| Random Forest | 0.716 | |
| XGBoost | 0.714 | |
| Neural Network | 0.709 | |

## Project Structure

```
project1-dsc680/
├── code/                     # Python analysis pipeline
├── figures/                  # All output visualizations
├── milestone1_proposal/      # Research proposal
├── milestone2_whitepaper/    # White paper + infographic
└── milestone3_final/         # Final presentation + white paper
```

## Deliverables
| Milestone | Deliverable |
|---|---|
| M1 | Research proposal |
| M2 | White paper + interactive infographic |
| M3 | Final white paper · 13-slide presentation · Q&A |

`Python` `scikit-learn` `XGBoost` `SHAP` `SMOTE` `Pandas` `Seaborn`

---

**Komal Shahid · Bellevue University MSDS · DSC680 · 2026**
