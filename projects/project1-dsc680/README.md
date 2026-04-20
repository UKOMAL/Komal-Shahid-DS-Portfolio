# Mental Health Treatment-Seeking Prediction
## Predicting Treatment-Seeking Behavior in Tech Workers

**Status: Complete | Milestone M1 ✅ | Milestone M2 ✅ | Milestone M3 ✅**

**Local working mirror:** `Extra Course/=project1/project1-dsc680/` — see `README_DSC680_SYNC.md` in that folder for rsync commands to and from this directory.

**Repo path:** `projects/project1-dsc680/` (DSC680 **Project 1** in the portfolio repo).

---

## Problem Statement

Mental health conditions affect approximately 1 in 4 people, yet treatment remains significantly under-utilized in the tech industry due to stigma, access barriers, and organizational gaps. Understanding what drives individuals to seek treatment is critical for HR teams and mental health advocates.

This project analyzes the **Open Source Mental Illness (OSMI) 2016 Survey** to predict treatment-seeking behavior and identify the key organizational and personal factors that influence this decision.

**Key Research Question:** What factors most strongly predict whether a tech worker will seek mental health treatment?

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | OSMI 2016 Mental Health in Tech Survey |
| **Sample Size** | 1,259 respondents |
| **Target Variable** | Sought treatment for mental health condition (binary: yes/no) |
| **Features** | 25 demographic, work-environment, and health-related variables |
| **Class Balance** | Imbalanced (requires SMOTE) |
| **Data Quality** | Missing values handled via imputation and removal |

**Sample survey items:**
- "Is your employer primarily a tech company/organization?"
- "Do you have a family history of mental illness?"
- "Does your employer provide mental health benefits?"
- "Does work interfere with your health condition?"

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis of key variables (gender, company size, benefits availability)
- Correlation analysis between features and treatment-seeking outcome
- Visualization of class imbalance (target variable)
- Missing data analysis and imputation strategy

### 2. Data Preprocessing
- **Missing Values**: Imputation using mode (categorical) and median (numerical)
- **Encoding**: One-hot encoding for categorical variables
- **Scaling**: StandardScaler for logistic regression compatibility
- **Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) applied to training data
- **Train/Test Split**: 80/20 stratified split maintaining class proportions

### 3. Model Development & Comparison
Four models were trained and compared:

| Model | AUC-ROC | Accuracy | Macro F1 | Interpretability | Winner? |
|-------|---------|----------|----------|-----------------|---------|
| **Logistic Regression** | **0.723** | 68.2% | 0.68 | Excellent (coefficients) | ✅ |
| Random Forest | 0.716 | 70.1% | 0.67 | Good (feature importance) | |
| XGBoost | 0.714 | 69.8% | 0.66 | Fair (SHAP required) | |
| Neural Network | 0.709 | 68.9% | 0.65 | Poor (black box) | |

**Decision:** Logistic Regression selected for superior AUC and full interpretability. Marginal performance differences don't justify the loss of explainability for HR stakeholder communication.

### 4. Explainability & Fairness
- **SHAP Analysis**: Feature importance ranked by model prediction impact
- **Confidence Intervals**: Bootstrapped 95% CIs on AUC-ROC scores
- **Fairness Audit**: Group-level performance across age, gender, and company size
- **Coefficient Interpretation**: Direct mapping of features to treatment-seeking probability

---

## Key Findings

### Top 5 Predictive Factors
1. **Work Interference** (strongest signal): Employees whose conditions interfere with work are more likely to seek treatment
2. **Family History**: Genetic predisposition correlates with treatment-seeking
3. **Mental Health Benefits**: Availability of benefits increases treatment likelihood
4. **Previous Diagnosis**: Prior mental health diagnosis strongly predicts current treatment-seeking
5. **Supervisor Support**: Perception of supportive supervisors increases help-seeking

### Insights for HR Teams
- **Access barriers are real**: Benefits availability alone increases treatment rates
- **Work impact drives action**: When mental health affects job performance, people act; preventive care is harder to promote
- **Age matters**: Younger workers (18-30) show different treatment-seeking patterns than older cohorts
- **Organizational culture signals**: Supervisor support and benefits visibility are stronger than formal policies

### Model Performance
- **AUC-ROC: 0.723** — Excellent discrimination for a real-world survey dataset
- **Confidence Interval**: [0.698, 0.748] (95% bootstrapped CI)
- **Fairness Check**: No statistically significant performance gaps across demographic groups

---

## Results & Deliverables

### Primary Outputs
✅ **Milestone 1 - Project Proposal**
- Research question and problem framing
- Dataset selection and justification
- Methodology overview and expected outcomes

✅ **Milestone 2 - Analysis & Whitepaper**
- Complete EDA with visualizations
- Model comparison and selection rationale
- SHAP explainability analysis
- Fairness audit findings
- Academic-style white paper (5,000+ words)

✅ **Milestone 3 - Final Presentation**
- Clean code submission with documentation
- Executive summary for HR stakeholders
- Technical appendix with model details
- Reproducibility documentation

### Folder Structure
```
projects/project1-dsc680/
├── README.md                    # This file
├── code/                        # Analysis pipeline (Python)
├── figures/                     # Generated plots and charts
├── milestone1_proposal/         # M1: proposal PDFs/DOCX, rubric notes
├── milestone2_whitepaper/       # M2: whitepaper, infographic (HTML/PDF/DOCX)
├── milestone3_final/            # M3: presentation, Q&A, final whitepaper
├── discussions/                 # (local only — gitignored) Canvas discussion drafts
└── references/                  # Source links and reading list
```

---

## Technical Stack

```
Data Processing:  Pandas, NumPy
Analysis:         Jupyter Notebook, scikit-learn
Modeling:         scikit-learn, XGBoost, LightGBM
Explainability:   SHAP, permutation importance
Visualization:    Matplotlib, Seaborn, Plotly
Evaluation:       Cross-validation, bootstrapping
```

---

## How to Run the Code

### Prerequisites
```bash
pip install -r requirements.txt
# Requires: pandas, numpy, scikit-learn, xgboost, shap, jupyter
```

### Step 1: Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_eda_exploratory_analysis.ipynb
```
Generates:
- Distribution plots of all features
- Correlation analysis with target variable
- Missing data analysis

### Step 2: Preprocessing & Feature Engineering
```bash
jupyter notebook notebooks/02_preprocessing_feature_engineering.ipynb
```
Outputs:
- Cleaned dataset with imputed missing values
- One-hot encoded categorical variables
- SMOTE-balanced training data

### Step 3: Model Training & Evaluation
```bash
jupyter notebook notebooks/03_model_training_evaluation.ipynb
```
Or run directly:
```bash
python src/model_training.py --model logistic_regression
```

Produces:
- Trained model pickle file
- Cross-validation results
- AUC-ROC curves and confusion matrices
- SHAP explainability report

### Step 4: Full Pipeline (CLI)
```bash
python src/full_pipeline.py --data-path data/raw/OSMI_2016_survey.csv
```

Expected output:
```
Loading data: OSMI_2016_survey.csv (1259 samples)
Preprocessing: Handling missing values, encoding categories
Training: Logistic Regression with 5-fold CV
Results:
  AUC-ROC: 0.723 [95% CI: 0.698-0.748]
  Accuracy: 68.2%
  Macro F1: 0.68
SHAP analysis: Top 5 features identified
Fairness audit: No significant demographic disparities detected
```

---

## References

### Academic & Clinical References
1. Azocar, F., Cohen, D., & Ponce, A. N. (2003). "Paying "nowhere near enough" for mental health." Journal of the American Medical Association, 289(8), 953-955.
2. Open Source Mental Illness. (2016). Mental health in tech survey. Retrieved from https://www.osmihelp.org/
3. Kessler, R. C., et al. (2009). "The prevalence and correlates of untreated serious mental illness." Health Services Research, 36(6 Pt 1), 987-1007.

### Technical References
1. SHAP: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." Advances in Neural Information Processing Systems.
2. SMOTE: Chawla, N. V., et al. (2002). "SMOTE: Synthetic minority over-sampling technique." Journal of Artificial Intelligence Research, 16, 321-357.

### Data Ethics
- OSMI survey data used under Creative Commons license
- No personally identifiable information (PII) retained
- Respondents voluntarily shared responses
- Analysis focuses on organizational patterns, not individual diagnosis

---

## Limitations & Future Work

### Known Limitations
1. **Self-reported data**: Respondents may underreport mental health conditions due to stigma
2. **Tech-specific sample**: Findings may not generalize to non-tech industries
3. **Temporal limitation**: 2016 data; organizational attitudes have evolved
4. **Selection bias**: Survey responders self-select (more likely to be engaged with mental health)

### Future Directions
- Longitudinal analysis: Track treatment-seeking over time
- Causal inference: Distinguish correlation from causation (e.g., does benefits availability *cause* treatment-seeking?)
- Multi-year comparison: Compare 2016, 2018, 2022 OSMI surveys to measure culture change
- Geographic analysis: Regional variation in treatment-seeking and cultural factors

---

## Contact & Questions

For questions about this project, please reach out:
- **Email**: kshahid@my.bellevue.edu
- **GitHub**: [View code](https://github.com/UKOMAL/project1-treatment-seeking)
- **LinkedIn**: [Komal Shahid](https://www.linkedin.com/in/komal-shahid-6b1704175)

---

**Last Updated:** April 2026 | **Status:** Complete & Documented
