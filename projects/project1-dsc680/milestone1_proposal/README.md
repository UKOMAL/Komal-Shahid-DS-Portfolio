# Milestone 1: Project Proposal

**Status: ✅ COMPLETE**

---

## Contents

This directory contains the formal project proposal submitted for Milestone 1.

### Course submission files (`shahid_dsc680_milestone1_proposal*.pdf` / `.docx`)
**Primary deliverables for M1** (PDF and Word versions, plus rubric/originality notes in this folder)

**Section Overview:**

1. **Title & Abstract**
   - Project title: Mental Health Treatment-Seeking Prediction in Tech
   - Abstract: 250-word summary of research question, dataset, methodology, and expected outcomes

2. **Problem Statement**
   - Why this matters: Mental health conditions affect millions; treatment remains underutilized
   - Research question: What factors predict treatment-seeking behavior in tech workers?
   - Stakeholders: HR teams, mental health advocates, tech companies

3. **Dataset Justification**
   - Source: OSMI 2016 Mental Health in Tech Survey
   - Sample size: 1,259 respondents
   - Features: 25 variables covering demographics, workplace, and health
   - Why suitable: Real-world data, public domain, sufficient sample size

4. **Proposed Methodology**
   - Data collection/loading: OSMI survey CSV
   - Exploratory analysis: Distribution and correlation analysis
   - Preprocessing: Handling missing values, encoding categorical variables
   - Modeling approach: Compare 4 models (Logistic Regression, Random Forest, XGBoost, NN)
   - Evaluation: AUC-ROC, accuracy, cross-validation, fairness analysis

5. **Expected Outcomes**
   - Prediction accuracy: Target AUC ≥ 0.70 on real-world data
   - Interpretability: Identify top 5 factors predicting treatment-seeking
   - Fairness: Audit for demographic disparities
   - Deliverables: Code, notebook, visualizations, whitepaper

6. **Timeline**
   - M1 (Project Proposal): Week 1-2
   - M2 (Analysis & Whitepaper): Week 3-6
   - M3 (Final Presentation): Week 7-8

---

## Rubric Compliance

### Code Functionality (25%)
- ✅ Objectives: Clear problem statement with measurable success criteria
- ✅ Requirements: All required analysis steps outlined
- ✅ Evaluation: Specific metrics identified (AUC, accuracy, fairness)

### Documentation/Naming Standards (25%)
- ✅ Clear proposal structure with organized sections
- ✅ Dataset justification with specific details
- ✅ Methodology explained at appropriate level of detail

### Content (25%)
- ✅ Covers full scope of work (EDA, modeling, evaluation)
- ✅ Addresses multiple skill areas (statistics, ML, ethics)
- ✅ Demonstrates breadth of MSDS program knowledge

### Assignment Specific Compliance (25%)
- ✅ Follows proposal guidelines
- ✅ Professional writing and formatting
- ✅ Realistic scope (achievable in 8 weeks)
- ✅ Clear milestones and deliverables

---

## Key Metrics from Proposal

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-ROC | ≥ 0.70 | Excellent discrimination for real survey data |
| Accuracy | ≥ 65% | Conservative target given class imbalance |
| Model coverage | 4+ models | Compare multiple approaches |
| Fairness audit | Yes | Ethics-first approach |
| Feature analysis | Top 5 | Actionable insights for stakeholders |

---

## Related Files

- **Main README**: `../README.md` — Complete project overview
- **M2 Whitepaper**: `../milestone2_whitepaper/` — Detailed analysis results
- **M3 Final Code**: `../code/` — Source code implementation

---

**Approved:** April 2026
**Status:** Milestone 1 ✅ Complete
