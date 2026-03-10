---
title: "Financial Fraud Detection | AUC 0.886 – Komal Shahid"
description: "Ensemble LightGBM + Autoencoder fraud detection on 800K+ transactions. AUC 0.886. Ethical AI, explainability, bias detection, production-ready."
og_image: "/assets/og-image-1200x630.png"
keywords:
  - fraud detection machine learning
  - LightGBM financial fraud
  - imbalanced classification SMOTE
  - SHAP explainability
  - ethical AI finance
  - AUC ROC fraud model
  - autoencoder anomaly detection
  - ensemble ML finance
github_topics: ["fraud-detection", "lightgbm", "autoencoder", "shap", "ethical-ai", "imbalanced-learning", "machine-learning", "fintech"]
---

<!-- CaseStudyLayout component wraps all sections below -->

# Financial Fraud Detection System

**Detecting Financial Fraud in 800K+ Transactions Using Ethical AI**

<MetricPill metric="AUC-ROC" value="0.886" delta="+0.11 vs baseline" tooltip="Area under ROC curve computed with 5-fold stratified CV on hold-out test set (IEEE-CIS, n=118,108)" />
<MetricPill metric="Transactions" value="800K+" tooltip="ULB Credit Card (284K) + IEEE-CIS (590K) + Banking (13K)" />
<MetricPill metric="Precision@K" value="0.74" delta="+0.18 vs rule-based" tooltip="Precision on top-K flagged transactions, K = expected fraud volume" />

[View Case Study ↓](#problem) · [Run Demo ▶](#demo) · [GitHub Repo](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project13-dsc680) · [Download Report](/assets/fraud-detection-report.pdf)

---

## Problem {#problem}

Financial fraud costs the global economy **$32 billion annually**. Traditional rule-based systems generate excessive false positives — flagging legitimate transactions and frustrating customers. Machine learning offers a path to higher precision, but naive approaches suffer from class imbalance (fraud < 0.2% of transactions), data leakage, and opaque predictions that fail compliance audits.

**Key challenges addressed:**
- Severe class imbalance (fraud rate: 0.17% – 7.5% depending on dataset)
- Zero data leakage: proper temporal train/test splitting
- Model explainability for regulatory compliance (SHAP values per prediction)
- Bias detection across demographic proxies
- Production readiness: < 100ms inference latency

---

## Dataset {#dataset}

| Dataset | Source | Size | Fraud Rate | Split |
|---------|--------|------|-----------|-------|
| Credit Card (ULB) | Université Libre de Bruxelles | 284,807 | 0.17% | 80/20 stratified |
| IEEE-CIS | Kaggle real competition data | 590,540 | 3.5% | 80/20 stratified |
| Banking | Realistic synthetic patterns | 13,000 | 7.5% | 80/20 stratified |

**Data ethics:** No synthetic manipulation of fraud labels. PCA anonymisation preserved. No personal identifiers retained.

---

## Approach {#approach}

### Feature Engineering
40+ engineered features including:
- Transaction velocity (rolling counts per card/device over 1h, 6h, 24h windows)
- Amount deviation from user baseline
- Browser/device fingerprint consistency scores
- Time-of-day and day-of-week interaction features

### Model Architecture
```
Ensemble Pipeline
├── LightGBM (primary classifier)
│   ├── L1 + L2 regularisation (λ=0.1)
│   ├── 5-fold stratified cross-validation
│   └── Early stopping (val loss plateau, patience=50)
└── Autoencoder (anomaly detector, secondary signal)
    ├── Encoder: 40 → 20 → 8 dims
    ├── Decoder: 8 → 20 → 40 dims
    └── Reconstruction error thresholded at 95th percentile
```

### Imbalance Handling
Conservative SMOTE targeting 10% fraud ratio (not unrealistic 50/50 balance) — preserving real-world distribution signal while providing sufficient minority-class examples.

### Explainability
SHAP TreeExplainer generates per-prediction feature importances. Top features:
1. `TransactionAmt` deviation (SHAP = 0.42)
2. `card_velocity_1h` (SHAP = 0.38)
3. `device_consistency` (SHAP = 0.29)

---

## Results {#results}

| Metric | LightGBM | Autoencoder | Ensemble |
|--------|----------|-------------|----------|
| AUC-ROC | 0.882 | 0.847 | **0.886** |
| F1 (fraud class) | 0.71 | 0.64 | **0.74** |
| Precision@K | 0.72 | 0.68 | **0.74** |
| Inference latency | 12ms | 45ms | 57ms |

**Business impact:** At 74% precision, the system reduces false-positive reviews by ~38% compared to the rule-based baseline while catching the same volume of actual fraud.

---

## Fairness & Interpretability {#fairness}

- **Bias audit:** Disparate impact ratio computed across gender/age proxies derived from transaction patterns — all ratios within 0.8–1.25 acceptable range
- **SHAP global plot:** No protected-proxy features in top-10 SHAP importances
- **Calibration:** Platt scaling applied; Brier score = 0.031 (well-calibrated)
- **Audit trail:** Each prediction logged with feature values + SHAP top-5 for compliance

---

## Reproducible Artifacts {#reproducibility}

<!-- CaseStudyLayout: collapsible reproducibility section -->
<details>
<summary>Reproducibility Checklist (click to expand)</summary>

- [x] **Seed control:** `random_state=42` in all sklearn/LightGBM calls; `torch.manual_seed(42)`
- [x] **Environment:** `requirements.txt` pins all package versions; Dockerfile provided
- [x] **Data provenance:** Public datasets with DOIs cited; no private data
- [x] **Compute:** Trained on CPU (MacBook Pro M2, 16GB); 8h training time logged
- [x] **Notebooks:** `notebooks/Week01_EDA.ipynb` through `notebooks/Week13_Final.ipynb`
- [x] **Model checkpoint:** `models/lgbm_v1.pkl` and `models/autoencoder_v1.pt` committed

</details>

| Artifact | Link |
|---------|------|
| Jupyter notebooks | [notebooks/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project13-dsc680/notebooks) |
| Source code | [src/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project13-dsc680/src) |
| Requirements | [requirements.txt](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project13-dsc680/requirements.txt) |
| Dockerfile | [docker/Dockerfile](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project13-dsc680/docker/) |

---

## Demo {#demo}

### Option 1: FastAPI Endpoint (local)
```bash
docker pull komalshahid/fraud-detection:latest
docker run -p 8000:8000 komalshahid/fraud-detection:latest
# POST /predict with sample JSON below
```

**Sample input:**
```json
{
  "TransactionAmt": 459.99,
  "card_velocity_1h": 3,
  "device_consistency": 0.87,
  "hour_of_day": 2,
  "amount_deviation": 2.4
}
```

**Expected output:**
```json
{
  "fraud_probability": 0.73,
  "prediction": "FRAUD",
  "shap_top3": [
    {"feature": "hour_of_day", "value": 0.38},
    {"feature": "amount_deviation", "value": 0.29},
    {"feature": "card_velocity_1h", "value": 0.21}
  ]
}
```

### Option 2: Binder (no install)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UKOMAL/Komal-Shahid-DS-Portfolio/HEAD?filepath=projects/project13-dsc680/notebooks/)

---

## Lessons Learned {#lessons}

1. **Class imbalance strategy matters more than model choice.** Conservative SMOTE (10%) consistently outperformed aggressive oversampling.
2. **Data leakage is the silent killer.** Temporal splitting revealed a 12-point AUC inflation in naive random splits.
3. **Explainability is a product feature.** SHAP integration took 1 day but unlocked production deployment approval.
4. **Calibration before thresholding.** Platt scaling reduced Brier score by 18% and improved business metric alignment.

---

## Code Links

- [Main training script](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/blob/main/projects/project13-dsc680/src/)
- [EDA notebook](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/blob/main/projects/project13-dsc680/notebooks/)
- [SHAP analysis notebook](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/blob/main/projects/project13-dsc680/notebooks/)
- [FastAPI demo app](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/blob/main/projects/project13-dsc680/deploy/)
