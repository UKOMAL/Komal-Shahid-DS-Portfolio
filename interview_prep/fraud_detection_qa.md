# Interview Prep: Financial Fraud Detection System
**Project:** DSC680 Capstone — AUC 0.886 on 800K+ transactions
**Role targets:** AI Engineer, ML Engineer, Data Scientist

---

## Talking Points (6 bullets for interviews)

1. **Scale and real data:** "I trained on 800K+ real-world transactions from three public datasets — ULB Credit Card, IEEE-CIS, and a banking dataset — without any synthetic label manipulation. Realistic results matter more than benchmark-inflated scores."
2. **Handling class imbalance correctly:** "The fraud rate was 0.17%. I used conservative SMOTE targeting 10% — not 50/50. Aggressive oversampling would have given a misleadingly high AUC but failed in production."
3. **No data leakage:** "I used temporal splitting, not random splits. A random split inflated my AUC from 0.886 to 0.999 — a 12-point phantom gain that would have collapsed in production."
4. **Explainability as a product feature:** "Every prediction comes with SHAP top-3 feature contributions. This unlocked production deployment approval because compliance could audit any flagged transaction."
5. **Bias audit:** "I checked disparate impact ratios across demographic proxies. All ratios were within the 0.80–1.25 acceptable range. Ethical AI isn't just a slide in a deck — it's a checklist in my pipeline."
6. **End-to-end ownership:** "The project includes a FastAPI endpoint, Dockerfile, GitHub Actions CI, and a reproducibility checklist. Any engineer can reproduce my results in one command."

---

## 10 Technical Interview Questions + Model Answers

### Q1: Why did you choose LightGBM over XGBoost or a neural network for fraud detection?

**Model answer:**
LightGBM uses leaf-wise tree growth instead of level-wise, which converges faster on large datasets (590K+ rows). It also has built-in categorical feature handling and efficient GPU/CPU switching. For tabular fraud data with 40+ engineered features, gradient-boosted trees consistently outperform neural networks unless you have sequence/graph structure. XGBoost was slower at the same hyperparameter budget. The ensemble with an Autoencoder gave the +0.004 AUC improvement by capturing anomaly patterns LightGBM's discriminative training misses.

```python
import lightgbm as lgb

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "reg_alpha": 0.1,   # L1
    "reg_lambda": 0.1,  # L2
    "num_leaves": 31,
    "min_child_samples": 20,
    "n_estimators": 1000,
}
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

---

### Q2: How did you handle the severe class imbalance (0.17% fraud rate)?

**Model answer:**
Three steps: (1) I used **stratified train/test splitting** to ensure the fraud rate was preserved in both sets. (2) I applied **conservative SMOTE** targeting a 10% minority ratio — not 50/50 — because aggressive oversampling teaches the model a distribution that doesn't exist in production. (3) I used **class_weight** in evaluation metrics and calibrated the final model with Platt scaling. The calibration reduced the Brier score from 0.052 to 0.031.

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

smote = SMOTE(sampling_strategy=0.10, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

### Q3: What is AUC-ROC and why is it the right metric for fraud detection?

**Model answer:**
AUC-ROC measures the model's ability to rank fraudulent transactions above legitimate ones across *all possible thresholds*. For fraud detection, you typically don't know the optimal threshold upfront (it depends on the cost ratio of false positives vs false negatives). AUC gives a threshold-independent measure of discriminative power. However, for heavily imbalanced data, **Average Precision (PR-AUC)** is often more informative than ROC-AUC because it focuses on the minority class. I reported both: ROC-AUC = 0.886, PR-AUC = 0.73.

---

### Q4: Explain how SHAP values work and why you used them.

**Model answer:**
SHAP (SHapley Additive exPlanations) assigns each feature a contribution to a specific prediction based on game-theoretic Shapley values. For a tree model, TreeSHAP computes exact values in O(TLD²) time (T=trees, L=leaves, D=depth). The sum of all SHAP values equals the model output minus the base rate.

I used SHAP because: (1) regulatory compliance requires per-prediction audit trails, (2) it's model-agnostic but fast for tree models, (3) it enables global feature importance *and* local prediction explanation in one framework.

```python
import shap

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

# For a single prediction explanation:
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0])
```

---

### Q5: What is data leakage and how did you prevent it?

**Model answer:**
Data leakage occurs when information from the test set (or future) influences the training set, producing artificially inflated metrics. In fraud detection, it commonly happens via: (1) random splitting time-series data (future labels leak into past features), (2) computing statistics (mean, std) on the full dataset before splitting, (3) fitting SMOTE before splitting.

I prevented it by: (1) **temporal splitting** — all transactions before date T go to train, after T to test, (2) computing all feature statistics (rolling means, etc.) using only training data and transforming test data separately, (3) applying SMOTE *inside* the cross-validation loop.

---

### Q6: How would you deploy this model to production?

**Model answer:**
I'd expose it as a FastAPI microservice with: a `/predict` endpoint accepting a JSON payload, input validation via Pydantic, the model loaded once on startup (not per-request), and SHAP values computed asynchronously if not needed in the synchronous response. For latency: LightGBM inference is ~12ms; the full endpoint including SHAP is ~57ms. For scale: containerise with Docker, deploy behind a load balancer, cache model weights in memory, and use a feature store for real-time feature lookups.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb, shap, numpy as np

app = FastAPI()
model = lgb.Booster(model_file="models/lgbm_v1.txt")
explainer = shap.TreeExplainer(model)

class Transaction(BaseModel):
    TransactionAmt: float
    card_velocity_1h: int
    device_consistency: float
    hour_of_day: int
    amount_deviation: float

@app.post("/predict")
def predict(tx: Transaction):
    features = np.array([[tx.TransactionAmt, tx.card_velocity_1h,
                           tx.device_consistency, tx.hour_of_day,
                           tx.amount_deviation]])
    prob = float(model.predict(features)[0])
    shap_vals = explainer.shap_values(features)[0].tolist()
    return {"fraud_probability": prob, "prediction": "FRAUD" if prob > 0.5 else "LEGIT",
            "shap_values": shap_vals}
```

---

### Q7: How did you measure and mitigate bias in your model?

**Model answer:**
I computed the **disparate impact ratio** (DIR) for demographic proxies derived from transaction patterns (e.g., gender-correlated spending patterns, geographic indicators). DIR = P(favorable outcome | group A) / P(favorable outcome | group B). EEOC/FHA guidelines require DIR ≥ 0.8. All proxy groups in my model had DIR between 0.83–1.21. I also verified that no protected-proxy features appeared in the SHAP top-10 global importance ranking.

---

### Q8: What cross-validation strategy did you use and why?

**Model answer:**
**5-fold stratified cross-validation** — stratified to preserve the 0.17% fraud rate in each fold. I did not use TimeSeriesSplit because the datasets don't have a clean temporal order across all three sources. For the IEEE-CIS dataset specifically (which has a temporal component), I used a time-aware split for final evaluation. CV AUC variance was ±0.009, indicating the model generalises well.

---

### Q9: Your AUC is 0.886. How do you explain that to a non-technical stakeholder?

**Model answer:**
"If I randomly pick one fraudulent transaction and one legitimate one, this model will correctly rank the fraud as riskier 88.6% of the time. For comparison, flipping a coin would give 50%. The current rule-based system achieves about 78%. Our model also flags 38% fewer legitimate transactions as suspicious, which means fewer customers get their cards blocked unnecessarily."

---

### Q10: What would you do differently if you had 6 more months?

**Model answer:**
1. **Graph Neural Networks** — model card/merchant/device relationships as a graph; GNN-based fraud detection catches collusion patterns tree models miss.
2. **Real-time feature store** — replace batch-computed velocity features with a streaming feature store (Feast + Kafka) for sub-second freshness.
3. **Concept drift monitoring** — fraud patterns evolve monthly; add a population stability index (PSI) monitor and automated retraining trigger.
4. **A/B test the threshold** — work with the business to find the optimal precision/recall operating point based on actual fraud investigation costs.
