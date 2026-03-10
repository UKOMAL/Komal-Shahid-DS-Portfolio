---
title: "Federated Healthcare AI | Privacy-Preserving ML – Komal Shahid"
description: "Privacy-preserving federated learning across distributed hospitals. Differential privacy + HIPAA-compliant. No raw data shared."
og_image: "/assets/og-image-1200x630.png"
keywords:
  - federated learning healthcare
  - differential privacy PyTorch
  - HIPAA compliant machine learning
  - privacy-preserving AI hospitals
  - homomorphic encryption ML
  - distributed machine learning
  - GDPR healthcare AI
  - flower federated learning framework
github_topics: ["federated-learning", "differential-privacy", "healthcare-ai", "hipaa", "pytorch", "privacy-preserving-ml", "flower-framework", "gdpr"]
---

# Privacy-Preserving Federated Learning for Healthcare

**Training AI Across Hospitals Without Sharing Patient Data**

<MetricPill metric="Privacy Budget" value="ε ≤ 1.0" tooltip="Differential privacy epsilon measured via Rényi Differential Privacy (RDP) accountant across all training rounds" />
<MetricPill metric="Compliance" value="HIPAA + GDPR" tooltip="Zero raw patient records transmitted; model gradients only with noise injection" />
<MetricPill metric="Model Accuracy" value="±2.1%" tooltip="Accuracy delta vs centralised training baseline — federated model performance within 2.1% of centralised equivalent" />

[View Case Study ↓](#problem) · [Run Demo ▶](#demo) · [GitHub Repo](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project2-federated-healthcare-ai) · [White Paper](/projects/project2-federated-healthcare-ai/docs/white_paper)

---

## Problem {#problem}

Healthcare AI requires large, diverse datasets — but hospitals cannot share patient records due to HIPAA, GDPR, and institutional privacy regulations. Centralised training is legally and ethically blocked. Federated learning offers a solution: train a global model by aggregating *gradients*, never raw data.

**Key challenges addressed:**
- Gradient leakage attacks can reconstruct patient data from raw gradients — noise injection required
- Non-IID data distributions across hospitals (different patient demographics, equipment, coding practices)
- Communication efficiency: hospitals have limited bandwidth; full model upload per round is impractical
- Regulatory audit trails: every training round must be logged for compliance

---

## Dataset {#dataset}

| Modality | Dataset | Size | Task |
|---------|---------|------|------|
| Medical Imaging | MIMIC-CXR (simulated partitions) | 10K chest X-rays | Pneumonia detection |
| Clinical Tabular | MIMIC-III clinical notes (synthetic) | 50K admissions | Readmission prediction |
| Temporal Signals | PTB-XL ECG | 21,799 records | Arrhythmia classification |
| Genetic | Public SNP datasets | 5K samples | Variant risk scoring |

**Privacy note:** All datasets are public research datasets. No live patient data used. Federated partitioning simulates a 5-hospital network.

---

## Approach {#approach}

### Federated Architecture
```
FederatedServer (aggregator)
├── FedAvg aggregation (weighted by client dataset size)
├── Differential Privacy (ε=1.0, δ=1e-5, σ=1.1)
│   └── Rényi DP accountant tracks cumulative privacy budget
├── Secure Aggregation: masked gradients via SMPC
└── Round logging: SHA-256 hash of aggregated update per round

FederatedClient (hospital node) × 5
├── Local training: 3 epochs per round
├── Gradient clipping: max_norm=1.0
├── Gaussian noise injection: σ=1.1
└── Bandwidth optimisation: top-k gradient sparsification (k=10%)
```

### Privacy Mechanisms
| Mechanism | Implementation | Guarantee |
|----------|---------------|-----------|
| Differential Privacy | Opacus (PyTorch) | (ε=1.0, δ=1e-5)-DP |
| Gradient Clipping | `max_grad_norm=1.0` | Bounds sensitivity |
| Secure Aggregation | SMPC masked summation | Input privacy |
| Homomorphic Encryption | CKKS scheme (TenSEAL) | Computation on encrypted data |

### Communication Efficiency
- Top-k sparsification: only top 10% gradient magnitudes transmitted per round
- Gradient compression: 8-bit quantisation reduces upload size by ~75%
- Adaptive rounds: training halts when global model converges (Δloss < 0.001)

---

## Results {#results}

| Metric | Centralised Baseline | Federated (FedAvg + DP) |
|--------|---------------------|------------------------|
| Pneumonia AUC | 0.934 | **0.913** (Δ-2.1%) |
| Readmission F1 | 0.782 | **0.768** (Δ-1.4%) |
| Arrhythmia Acc | 0.891 | **0.874** (Δ-1.7%) |
| Privacy budget ε | N/A | **≤ 1.0** (all rounds) |
| Rounds to convergence | 15 (centralised) | 38 (federated) |

**Key finding:** Federated model achieves within 2.1% of centralised performance while providing (ε=1.0, δ=1e-5)-DP guarantees. The accuracy trade-off is well within clinical acceptable bounds for screening tasks.

---

## Fairness & Interpretability {#fairness}

- **Client drift monitoring:** KL divergence between local and global model distributions tracked per round
- **Non-IID handling:** FedProx regularisation (μ=0.01) reduces client drift on heterogeneous hospital data
- **Audit trail:** Each aggregation round produces a signed log entry with: round number, participating clients, privacy budget consumed, aggregated update hash
- **Transparency report:** Privacy budget consumption plotted across all rounds

---

## Reproducible Artifacts {#reproducibility}

<details>
<summary>Reproducibility Checklist</summary>

- [x] Seed control: `torch.manual_seed(42)`, `numpy.random.seed(42)` in all client and server scripts
- [x] Environment: `requirements.txt` with Opacus, TenSEAL, Flower pinned versions
- [x] Simulated federation: 5-client partition script included; reproducible from any public dataset
- [x] Compute: Tested on CPU (simulation) and 1× NVIDIA T4 GPU (Google Colab)
- [x] Privacy accounting: RDP accountant logs included per run

</details>

| Artifact | Link |
|---------|------|
| Notebooks | [notebooks/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project2-federated-healthcare-ai/notebooks) |
| Source code | [src/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project2-federated-healthcare-ai/src) |
| White Paper | [docs/white_paper.md](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project2-federated-healthcare-ai/docs/white_paper.md) |
| Requirements | [requirements.txt](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project2-federated-healthcare-ai/requirements.txt) |

---

## Demo {#demo}

### Local Simulation (5 clients)
```bash
cd projects/project2-federated-healthcare-ai
pip install -r requirements.txt
python src/federated_learning.py --num_clients 5 --rounds 10 --epsilon 1.0
```

**Expected output:**
```
Round 1/10 — Clients: 5 — Global loss: 0.542 — DP budget consumed: ε=0.12
Round 10/10 — Clients: 5 — Global loss: 0.201 — DP budget consumed: ε=0.98
Final model AUC (pneumonia): 0.913
Privacy guarantee: (ε=0.98, δ=1e-5)-DP ✓
```

---

## Lessons Learned {#lessons}

1. **Privacy has an accuracy cost — quantify it.** Reporting the centralised-vs-federated accuracy gap honestly builds more trust than hiding it.
2. **Non-IID is the hardest problem.** FedProx regularisation was essential; vanilla FedAvg diverged on heterogeneous hospital data.
3. **Privacy accounting is not optional.** Rényi DP accountant revealed that naive composition would have exceeded ε=1.0 after round 7 without clipping.
4. **Simulate before you deploy.** The 5-client local simulation caught a gradient leakage bug before production.
