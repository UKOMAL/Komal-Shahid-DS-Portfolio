---
title: "AI Depression Detection | NLP Case Study – Komal Shahid"
description: "DistilBERT-powered depression severity classifier across 4 classes. 91% accuracy on clinical text. Full case study with reproducible notebooks and live demo."
og_image: "/assets/og-image-1200x630.png"
keywords:
  - NLP depression detection
  - DistilBERT mental health
  - clinical text classification
  - transformer severity classification
  - mental health AI ethics
  - HuggingFace fine-tuning
  - attention visualization NLP
  - responsible AI healthcare
github_topics: ["nlp", "distilbert", "mental-health", "huggingface", "transformers", "clinical-nlp", "depression-detection", "responsible-ai"]
---

# AI-Powered Depression Detection System

**NLP Severity Classifier: Minimum · Mild · Moderate · Severe**

<MetricPill metric="Accuracy" value="91%" tooltip="4-class accuracy on hold-out test set, DistilBERT fine-tuned on clinical text" />
<MetricPill metric="Classes" value="4" tooltip="Minimum / Mild / Moderate / Severe — following PHQ-9 severity bands" />
<MetricPill metric="Inference" value="~210ms" tooltip="CPU inference time per text sample on MacBook Pro M2" />

[View Case Study ↓](#problem) · [Run Demo ▶](#demo) · [GitHub Repo](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection) · [White Paper](/projects/project1-depression-detection/docs/white_paper)

---

## Problem {#problem}

Mental health conditions affect 1 in 4 people, yet clinical screening remains slow, expensive, and under-resourced. NLP tools can triage written self-reports at scale — but only if they are accurate, interpretable, and ethically deployed.

**Key challenges addressed:**
- Multi-class severity classification (not just binary positive/negative)
- Confidence calibration so low-confidence predictions trigger human review
- Attention visualisation to explain *why* a text was classified at a given severity
- Ethical guardrails: system flags for human clinician review, never replaces diagnosis

---

## Dataset {#dataset}

| Source | Type | Size | Labels |
|--------|------|------|--------|
| Reddit mental health subreddits (r/depression, r/mentalhealth) | Social text | ~10K posts | PHQ-9 aligned severity bands |
| Synthetic clinical vignettes | Controlled text | ~2K samples | Expert-annotated |

**Data ethics:** No personally identifiable information retained. Posts used under Reddit API terms. Synthetic vignettes generated from clinical literature, not real patient records.

---

## Approach {#approach}

### Model Architecture
```
DistilBERT (distilbert-base-uncased)
├── Pre-trained weights: HuggingFace Hub
├── Fine-tuning: 3 epochs, lr=2e-5, batch=16
├── Classification head: Linear(768 → 4)
└── Rule-based post-processing:
    ├── Keyword boosting for severity markers (suicidal ideation → severe)
    └── Confidence threshold: < 0.6 → "Review Required" flag
```

### Training Details
- Tokeniser: `DistilBertTokenizerFast`, max_length=128, padding/truncation
- Optimiser: AdamW with linear warmup (10% steps)
- Loss: CrossEntropyLoss with class weights (inverse frequency)
- Evaluation: Accuracy, macro-F1, per-class F1, confusion matrix

### Visualisations Produced
- Confusion matrix (transformer vs rule-based)
- Attention heatmaps per prediction
- Word frequency by severity class
- Sentiment distribution across severity bands
- Interactive model comparison chart

---

## Results {#results}

| Metric | Rule-Based Baseline | DistilBERT Fine-tuned |
|--------|--------------------|-----------------------|
| Accuracy | 71% | **91%** |
| Macro-F1 | 0.68 | **0.89** |
| Severe class F1 | 0.72 | **0.93** |
| Minimum class F1 | 0.65 | **0.87** |
| Calibration (ECE) | 0.18 | **0.06** |

**Key finding:** Rule-based systems over-classify mild cases as moderate. DistilBERT learns contextual nuance (e.g., past tense vs present distress) that keyword matching misses.

---

## Fairness & Interpretability {#fairness}

- **Attention visualisation:** Highlights tokens driving classification — enables clinician review of model reasoning
- **Confidence output:** All 4 class probabilities returned; any prediction with max confidence < 0.60 triggers "Review Required" flag
- **Bias review:** Tested for demographic parity across gender-associated language patterns — no statistically significant disparity found
- **Ethical boundary:** System explicitly positioned as a *triage support tool*, not a diagnostic system

---

## Reproducible Artifacts {#reproducibility}

<details>
<summary>Reproducibility Checklist</summary>

- [x] Seed control: `transformers.set_seed(42)` applied globally
- [x] Environment: `requirements.txt` with pinned torch, transformers, datasets versions
- [x] Model checkpoint: `models/transformer/` contains fine-tuned weights
- [x] Data provenance: Reddit scrape date logged; synthetic vignette generation script included
- [x] Compute: Fine-tuning on CPU (6h) and GPU (45min) — both paths documented

</details>

| Artifact | Link |
|---------|------|
| Notebooks | [notebooks/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection) |
| Source code | [src/](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection/src) |
| White Paper | [docs/white_paper.md](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection/docs/white_paper.md) |

---

## Demo {#demo}

### CLI Usage
```bash
cd projects/project1-depression-detection
pip install -r requirements.txt
python src/depression_detection.py --text "I have been feeling completely hopeless for weeks"
```

**Expected output:**
```
Classification: MODERATE
Confidence: 0.81
Breakdown: minimum=0.04, mild=0.11, moderate=0.81, severe=0.04
Attention highlight: "completely hopeless" (weight: 0.73), "for weeks" (weight: 0.41)
```

### Binder (no install)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UKOMAL/Komal-Shahid-DS-Portfolio/HEAD?filepath=projects/project1-depression-detection/)

---

## Lessons Learned {#lessons}

1. **Confidence calibration is non-negotiable in healthcare.** Temperature scaling reduced ECE from 0.18 to 0.06.
2. **Rule-based post-processing adds safety without sacrificing accuracy.** Keyword overrides handle edge cases the model misses.
3. **Attention ≠ explanation, but it helps.** Attention weights provided useful entry points for clinician review even knowing their limitations.
4. **Ethics documentation is a deliverable.** The white paper became the most-read artifact in the project.
