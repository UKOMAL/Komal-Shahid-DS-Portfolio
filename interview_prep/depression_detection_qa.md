# Interview Prep: AI-Powered Depression Detection
**Project:** DSC670 — 91% accuracy, 4-class NLP classifier
**Role targets:** AI Engineer (NLP), ML Engineer, Data Scientist

---

## Talking Points (6 bullets for interviews)

1. **Transformer fine-tuning on a constrained budget:** "I fine-tuned DistilBERT — not full BERT — to hit 91% accuracy on 4-class severity classification within CPU-feasible training time. DistilBERT retains 97% of BERT's performance with 40% fewer parameters."
2. **Calibrated uncertainty:** "Every prediction includes all 4 class probabilities. Any prediction with max confidence < 0.60 triggers a 'Review Required' flag. In healthcare NLP, knowing when to abstain is as important as knowing when to predict."
3. **Rule-based safety layer:** "I layered rule-based overrides on top of the model — explicit suicidal ideation markers always escalate to severe regardless of the transformer output. Defense-in-depth for high-stakes decisions."
4. **Attention as an entry point for clinicians:** "I visualised attention weights not as ground-truth explanations, but as conversation starters for clinical review. Attention shows *where* the model looked; clinicians can verify *whether* that's the right place."
5. **Ethical framing from day one:** "The white paper explicitly positions this as a triage support tool, not a diagnostic system. That framing prevented scope creep and guided every design decision."
6. **Reproducible NLP pipeline:** "`transformers.set_seed(42)` applied globally; model checkpoint committed; tokeniser config serialised. Any collaborator can reproduce my fine-tuning from scratch."

---

## 10 Technical Interview Questions + Model Answers

### Q1: Why DistilBERT instead of full BERT or a larger model?

**Model answer:**
DistilBERT retains 97% of BERT-base performance on GLUE benchmarks while being 40% smaller and 60% faster. For a classification task on short social/clinical text (typically < 100 tokens), the performance gap between DistilBERT and BERT-base is negligible. Training on CPU (6h) vs BERT-base (estimated 24h+) was a practical constraint. If production latency and GPU cost were less of a concern, I'd experiment with RoBERTa or DeBERTa-v3-base, which consistently outperform BERT-base on text classification benchmarks.

---

### Q2: Explain the fine-tuning process for DistilBERT.

**Model answer:**
Fine-tuning involves: (1) loading pre-trained weights from HuggingFace Hub, (2) replacing the final NSP/MLM heads with a classification head `Linear(768, num_classes)`, (3) training all layers end-to-end with a low learning rate (2e-5) and linear warmup scheduler, (4) using `CrossEntropyLoss` with class weights for imbalance.

```python
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)

training_args = TrainingArguments(
    output_dir="./models/transformer",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()
```

---

### Q3: How did you evaluate multi-class classification? Why not just accuracy?

**Model answer:**
For a 4-class imbalanced problem, accuracy alone is misleading. I reported: **macro-F1** (treats all classes equally — important when all 4 severity levels matter clinically), **per-class F1** (identifies if any class is being ignored), and **confusion matrix** (shows which severity levels get confused with each other). The severe class had the highest clinical consequence, so I specifically monitored severe-class recall to ensure we weren't missing critical cases.

---

### Q4: What is "attention" in transformers and is it an explanation?

**Model answer:**
Attention weights represent how much each token influences the final representation of a target token during the self-attention computation: `softmax(QK^T / √d_k) × V`. They are not causal explanations — Jain & Wallace (2019) showed that attention weights are not faithful to the prediction (you can change attention without changing the output). However, they remain useful as an *interpretability entry point*: they show which tokens the model focused on, even if that's not the complete mechanistic explanation. For clinical use, I presented attention as "what the model looked at" with the caveat that human review is still required.

---

### Q5: How did you handle the ethical concerns of building a mental health classifier?

**Model answer:**
Several concrete steps: (1) **Framing** — documented explicitly in the white paper that this is a triage tool, not a diagnostic system; never positions itself as replacing clinical judgment. (2) **Confidence gating** — predictions below 0.60 confidence trigger a "Review Required" flag rather than a severity label. (3) **Suicidal ideation override** — explicit keyword/pattern overrides escalate to severe regardless of model output. (4) **No PII** — Reddit posts used at aggregate level with no user identification. (5) **Bias review** — tested for demographic parity across gender-associated language patterns.

---

### Q6: How would you improve this model's accuracy further?

**Model answer:**
1. **More data:** The primary bottleneck is labeled data. Clinical annotations from licensed psychologists (not crowd-sourced) would significantly improve quality.
2. **Domain-adaptive pre-training (DAPT):** Continue pre-training DistilBERT on a mental health corpus before fine-tuning (Gururangan et al., 2020 showed 2–5% improvement from DAPT).
3. **Larger model:** RoBERTa-base or DeBERTa-v3-base would likely improve by 1–3% on this task.
4. **Data augmentation:** Back-translation or synonym replacement for the minority classes.
5. **Multi-task learning:** Train simultaneously on related tasks (sentiment, emotion detection) to improve representations.

---

### Q7: What is the difference between accuracy and macro-F1 in this context?

**Model answer:**
Accuracy = (correct predictions) / (total predictions) — each sample contributes equally. If 60% of the dataset is "minimum" severity, a model that always predicts "minimum" achieves 60% accuracy but learns nothing. Macro-F1 = average of per-class F1 scores, giving equal weight to each class regardless of size. For a clinical severity classifier, all four classes matter equally — you don't want to trade severe-class recall for minimum-class precision. Macro-F1 was 0.89 vs accuracy of 0.91 — the gap indicates slight class imbalance in the data.

---

### Q8: How did you tokenise the input text?

**Model answer:**
Using `DistilBertTokenizerFast` with `max_length=128`, `padding="max_length"`, `truncation=True`. The 128 token limit covers 95%+ of the text samples (clinical texts and Reddit posts are typically under 100 tokens after preprocessing). Longer texts are rare and get truncated from the right. For production, I'd add a pre-check that logs if truncation occurs to monitor distribution shift.

```python
from transformers import DistilBertTokenizerFast

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=128)
```

---

### Q9: How would you deploy this NLP model in production?

**Model answer:**
1. Export to ONNX for runtime-agnostic serving (`transformers` has `convert_graph_to_onnx.py`)
2. Serve via FastAPI with batched inference (HuggingFace pipeline with `batch_size=32`)
3. Add an async confidence check — responses below threshold route to a "manual review" queue
4. Cache tokeniser and model on startup
5. Monitor: input text length distribution, class distribution of outputs, confidence score distribution over time (drift detection)

---

### Q10: What would you do differently if you had 6 more months?

**Model answer:**
1. **Clinical validation study** — partner with a psychologist or clinical researcher to evaluate predictions against actual PHQ-9 scores on a prospective dataset.
2. **LLM zero-shot baseline** — benchmark against GPT-4 zero-shot prompting to understand the ceiling of this task.
3. **Longitudinal modelling** — model sequences of posts over time to detect trajectory (improving/worsening), not just point-in-time severity.
4. **Multilingual support** — mental health NLP is critically under-resourced for non-English speakers; mBERT or XLM-R would enable cross-lingual transfer.
