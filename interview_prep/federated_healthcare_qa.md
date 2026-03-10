# Interview Prep: Privacy-Preserving Federated Healthcare AI
**Project:** DSC640 — ε=1.0 DP, 5-hospital simulation, PyTorch + Flower
**Role targets:** AI Engineer, ML Engineer (privacy/healthcare), Research Scientist

---

## Talking Points (6 bullets for interviews)

1. **Solving the real problem:** "Hospitals have the data. Researchers have the algorithms. HIPAA prevents sharing the data. Federated learning closes that gap — and I implemented it with quantifiable privacy guarantees, not just a HIPAA checkbox."
2. **Quantified privacy:** "I used Rényi differential privacy accounting — not the naive composition theorem — because naive composition is exponentially pessimistic. The RDP accountant lets me run more rounds within the same ε budget."
3. **Honest about the trade-off:** "Federated model was 2.1% less accurate than centralised. I could have hidden that. Instead, I reported it and showed the privacy-utility trade-off curve so stakeholders can make an informed decision."
4. **Non-IID is the hard problem:** "Vanilla FedAvg diverged on heterogeneous hospital data. FedProx with μ=0.01 regularisation brought the federated model back within 2.1% of centralised — that proximal term is what makes FL practical in healthcare."
5. **Communication efficiency:** "8-bit gradient quantisation reduced upload size by 75% — critical for hospitals with limited bandwidth to a central aggregator."
6. **Simulation discipline:** "The 5-client simulation used a reproducible partition script. Any collaborator can replicate the IID vs non-IID data splits and observe the same convergence behaviour."

---

## 10 Technical Interview Questions + Model Answers

### Q1: What is federated learning and how does it differ from centralised ML?

**Model answer:**
In centralised ML, all training data is collected on a single server and training runs there. In federated learning (FL), data stays on each client (hospital). In each round: (1) the server sends the current global model to all clients, (2) each client trains locally for a few epochs on its private data, (3) clients send model updates (gradients or weight deltas) back to the server, (4) the server aggregates (FedAvg: weighted average by dataset size) and broadcasts the updated global model. No raw data ever leaves the client. The privacy guarantee requires additional mechanisms (differential privacy) because raw gradients can leak information through gradient inversion attacks.

---

### Q2: What is differential privacy and what does ε = 1.0 mean?

**Model answer:**
(ε, δ)-differential privacy guarantees that for any two datasets D and D' differing in one record, and any subset of outputs S: P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ. Intuitively: the output distribution changes by at most a factor of e^ε when you add or remove any one person's data.

ε = 1.0 is considered a **strong privacy guarantee** in practice (e^1.0 ≈ 2.7× bound on the probability ratio). Lower ε = stronger privacy but more noise = less accuracy. The δ = 1e-5 is the probability of a "catastrophic failure" — by convention it should be less than 1/n where n is dataset size.

For our 5-hospital simulation with n = 10K records per hospital, δ = 1e-5 << 1/10000 = 1e-4 ✓

---

### Q3: How does Gaussian noise injection achieve differential privacy?

**Model answer:**
The Gaussian mechanism adds noise calibrated to the sensitivity of the function. For gradient descent, the sensitivity is bounded by **gradient clipping** (`max_grad_norm = 1.0`). The noise parameter σ is chosen using the DP accountant to achieve the target (ε, δ) over the planned number of training steps.

```python
from opacus import PrivacyEngine

model = MyNeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=num_epochs,
    target_epsilon=1.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
)
```

Opacus internally sets σ using the Rényi DP accountant and tracks cumulative privacy budget across all steps.

---

### Q4: What is the FedAvg algorithm and what are its limitations?

**Model answer:**
FedAvg (McMahan et al., 2017): Global model ← weighted average of client model updates, where weights = fraction of total samples on each client.

```
w_global = Σ_k (n_k / n) × w_k
```

**Limitations:**
1. **Non-IID convergence problems:** When clients have heterogeneous data distributions (different hospitals see different patient populations), client models drift apart and FedAvg diverges.
2. **Communication cost:** Full model uploads per round are expensive.
3. **Stragglers:** Slow clients block round completion.
4. **No privacy guarantee:** FedAvg alone doesn't prevent gradient inversion attacks.

FedProx addresses (1) by adding a proximal term μ||w - w_global||² to each client's local loss.

---

### Q5: What is gradient inversion and how serious a threat is it?

**Model answer:**
Gradient inversion (Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019) shows that an adversary with access to gradients can reconstruct the training samples by optimising a dummy input to match the observed gradients. The attack is most effective for small batch sizes and shallow networks. For large batches and deep networks, reconstruction becomes computationally infeasible but is still a theoretical risk. Differential privacy (specifically, the noise injection) defends against gradient inversion by making the gradients indistinguishable from a neighboring dataset's gradients.

---

### Q6: How does the Rényi DP accountant differ from basic composition?

**Model answer:**
**Basic composition:** If you run M mechanisms each with ε_i, the total ε = Σε_i. This is correct but very pessimistic — it assumes worst-case across all rounds simultaneously.

**Rényi DP (RDP):** Uses Rényi divergence of order α to track privacy loss. RDP composes much more tightly: multiple Gaussian mechanisms compose as RDP(α) = T × RDP_per_step(α), which converts to (ε, δ)-DP via a tighter bound. In practice, RDP accounting allows 3–5× more training rounds within the same ε budget compared to basic composition. This is why Opacus uses the RDP accountant by default.

---

### Q7: What is non-IID data and why is it the hardest problem in federated learning?

**Model answer:**
IID = independent and identically distributed. In federated learning, hospital A might primarily see elderly cardiac patients, hospital B sees pediatric cases, hospital C has a high prevalence of a rare disease. The local data distributions are non-IID — the feature and label distributions differ across clients. When FedAvg averages these heterogeneous local models, the global model can diverge or oscillate, especially with more local epochs.

**Mitigation strategies:**
1. **FedProx:** Adds proximal regularisation to pull local models toward the global model
2. **FedMA:** Model aggregation via matching neurons/filters instead of simple averaging
3. **Personalized FL:** Maintain per-client final layers while sharing the feature extractor
4. **Data sharing (small):** Share a tiny amount of representative data across clients — violates strict FL but improves convergence

---

### Q8: What is SMPC (Secure Multi-Party Computation) and why did you use it?

**Model answer:**
SMPC allows multiple parties to jointly compute a function over their private inputs without revealing those inputs to each other. In federated learning, the server aggregates gradients — but even the server shouldn't see individual client gradients (they can leak information about local data). Secure aggregation using SMPC means: each client masks its update with random values that cancel out in the sum. The server sees only the aggregate, not any individual client's update.

This is especially important for small federations (< 20 clients) where individual gradients carry more information about the client's dataset.

---

### Q9: How did you measure that your federated model is "within 2.1% of centralised"?

**Model answer:**
I trained a centralised baseline by pooling all client data and training normally (this violates FL assumptions but establishes the performance ceiling). Then I ran the federated training with FedAvg + FedProx + DP and compared AUC/accuracy on the same held-out test set from each modality. The gap was: pneumonia AUC 0.934 (centralised) vs 0.913 (federated) = 2.1% gap. This is within clinical acceptable bounds for a screening task (where recall matters more than absolute accuracy).

---

### Q10: What would you do differently if you had 6 more months?

**Model answer:**
1. **Real hospital data** — run the simulation on actual EHR data (with IRB approval) through a partnership, not public research datasets with simulated partitioning.
2. **Cross-silo → cross-device** — extend to wearable edge devices using a cross-device FL framework with async communication and client dropout handling.
3. **Personalised FL** — implement FedPer (federated personal layers) so each hospital's model adapts to its specific patient population while still benefiting from global knowledge.
4. **Formal verification** — move beyond empirical ε tracking to formally verify the DP guarantee using TensorFlow Privacy's DP-SGD audit tools.
5. **Privacy-utility Pareto curve** — run a hyperparameter sweep over ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0} and plot the trade-off to give stakeholders a concrete decision surface.
