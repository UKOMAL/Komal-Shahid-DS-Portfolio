# Week 1 Discussion — Post 4 (Reply): Project Methods & Peer Review

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 1 | **Type:** Reply (~200 words)

---

[Reply to classmate sharing their project plan and asking for feedback]

Happy to swap feedback as you move into the draft. A few thoughts on what you described:

On model selection — if you are using multiple classifiers, I would lean toward showing a compact comparison table rather than leading with just one winner. It demonstrates that the choice was deliberate, not just convenient, and gives an instructor or stakeholder visibility into the trade-offs. For my own project I'm using nested cross-validation so the reported numbers are honest estimates of out-of-sample performance, which also makes the comparison more defensible.

On presenting results to a mixed audience — I have found that pairing every figure with a one-sentence "so what" caption makes a significant difference. It tells the reader what to take away without making them reverse-engineer it from the axis labels. SHAP beeswarm plots in particular need that context because most non-practitioners do not read them intuitively.

On feature engineering — I agree with your instinct to get the data dictionary clean before tuning. Garbage features going in means the hyperparameter search is optimizing on noise. Getting that foundation right first is worth the extra time upfront.

Would be glad to review a draft section when you have one ready.

**Reference:**  
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.
