# Week 3 Discussion — Post 4 (Reply): On Model Choice and Interpretability Trade-offs

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 3 | **Type:** Reply Post (~150 words)

---

The interpretability-performance trade-off you're describing is real, but I think it is worth separating two things: which model you use for prediction, and which model you use for explanation. They do not have to be the same one.

For my project, Logistic Regression ended up with the best AUC (0.723 ± 0.009) — not what I expected going in. But even if RF had won, I would still use LR coefficients and odds ratios as the primary stakeholder-facing explanation, because the audience is HR professionals and org leaders, not ML practitioners. An odds ratio of 2.31 for the Employer Support Index is immediately actionable. A SHAP beeswarm plot takes five minutes to explain to someone who hasn't seen one before.

That said, I did run SHAP on the Random Forest as a secondary explainability layer, specifically to check for interaction effects that the linear model would miss. It is worth the extra work because it gives you confidence that LR's clean story isn't hiding something the tree models are picking up.
