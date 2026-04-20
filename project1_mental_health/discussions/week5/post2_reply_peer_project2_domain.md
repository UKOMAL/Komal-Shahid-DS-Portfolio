# Week 5 Discussion — Post 2 (Reply): Responding to a Classmate's Project 2 Domain Choice

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 5 | **Type:** Reply Post (~150 words)

---

Really interesting direction — healthcare claims data has a lot of the same structural challenges as fraud detection (severe imbalance, high cost of false negatives) but with a completely different ethical texture. The regulatory layer alone (HIPAA, CMS billing rules) makes feature selection non-trivial because some of the most predictive signals are also the most sensitive.

One thing worth thinking about early: with claims data, the label itself is often noisy. "Fraudulent" frequently means "flagged and confirmed after investigation," which means your training set reflects past investigator capacity, not the true underlying fraud rate. That label noise can distort your model in ways that are hard to audit after the fact.

Did you look at the Medicare Part D dataset from CMS? It is publicly available and has been used in several published fraud detection studies, so there is some benchmark literature to compare against.

**Reference:**  
Bauder, R. A., & Khoshgoftaar, T. M. (2018). Medicare fraud detection using machine learning methods. *2018 17th IEEE International Conference on Machine Learning and Applications*, 253–260.
