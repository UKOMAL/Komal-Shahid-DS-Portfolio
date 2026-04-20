# Week 3 Discussion — Post 2 (Reply): On Feature Selection Approaches

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 3 | **Type:** Reply Post (~150 words)

---

The tension you described between filter methods and wrapper methods is something I ran into too. I started with filter-based selection (Cramér's V for categoricals, mutual information as a secondary check) and then let the model's nested CV performance tell me whether dropping a borderline feature actually helped or hurt. In most cases, dropping a feature with V < 0.10 did not move AUC more than 0.005, which suggests those variables really were noise at this sample size.

What I would push back on slightly is using p-values from chi-square tests as the primary selection criterion. With N=1,259, almost anything with even a weak association is going to hit p < 0.05. Effect size and p-value are telling you different things. I made that mistake in the first iteration and ended up with a model that had 31 features and basically the same AUC as the one with 22. The simpler model was easier to explain and only marginally worse — which, for a practitioner audience, matters a lot.
