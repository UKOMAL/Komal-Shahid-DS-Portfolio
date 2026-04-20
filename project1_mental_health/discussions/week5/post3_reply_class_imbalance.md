# Week 5 Discussion — Post 3 (Reply): On Class Imbalance Strategies

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 5 | **Type:** Reply Post (~150 words)

---

You raised something I've been thinking about a lot — whether SMOTE is actually the right tool when the imbalance is as extreme as 1% positive rate. In those cases, SMOTE generates synthetic minority samples in a feature space where the minority class is genuinely sparse, which means the synthetics can end up in ambiguous regions that don't reflect real fraud patterns. I have seen papers argue that undersampling the majority is actually cleaner in that regime because you are preserving real signal rather than interpolating it.

The other thing I keep coming back to is threshold calibration. Even if you leave the class balance alone during training, moving your decision threshold from 0.5 to something like 0.1 or 0.05 can recapture a lot of recall without needing resampling at all. Precision-recall AUC is probably a more honest metric than ROC-AUC for the 1% case anyway — ROC is too forgiving when the negative class is that large.

**Reference:**  
Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.
