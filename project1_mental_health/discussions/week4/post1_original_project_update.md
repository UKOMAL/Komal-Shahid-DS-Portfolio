# Week 4 Discussion — Post 1 (Original): Project 1 Findings & Reflection

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 4 | **Type:** Original Post (~300 words)

---

For my first project in DSC680 I analyzed the 2016 OSMI Mental Health in Tech Survey to predict whether a tech-industry employee seeks mental health treatment. This felt like a meaningful topic because mental health conversations in workplaces are often avoided, and data can help make the invisible visible.

The dataset includes 1,259 responses from tech workers globally. After cleaning and feature engineering, I built a composite variable called the Employment Support Index, which combines employer mental health benefits, willingness to discuss mental health with supervisors, and leave availability. The idea was to capture workplace environment as a single score rather than treating each variable separately.

I tested four classifiers — logistic regression, random forest, support vector machine, and XGBoost — using nested cross-validation to avoid data leakage. Logistic regression performed best with an AUC of 0.723. SHAP values revealed that work interference with daily tasks was the strongest predictor of treatment-seeking, followed by family history and the Employment Support Index.

What surprised me most was how much the workplace environment mattered beyond family history alone. People in more supportive workplaces were more likely to seek help, which suggests that stigma reduction is not just a personal responsibility but an organizational one. Supportive leave policies and open conversations with managers matter — and that is something employers can actually change.

I am curious whether others have found similar patterns when working with survey data — how do you handle response bias when the people most affected by a problem may be least likely to respond?

**Reference:**  
Osborne, J. W. (2008). *Best practices in quantitative methods*. SAGE Publications.
