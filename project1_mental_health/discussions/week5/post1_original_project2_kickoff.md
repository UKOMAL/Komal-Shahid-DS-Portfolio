# Week 5 Discussion — Post 1 (Original): Starting Project 2 & Lessons from Project 1

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 5 | **Type:** Original Post (~300 words)

---

Wrapping up Project 1 and moving into Project 2 has been a good moment to reflect on what actually made the first project work and what I would do differently. With the OSMI mental health survey, I leaned heavily on a composite feature I built called the Employer Support Index — combining four binary workplace variables into a single ordinal score. That engineering decision ended up being the second-strongest predictor in the model, which was satisfying because it came from domain thinking, not from running a feature importance function and picking the top ten. The model with that composite generalized better than the one without it.

The lesson I am carrying into Project 2 is that time spent on the problem definition and feature design phase pays off more than time spent on hyperparameter tuning. I spent a lot of energy in Project 1 on GridSearchCV and nested CV infrastructure, which was important for honest evaluation, but the AUC gap between a well-tuned model and a default one was maybe 0.02. The gap between a thoughtful feature matrix and a lazy one was much larger.

For Project 2, I am planning to work with a financial or fraud-related dataset — a domain where class imbalance is genuinely severe (sometimes 1% positive rate), and where the ethical stakes around model errors are high and asymmetric. A false negative in fraud detection costs money; a false positive costs customer trust. That asymmetry changes how you set thresholds, how you evaluate the model, and how you communicate results, which I think will make for an interesting contrast with Project 1.

What domains is everyone else exploring for Project 2? I am curious whether others are staying close to their Project 1 domain or making a deliberate jump.

**Reference:**  
Provost, F., & Fawcett, T. (2013). *Data science for business*. O'Reilly Media.
