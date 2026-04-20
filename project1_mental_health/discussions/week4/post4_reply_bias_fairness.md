# Week 4 Discussion — Post 4 (Reply): Bias & Fairness in ML

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 4 | **Type:** Reply (~200 words)

---

[Reply to classmate's post about fairness and bias in machine learning]

The bias discussion you raised is one I find myself returning to constantly. It is easy to treat fairness as a post-hoc checklist item — run a demographic parity check at the end — but by the time you are evaluating a trained model, most of the bias-relevant decisions have already been made upstream in how data was collected and what features were included.

For my own project, I ran a fairness audit across age and gender groups after finalizing the model and the results were close enough that I did not flag a major concern. But I was also aware that the OSMI survey is a self-selected sample — the people who participated are already different from the broader tech workforce in ways I cannot fully account for. A model trained on self-reported data from motivated survey respondents may not generalize well to people who would never fill out a mental health survey in the first place.

That gap between who your data represents and who your model will be applied to is where a lot of real-world harm in ML tends to originate. Post-hoc fairness checks are useful, but they do not substitute for a sampling problem baked in at the data collection stage.
