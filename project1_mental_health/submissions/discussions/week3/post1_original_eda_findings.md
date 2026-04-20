# Week 3 Discussion — Post 1 (Original): EDA Surprises and Feature Engineering Decisions

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 3 | **Type:** Original Post (~300 words)

---

Now that I am deep into EDA on the OSMI mental health dataset, I want to share two findings that genuinely surprised me and changed the direction of my feature engineering.

The first surprise was how weak the individual workplace variables were on their own. When I ran Cramér's V between `seek_help`, `benefits`, `care_options`, and `wellness_program` individually against the treatment outcome, each one came in below V = 0.15 — technically significant but practically small. My initial instinct was to drop the weaker ones. Instead, I stepped back and asked whether these four variables might be capturing the same underlying construct: a general employer commitment to mental health support. I built a composite score (0–4, one point per binary indicator) that I am calling the Employer Support Index. The ESI as a composite has V = 0.31 — more than double any individual component. That is a meaningful lift, and it came from thinking about the data structurally rather than just running a feature importance scan.

The second surprise was the gender field. I expected two or three values. The raw data had 49 distinct free-text entries. "Male" and "male" and "Male " (trailing space) are three separate values in the raw file. There were also genuinely ambiguous responses that required judgment calls. I ended up with Male (77%), Female (20%), and Non-binary/Other (3%) — but I want to be upfront in the paper that this consolidation loses real granularity.

The broader lesson I am taking away: EDA is not just descriptive, it is prescriptive. What you find shapes what you build.

Anyone else finding that their initial feature list needs to be reconsidered after actually looking at the distributions?

**Reference:**  
Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society, 42*(3), 323–328.
