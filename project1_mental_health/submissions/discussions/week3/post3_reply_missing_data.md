# Week 3 Discussion — Post 3 (Reply): On Handling Missing Data

**Course:** DSC680-T301 Applied Data Science  
**Author:** Komal Shahid  
**Week:** 3 | **Type:** Reply Post (~150 words)

---

The missingness question is harder than it looks — you are right to flag it. In my dataset, `work_interfere` had about 18% missing, and the pattern was not random. Respondents who said they do not have a mental health condition were the ones most likely to skip it, which makes sense: if you do not have a condition, the question "does it interfere with work" is not really applicable. That is missing-not-at-random (MNAR), which means simple mean imputation would have introduced bias in a particular direction.

My solution was to encode missingness explicitly as its own category ("Does Not Apply") rather than imputing. This let the model learn that the absence of a response is itself informative. It is not a perfect solution — you are adding a category that is conceptually different from the others — but it is more honest than pretending those rows had an average level of work interference.

**Reference:**  
van Buuren, S. (2018). *Flexible imputation of missing data* (2nd ed.). CRC Press. https://stefvanbuuren.name/fimd/
