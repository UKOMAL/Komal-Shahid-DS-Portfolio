# DSC680 - Final Project Proposal
** Name:** Komal Shahid  
**Course:** DSC680 - Applied Data Science  
**Project 1:** AI-Powered Early Detection System for Depression from Digital Writing Patterns  
**Date:** March 17, 2024

## 1. Topic
**AI-Powered Early Detection System for Depression from Digital Writing Patterns**

This project will develop a machine learning system that analyzes linguistic patterns in digital writing (social media posts, messages, notes) to identify early warning signs of depression, providing timely intervention opportunities while maintaining privacy and ethical standards.

## 2. Business Problem
Depression often develops gradually with subtle linguistic changes that may be expressed through digital writing before becoming clinically apparent. Early detection and intervention can significantly improve outcomes, reduce healthcare costs, and potentially save lives. This project aims to:

- Identify specific linguistic patterns in digital writing that correlate with developing depression
- Develop privacy-preserving algorithms that can detect early warning signs of depression without compromising user confidentiality
- Create an ethical framework for intervention that balances early detection with user autonomy
- Design a system that could be integrated with existing mental health support services or employee wellness programs

**Research Questions:**
1. What specific linguistic patterns (word choice, sentence structure, emotional content) most strongly correlate with developing depression?
2. Can machine learning models accurately identify early warning signs of depression from writing patterns while minimizing false positives?
3. How can we develop privacy-preserving NLP algorithms that protect sensitive user data while still enabling effective detection?
4. What intervention frameworks are most effective and ethical when potential depression warning signs are detected?

## 3. Datasets
This project will utilize multiple anonymized and ethically sourced datasets focused specifically on depression:

1. **CLEF eRisk Depression Dataset**
   - Source: [Early Risk Detection on the Internet](https://early.irlab.org/2022/index.html)
   - Content: Social media posts for early detection of depression
   - Format: Text data with timestamps
   - Variables: Writing patterns, linguistic markers, temporal changes
   - Size: Thousands of posts from hundreds of users

2. **Depression Reddit Dataset**
   - Source: [University of Maryland Depression Dataset](https://github.com/md-dmr/Depression-Detection)
   - Content: Reddit posts from depression-related subreddits and control groups
   - Format: Text data with timestamps
   - Variables: Linguistic patterns, community interactions, temporal progression
   - Size: Over 500,000 posts from 9,000+ users

3. **Distress Analysis Interview Corpus**
   - Source: [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)
   - Content: Transcribed clinical interviews with depression assessments
   - Format: Text transcriptions with clinical scores
   - Variables: Linguistic patterns, clinical depression scores (PHQ-8)
   - Size: 189 sessions with validated clinical assessments

4. **The CLPsych Shared Task Dataset**
   - Source: [Computational Linguistics and Clinical Psychology](https://clpsych.org/shared-tasks/)
   - Content: Social media posts with depression annotations
   - Format: Annotated text data
   - Variables: Linguistic features, clinical annotations
   - Size: Thousands of annotated posts

## 4. Methods
This project will employ a focused analytical approach, leveraging concepts and techniques from multiple DSC courses:

### Exploratory Data Analysis
- Linguistic analysis of text data for depression markers (DSC520: Data Exploration and Analysis)
- Temporal analysis of writing pattern changes (DSC530: Statistical Methods for Data Science)
- Comparative analysis between depressed and non-depressed writing samples (DSC520, DSC530)

### Feature Engineering
- Extraction of linguistic markers associated with depression (pronoun usage, negative emotion words, absolutist thinking) (DSC550: Data Mining)
- Development of privacy-preserving features that capture writing patterns without exposing sensitive content (DSC540: Advanced Programming with Data)
- Creation of temporal change indicators to track progression (DSC550, DSC640: Data Visualization)

### Predictive Modeling
- Natural Language Processing for depression-specific linguistic pattern analysis (DSC550, DSC670: Applied AI and Deep Learning)
- BERT and transformer-based models for contextual understanding (DSC670)
- Time-series analysis for detecting changes in writing patterns over time (DSC630: Predictive Analytics)
- Ensemble methods combining multiple linguistic signals into stronger predictors (DSC630, DSC670)

### Ethical Framework Development
- Design of tiered intervention protocols based on confidence levels (DSC500: Data Science Essentials, DSC510: Introduction to Programming)
- Creation of explainable AI components to justify depression risk detection (DSC670)
- Development of user control mechanisms and transparency tools (DSC640, DSC510)

### Validation and Testing
- Cross-validation with clinical depression assessments (PHQ-9 scores where available) (DSC530, DSC630)
- Evaluation using clinical metrics (sensitivity, specificity, positive predictive value) (DSC530)
- Comparison with baseline clinical screening methods (DSC630)

## 5. Ethical Considerations
This project involves critical ethical dimensions specific to depression detection, drawing on principles from DSC500 (Data Science Essentials) and DSC630 (Predictive Analytics):

- **Privacy Protection**: Ensuring all analysis preserves user privacy through anonymization and secure processing
- **Informed Consent**: Developing frameworks for meaningful consent in depression monitoring
- **False Positives**: Balancing the harm of missed depression cases against unnecessary interventions
- **Stigmatization**: Avoiding reinforcement of stigma around depression
- **Autonomy**: Respecting user agency and control over their own data and intervention decisions
- **Clinical Boundaries**: Clearly defining the system as a screening tool, not a diagnostic replacement
- **Transparency**: Making detection criteria understandable to users and practitioners

## 6. Challenges/Issues
Anticipated challenges include:

- **Clinical Validity**: Ensuring the system correlates with clinically validated depression measures
- **Privacy Preservation**: Developing effective algorithms that don't compromise privacy
- **Linguistic Nuance**: Capturing subtle linguistic indicators of depression across different writing styles
- **Intervention Design**: Creating appropriate response frameworks that don't overreach
- **Cultural Differences**: Accounting for cultural variations in depression expression
- **Technical Integration**: Designing systems that could integrate with existing support services
- **Regulatory Compliance**: Navigating healthcare and data privacy regulations for depression screening

## 7. References

Yang, K., Ji, S., Zhang, T., Xie, Q., Kuang, Z., & Ananiadou, S. (2023). Towards interpretable mental health analysis with large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 6056-6077.

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. (2023). Large language models encode clinical knowledge. Nature, 620(7972), 172-180.

Pérez, A., Warikoo, N., Wang, K., Parapar, J., & Gurevych, I. (2023). Semantic similarity models for depression severity estimation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 16104-16118.

Hua, Y., Liu, F., Yang, K., Li, Z., Sheu, Y., Zhou, P., Moran, L. V., Ananiadou, S., & Beam, A. (2024). Large language models in mental health care: a scoping review. arXiv preprint arXiv:2401.02984.

Ji, S., Zhang, T., Yang, K., Ananiadou, S., Cambria, E., & Tiedemann, J. (2023). Domain-specific continued pretraining of language models for capturing long context in mental health. arXiv preprint arXiv:2304.10447.

Tejaswini, V., Sathya Babu, K., & Sahoo, B. (2024). Depression detection from social media text analysis using natural language processing techniques and hybrid deep learning model. ACM Transactions on Asian and Low-Resource Language Information Processing, 23(1), 1-20.

Wu, J., Wu, X., Hua, Y., Lin, S., Zheng, Y., & Yang, J. (2023). Exploring social media for early detection of depression in COVID-19 patients. arXiv preprint arXiv:2302.12044.

Guo, Y., Ding, Z., Jin, Y., Feng, Y., Zhang, Y., Qu, L., & Liu, Z. (2023). A prompt-based topic-modeling method for depression detection on low-resource data. IEEE Transactions on Computational Social Systems.

Nguyen, T., Yates, A., Zirikly, A., Desmet, B., & Cohan, A. (2022). Improving the generalizability of depression detection by leveraging clinical questionnaires. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 8446-8459.

Eichstaedt, J. C., Smith, R. J., Merchant, R. M., Ungar, L. H., Crutchley, P., Preoţiuc-Pietro, D., Asch, D. A., & Schwartz, H. A. (2018). Facebook language predicts depression in medical records. Proceedings of the National Academy of Sciences, 115(44), 11203-11208. 