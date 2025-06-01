# AI-Powered Depression Detection System

## Early Detection of Depression Through Digital Writing Analysis

**by: Komal Shahid**  
DSC680 - Applied Data Science

---

# Agenda

- Executive Summary
- The Challenge of Underdiagnosed Depression
- Research Objectives
- Methodology Overview
- Advanced Modeling Approach
- Key Findings
  - Word Frequency Analysis
  - Dimensional Understanding
  - Sentiment Distribution
  - Classification Performance
- Dataset Characteristics
- Ethical Implementation
- Advancing Depression Detection
- Conclusions

---

# Executive Summary

- Depression affects over 300 million people worldwide
- Significantly underdiagnosed condition
- AI-powered system to detect depression from writing patterns
- Fine-tuned RoBERTa transformer model achieved 78.5% accuracy
- Substantial improvement over traditional ML approaches
- Non-invasive, accessible screening tool for early intervention

---

# The Challenge of Underdiagnosed Depression

- Depression is one of the most common mental health disorders
- Average 4-year gap between onset and diagnosis
- Delayed diagnosis significantly impacts outcomes and quality of life
- Digital writing analysis enables early detection
- Linguistic patterns serve as reliable indicators of mental health

---

# Research Objectives

- Explore advanced NLP and deep learning for depression severity detection
- Identify linguistic markers most correlated with depression severity
- Determine optimal model architectures for classification performance
- Ensure ethical implementation in real-world contexts

---

# Methodology Overview

- Multi-phase approach combining NLP and transformer models
- Rigorous text preprocessing
- Comprehensive feature extraction
- Advanced model development
- Professionally labeled datasets with balanced severity representation

---

# Advanced Modeling Approach

- Initial experiments with traditional ML models showed moderate success
  - Accuracies ranging from 63.0% to 66.3%
- Transformer-based deep learning models achieved breakthrough performance
  - Fine-tuned BERT-base: 75.9% accuracy
  - Fine-tuned RoBERTa: 78.5% accuracy
- Substantial improvement over traditional approaches
- Underscores power of contextual language understanding

---

# Word Frequency Analysis

- Distinctive linguistic patterns across severity levels
- Common words like "just," "like," "feel" appear consistently
- Severe category uniquely contains expletives
- First-person pronouns appear with high frequency across categories

---

# Dimensional Understanding of Depression

- Depression exists on a continuum, not discrete categories
- Substantial overlap between severity categories in feature space
- Severe category shows most coherent clustering
- Dimensional perspective explains classification challenges
- Reinforces spectrum nature of depression severity

---

# Sentiment Distribution Patterns

- Subtle but meaningful patterns across severity levels
- All categories show predominantly negative sentiment
- Minimum category exhibits distinctive bimodal distribution
- Categories span full range from maximally negative to positive
- Sentiment alone insufficient for severity differentiation

---

# Classification Performance

- RoBERTa model shows strong performance, especially for minimum and severe categories
- Unexpected confusion between mild and severe categories
- Suggests potential linguistic similarities
- Opens avenues for research into nuanced language patterns
- Substantial confusion between adjacent categories expected

---

# Dataset Characteristics

- Balanced dataset with sufficient representation across severity categories
- Enables fair model evaluation and robust statistical analysis
- Approximates real-world prevalence while ensuring effective training
- Equal category sizes facilitate direct comparison of linguistic patterns

---

# Ethical Implementation Framework

- Responsible deployment requires careful consideration of:
  - Privacy
  - Bias
  - Potential harms
- Framework emphasizes:
  - Clear limitations disclaimers
  - Transparent model explanation
  - Strict data minimization
  - Professional oversight
  - Integration with existing healthcare infrastructure
  - AI as supportive tool within broader mental health ecosystem

---

# Advancing Depression Detection

- RoBERTa-based system represents significant advancement
- Applications in early screening, clinical support, longitudinal monitoring
- Future development areas:
  - Multimodal integration
  - Cultural adaptation
  - Enhanced interpretability
- Promising opportunities to address underdiagnosed depression globally
- Accessible, scalable, privacy-conscious approaches

---

# Conclusions

- AI can effectively detect depression indicators in text (78.5% accuracy)
- RoBERTa model outperformed other approaches
- Distinctive linguistic patterns across severity levels
- Substantial category overlap in feature space
- Promising tool for early screening
- Human expertise crucial for diagnosis

---

# Questions?

Thank you for your attention!

Code repository: https://github.com/UKOMAL/Depression-Detection-System

---

# Model Interpretability

- Attention visualization
  - Shows which words the model focuses on
  - Helps validate that meaningful phrases are attended to
- Feature importance analysis
  - Identifies most predictive features overall
  - Textual features and post metadata rank highly

---

# Interactive Demo

- Command-line tool to explore model behavior
  - Analyze individual texts
  - Predicted severity + confidence scores
  - Visualize attention weights
- Explore batch analysis results
  - Severity & sentiment distributions
  - Performance metrics
  - Feature importance 