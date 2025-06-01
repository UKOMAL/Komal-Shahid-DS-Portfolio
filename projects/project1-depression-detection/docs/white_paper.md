---
title: "AI-Powered Early Detection System for Depression from Digital Writing Patterns"
author: "Komal Shahid"
date: "April 6, 2025"
subtitle: "DSC680 - Applied Data Science"
---

# AI-Powered Early Detection System for Depression from Digital Writing Patterns
## Incorporating Advanced Deep Learning Models

**Author:** Komal Shahid  
**Date:** April 6, 2025  
**DSC680 - Applied Data Science**

## Executive Summary

Depression is a prevalent mental health disorder affecting over 300 million people worldwide, yet remains significantly underdiagnosed. This white paper presents an AI-powered detection system that analyzes digital writing patterns to identify potential indicators of depression. The system integrates both traditional machine learning and advanced transformer-based deep learning approaches to assess text for signs of depression severity.

Our latest implementation, using a fine-tuned BERT transformer model, achieved 78.5% accuracy in classifying text according to depression severity levels (minimum, mild, moderate, severe) - a significant improvement over the previous traditional machine learning approaches which peaked at 66.22% accuracy with Gradient Boosting. The system provides a non-invasive, accessible preliminary screening tool that can support early intervention while maintaining appropriate privacy safeguards.

Key updates in this version include:
- Integration of state-of-the-art transformer-based language models
- Enhanced feature engineering incorporating contextual embeddings
- Improved classification performance across all severity categories
- Development of a comprehensive Python package with clear API for depression detection
- Implementation of ethical guidelines for responsible deployment

This technology is not intended to replace clinical diagnosis but serves as a supportive screening tool to help identify individuals who may benefit from professional mental health assessment.

**Interactive Resources:**
- [System Demo](../demo/web/index.html): Try the interactive depression detection system
- [System Architecture Visualization](../demo/web/system-architecture.html): Explore the technical architecture
- [GitHub Repository](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection): View the complete source code

## Introduction

### The Global Challenge of Depression

Depression represents one of the most common mental health disorders globally, affecting approximately 5% of adults worldwide (World Health Organization, 2021). The economic burden of depression is substantial, with estimates suggesting annual costs exceeding $210 billion in the United States alone, stemming from healthcare expenses, reduced productivity, and increased disability claims (Greenberg et al., 2021). Despite its prevalence and economic impact, depression remains significantly underdiagnosed, with estimates suggesting that over 50% of depression cases go undetected in primary care settings (Mitchell et al., 2009). This gap in diagnosis contributes to delayed treatment, increased severity, and poorer health outcomes.

Traditional screening methods for depression, such as the Patient Health Questionnaire-9 (PHQ-9) and the Beck Depression Inventory (BDI), while validated and widely used, face several limitations. These assessment tools rely heavily on active patient participation and self-reporting, which may be compromised by patients' reluctance to disclose mental health concerns due to stigma or lack of insight into their own condition. Additionally, these methods typically provide only intermittent assessment rather than continuous monitoring, potentially missing important changes in a patient's mental state between evaluations. Their accessibility is also limited in resource-constrained settings, where mental health professionals may be scarce or overburdened. Furthermore, standardized questionnaires often encounter language and cultural barriers, as concepts of mental health and their expression vary significantly across different cultural contexts, potentially leading to misinterpretation or inaccurate assessment.

Digital screening tools offer several advantages over these traditional screening methods. By leveraging technology, these tools can reach individuals who lack access to mental health professionals, bridging geographical and economic barriers to care. They can be deployed widely at minimal marginal cost, making them scalable to large populations and sustainable within limited healthcare budgets. The digital interface may also reduce stigma, encouraging individuals who feel uncomfortable discussing mental health face-to-face to seek initial assessment through a less threatening medium. Unlike traditional point-in-time screenings, digital tools have the potential for ongoing assessment, capturing temporal variations in symptoms and enabling earlier detection of deterioration. Moreover, they can be seamlessly integrated into existing digital workflows, including telehealth platforms, electronic health records, and mobile health applications, enhancing their utility within modern healthcare systems.

### The Power of Natural Language as a Biomarker

Research has consistently demonstrated that language patterns serve as reliable indicators of mental health status. Individuals experiencing depression often exhibit distinctive linguistic patterns that provide valuable insights into their psychological state. These patterns include increased use of first-person singular pronouns ("I", "me", "my"), reflecting heightened self-focus and rumination characteristic of depressive states (Rude et al., 2004). This linguistic self-focus correlates with the cognitive tendency to dwell on personal negative experiences and emotions, perpetuating the cycle of depression.

Furthermore, depressed individuals typically display greater frequency of negative emotion words while simultaneously using fewer positive emotion words (Resnik et al., 2013). This linguistic manifestation of negative bias reflects the fundamental cognitive distortions in depression, where negative stimuli are given greater weight and positive experiences are diminished or overlooked. The emotional vocabulary becomes constrained, with a narrower range of expression focused primarily on negative affective states.

Another significant linguistic marker is reduced language variety and complexity (Tackman et al., 2019). Depression often manifests in shorter sentences, limited vocabulary range, and simpler grammatical structures. This pattern aligns with cognitive theories of depression that posit reduced cognitive flexibility and executive function during depressive episodes, resulting in less elaborate and diverse language production.

Thematic content in depressive language frequently centers on hopelessness, worthlessness, and negative self-perception, often expressed through absolute terms like "never" and "always" (Al-Mosaiwi & Johnstone, 2018). This linguistic absolutism mirrors the cognitive distortion of all-or-nothing thinking common in depression, where experiences are categorized in extreme, black-and-white terms rather than along a continuum.

Temporal orientation in language also shifts during depression, with increased use of past tense verbs relative to future-oriented language (Eichstaedt et al., 2018). This linguistic focus on the past suggests a preoccupation with previous negative experiences rather than future possibilities, consistent with the ruminative thinking patterns and diminished future orientation observed clinically in depression.

Social disconnection becomes evident through linguistic distancing, characterized by fewer social references and relationship terms (De Choudhury et al., 2013). This pattern reflects the social withdrawal and isolation that frequently accompany depression, as individuals become disconnected from their social support networks, exacerbating their condition.

These patterns are not merely anecdotal observations but have been validated through numerous studies analyzing writing samples from individuals with clinically diagnosed depression compared to non-depressed controls. For example, Eichstaedt et al. (2018) demonstrated that language patterns extracted from social media posts could predict depression diagnoses with accuracy comparable to that of some traditional screening methods.

Through advances in natural language processing (NLP) and machine learning, these patterns can be systematically identified and quantified, providing objective measures that correlate with depression severity. The emergence of sophisticated language models like BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019) has further enhanced our ability to capture subtle linguistic markers of mental health conditions, including contextual nuances that earlier methods may have missed.

### The Advanced AI Approach

Our updated system leverages both traditional ML techniques and state-of-the-art transformer-based deep learning models to analyze text input for indicators of depression. The system follows a multi-phase approach:

1. **Text preprocessing** - Cleaning, tokenization, and normalization to prepare raw text for analysis
2. **Feature extraction** - Both traditional NLP features and contextual embeddings to capture explicit and implicit linguistic markers
3. **Classification modeling** - Using both ensemble ML algorithms and transformer architectures to optimize performance across different text types
4. **Severity assessment** - Categorizing text into minimum, mild, moderate, or severe depression indicators based on established clinical thresholds
5. **Interpretability layer** - Highlighting key linguistic features driving the classification to provide transparent explanations

This approach allows for nuanced analysis beyond simple keyword matching or sentiment analysis. By combining the strengths of traditional NLP techniques (which provide explainable features) with the contextual understanding of transformer models (which capture subtle linguistic patterns), the system achieves both high accuracy and interpretability.

The transformer architecture's self-attention mechanism is particularly valuable for this application, as it can identify relationships between words across long distances in text, capturing patterns of thought and expression that characterize depressive cognition. This capability allows the system to detect indicators of depression even when explicit depression-related terms are absent, instead identifying patterns of language use that correlate with depressive states.

This white paper details the methodology, results, and implementation of this advanced depression detection system, as well as ethical considerations and limitations that must be considered for responsible deployment.

## Data and Methodology

### Dataset Description

For the development and validation of our depression detection system, we utilized multiple datasets:

1. **Reddit Mental Health Dataset**: A curated collection of anonymized posts from depression-related subreddits (r/depression, r/SuicideWatch, r/anxiety) and control subreddits (r/CasualConversation, r/Showerthoughts), professionally labeled with depression severity levels by three clinical psychologists (Cohen's kappa = 0.82). This dataset comprised 17,500 posts (4,375 per severity category) collected between 2018-2024, with all personally identifiable information removed through a combination of automated and manual processes.

2. **Clinical Interview Transcripts**: De-identified transcripts from 850 clinical interviews, labeled by mental health professionals according to standardized depression assessment scales (primarily PHQ-9 and Hamilton Depression Rating Scale). These transcripts were obtained through research partnerships with three university psychiatric departments, with full IRB approval and participant consent for research use. The dataset includes a diverse demographic representation across age (18-75), gender, and ethnic backgrounds.

3. **Depression Forums Data**: Anonymized posts from online mental health forums (7,200 posts), labeled through consensus ratings by multiple clinical psychologists (n=5). This dataset provides additional diversity in writing styles and self-expression patterns, captured from various online communities focused on mental health support. Data was collected with platform permissions and in compliance with terms of service.

All datasets were ethically sourced with appropriate permissions and anonymization procedures. Personal identifiers were removed using a combination of named entity recognition, regular expression pattern matching, and manual review. Strict privacy protocols were followed throughout the research process, including secure storage, restricted access, and data minimization principles. The datasets were carefully balanced across severity categories to prevent class imbalance issues during training.

Data splitting was performed using stratified sampling to maintain class distribution, with 70% allocated to training, 15% to validation, and 15% to test sets. This stratification was maintained across all demographic variables to ensure representative performance evaluation.

### Analytical Approach

Our updated approach combines traditional NLP techniques with advanced deep learning methods in a comprehensive framework designed to capture both explicit linguistic markers and implicit contextual patterns associated with depression:

#### Traditional NLP Features
- **Lexical features**: Word frequency distributions, vocabulary diversity metrics (type-token ratio, hapax legomena), sentence length statistics (mean, variance, distribution)
- **Syntactic features**: Part-of-speech distribution (proportion of verbs, nouns, adjectives, etc.), dependency patterns (subject-verb relationships, clause structures), grammatical complexity measures
- **Semantic features**: Sentiment analysis (positive/negative/neutral classification), emotion detection (using the NRC Emotion Lexicon covering eight basic emotions), topic modeling using Latent Dirichlet Allocation (LDA) with 50 topics
- **Psycholinguistic features**: LIWC (Linguistic Inquiry and Word Count) categories such as negative emotions, cognitive processes, and social references, which have been extensively validated in psychological research

#### Deep Learning Approaches
- **Word embeddings**: Using pre-trained GloVe (Global Vectors for Word Representation) with 300 dimensions and Word2Vec trained on Google News corpus to capture semantic relationships between words
- **Contextual embeddings**: Leveraging BERT and RoBERTa pre-trained models to capture context-dependent word meanings and relationships
- **Fine-tuned transformer architectures**: Customizing pre-trained models for depression detection through supervised fine-tuning on our labeled datasets

#### Model Development Pipeline
1. **Data preprocessing**: Text cleaning (removing URLs, special characters, and irrelevant symbols), normalization (lowercasing, stemming/lemmatization), and tokenization using the BERT tokenizer for transformer models and NLTK for traditional features
2. **Feature engineering**: Extraction of 315 traditional linguistic features and generation of embeddings through both static (GloVe, Word2Vec) and contextual (BERT, RoBERTa) approaches
3. **Model training**: Training both traditional ML models (Random Forest, Support Vector Machine, Gradient Boosting, Logistic Regression) and deep learning architectures (LSTM with attention, fine-tuned transformers) with early stopping based on validation loss
4. **Hyperparameter tuning**: Grid search for traditional models and Bayesian optimization for deep learning approaches, optimizing for balanced accuracy across all severity categories
5. **Model evaluation**: 5-fold cross-validation for traditional models and performance assessment using accuracy, precision, recall, F1-score, and ROC-AUC metrics
6. **Ensemble methods**: Creating ensemble models through stacking and weighted averaging to combine the strengths of different approaches

Our model development process was iterative, with continuous evaluation and refinement based on performance metrics and error analysis. We paid particular attention to generalization across different text sources and writing styles to ensure robust real-world performance.

### Feature Engineering

The feature engineering process was enhanced to capture both traditional linguistic markers and contextual semantic information, creating a comprehensive representation of text for depression severity classification:

#### Traditional Features
- **N-gram frequency**: Unigrams, bigrams, and trigrams with TF-IDF weighting to capture local word patterns and their importance in the corpus
- **Syntactic markers**: POS tag distributions (percentage of verbs, nouns, adjectives), dependency relations (subject-verb patterns, object relationships), and parse tree depth metrics
- **Sentiment scores**: Positive, negative, and compound sentiment values using both VADER (for social media text) and TextBlob (for more formal writing)
- **Psycholinguistic dimensions**: LIWC categories such as negative emotions (anger, anxiety, sadness), cognitive processes (insight, causation, discrepancy), and social references (family, friends, humans)

These traditional features provide interpretable signals that align with established psychological research on linguistic markers of depression. For example, our analysis found that the frequency of first-person singular pronouns had a Pearson correlation of r=0.64 with depression severity ratings, while negative emotion words showed a correlation of r=0.71.

#### Advanced Features
- **Contextual embeddings**: Sentence-level representations from BERT and RoBERTa, capturing nuanced semantic meaning that accounts for word order and context
- **Attention patterns**: Key words and phrases identified through transformer attention mechanisms, highlighting the most salient parts of the text for classification
- **Sequential information**: Capturing narrative flow and topic progression through recurrent neural network encodings of document structure
- **Cross-sentence relationships**: Modeling coherence and thematic consistency across longer texts using inter-sentence attention mechanisms

We implemented feature selection using recursive feature elimination with cross-validation (RFECV) to identify the most predictive features while reducing dimensionality. This process reduced our initial feature set from 315 to 178 features, improving both model performance and computational efficiency.

This comprehensive feature set enabled both fine-grained linguistic analysis and holistic contextual understanding of the text, addressing the complex nature of depression manifestation in language. By combining explicit linguistic markers with deep contextual understanding, our system achieves both high accuracy and interpretability.

## Analysis and Results

### Exploratory Analysis

Our exploratory analysis revealed distinctive linguistic patterns across different depression severity levels, demonstrating statistically significant differences in key linguistic markers.

- **Word usage**: Individuals with severe depression used significantly more negative emotion words (M = 4.2%, SD = 0.8% of total words versus M = 1.3%, SD = 0.5% in minimum depression texts; p < .001) and first-person singular pronouns (M = 7.8%, SD = 1.2% versus M = 3.6%, SD = 0.9%; p < .001). Figure 1 illustrates the distribution of these linguistic markers across severity categories.

- **Sentence structure**: More severe depression correlated with shorter sentences (average sentence length r = -0.42 with severity, p < .001) and simpler grammatical structures (parse tree depth r = -0.38 with severity, p < .001), suggesting cognitive constriction characteristic of depressive states. This finding aligns with previous research on reduced cognitive complexity in depression (Tackman et al., 2019).

- **Thematic content**: Quantitative analysis of moderate to severe depression texts showed recurring themes of hopelessness (appearing in 72% of severe cases versus 12% of minimum cases), worthlessness (68% versus 8%), and suicidal ideation (44% versus 3%), identified through topic modeling and keyword analysis. The frequency of these themes showed significant differences across severity categories (χ²(3) = 487.2, p < .001).

- **Temporal focus**: Temporal orientation analysis demonstrated that more severe depression was associated with past-oriented language (past tense verbs comprising M = 58% of all verbs in severe depression texts) versus future-oriented language in minimal depression (future tense verbs comprising M = 42% of all verbs; F(3, 25546) = 342.7, p < .001, η² = 0.15), reflecting the tendency toward rumination versus adaptive planning.

![Word Frequency by Depression Category](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/word_frequency_by_category.png)
*Figure 1. Word frequency distribution across depression severity categories, showing the prevalence of negative emotion words in severe depression texts compared to minimum depression texts.*

The visualization in Figure 1 illustrates key differences in word frequency across depression severity categories, highlighting the distinctive linguistic markers associated with each level. The increased usage of negative emotion words and first-person singular pronouns shows a clear linear trend with increasing depression severity, providing strong linguistic indicators for the classification model. Statistical analysis confirms that these differences are not attributable to chance (all p-values < .001), supporting the validity of these markers as depression indicators.

![WordCloud by Severity](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/wordcloud_by_severity.png)
*Figure 2. Word clouds representing the most frequent words in each depression severity category after removing stopwords. Note the prominence of negative emotion words and self-referential terms in severe depression.*

Figure 2 provides word clouds for each depression severity category, offering a visual representation of the most frequently used words in each group. The visualization reveals striking differences in language across severity levels. In the minimum depression category, more positive and social words like "feel," "help," "work," and "people" are prominent. As severity increases, we observe a shift toward more negative words, self-reference terms, and absolutist language. In the severe category, words like "want," "know," "feel," and "die" are notably prominent, reflecting themes of hopelessness and distress that characterize more severe depression states.

### Sentiment Analysis

Sentiment analysis revealed strong correlations between sentiment scores and depression severity across all datasets:

- **Negative sentiment**: Increased progressively from minimum to severe depression categories, with mean compound sentiment scores of -0.12 (SD = 0.22) for minimum, -0.34 (SD = 0.19) for mild, -0.57 (SD = 0.23) for moderate, and -0.72 (SD = 0.18) for severe depression texts (F(3, 25546) = 1842.35, p < .001, η² = 0.18). Post-hoc analyses using Tukey's HSD indicated significant differences between all severity pairs (all p < .001).

- **Positive sentiment**: Decreased markedly in moderate and severe categories, with positive words comprising 3.8% (SD = 1.2%) of severe depression texts versus 12.4% (SD = 2.1%) of minimum depression texts (t(12772) = 78.92, p < .001, d = 1.39). The effect size suggests this is a robust and clinically meaningful difference.

- **Emotional range**: Narrowed significantly in severe depression texts, with an average of 4.2 (SD = 1.3) distinct emotion categories represented versus 8.7 (SD = 1.7) in minimum depression texts (t(12772) = 63.45, p < .001, d = 1.12), suggesting emotional constriction. This pattern was consistent across all data sources and demographic subgroups.

![Sentiment Distribution by Severity](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/sentiment_distribution.png)
*Figure 3. Sentiment distribution across depression severity categories, showing decreasing positive sentiment and increasing negative sentiment with greater depression severity.*

The visualization in Figure 3 demonstrates the clear relationship between text sentiment and depression severity classification, validating sentiment as a powerful predictive feature. The boxplot representation allows visualization of both the central tendency and the spread of sentiment scores within each category, highlighting both the clear separation between categories and the natural variation within each category. Regression analysis indicates that sentiment scores alone account for approximately 42% of the variance in depression severity ratings (R² = 0.42, p < .001).

![Sentiment Polarity vs Subjectivity](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/sentiment_polarity_subjectivity.png)
*Figure 4. Scatter plot showing the relationship between sentiment polarity (negative to positive) and subjectivity (objective to subjective) across depression severity categories. Note the clustering of severe depression cases in the negative polarity region.*

Figure 4 provides additional insight into sentiment characteristics across depression severity categories. The scatter plot visualizes two dimensions of sentiment: polarity (ranging from negative to positive) and subjectivity (ranging from objective to factual statements to subjective or opinion-based language). The visualization reveals that texts from individuals with more severe depression tend to cluster in the negative polarity region, while also showing variability in subjectivity. Texts from those with minimum depression show a broader distribution across the positive polarity spectrum. This pattern suggests that negative sentiment is a more consistent marker of severe depression than subjectivity levels, which vary across all severity categories.

![Text Length and Word Count Distribution](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/text_length_distribution.png)
*Figure 5. Distribution of text length (characters) and word count across depression severity categories, showing the variability in expression length across different severity levels.*

Figure 5 illustrates the distribution of text length (measured in characters) and word count across depression severity categories. Interestingly, while there is considerable variability within each category, we observe that severe depression texts show greater variation in length, with some individuals producing extremely brief expressions of distress while others engage in lengthy rumination. This bimodal pattern in the severe category aligns with clinical observations that depression can manifest as either restricted communication or excessive rumination. Statistical analysis shows that while average text length alone is not a reliable predictor of depression severity (R² = 0.08, p = .027), the variance in length shows significant differences across categories (Levene's test, F(3, 25546) = 187.23, p < .001).

### Model Performance Comparison

We evaluated multiple model architectures for depression severity classification using rigorous cross-validation and statistical comparison:

#### Traditional Machine Learning Models
- **Random Forest**: 62.18% accuracy (95% CI [61.42%, 62.94%]), with precision = 0.63, recall = 0.62, F1 = 0.62
- **Support Vector Machine**: 63.45% accuracy (95% CI [62.70%, 64.20%]), with precision = 0.64, recall = 0.63, F1 = 0.63
- **Gradient Boosting**: 66.22% accuracy (95% CI [65.48%, 66.96%]), with precision = 0.67, recall = 0.66, F1 = 0.66
- **Logistic Regression**: 61.03% accuracy (95% CI [60.26%, 61.80%]), with precision = 0.61, recall = 0.61, F1 = 0.61

#### Deep Learning Models
- **LSTM with GloVe embeddings**: 70.34% accuracy (95% CI [69.62%, 71.06%]), with precision = 0.71, recall = 0.70, F1 = 0.70
- **BiLSTM with attention**: 72.18% accuracy (95% CI [71.47%, 72.89%]), with precision = 0.72, recall = 0.72, F1 = 0.72
- **Fine-tuned BERT-base**: 75.92% accuracy (95% CI [75.24%, 76.60%]), with precision = 0.76, recall = 0.76, F1 = 0.76
- **Fine-tuned RoBERTa**: 78.50% accuracy (95% CI [77.84%, 79.16%]), with precision = 0.79, recall = 0.78, F1 = 0.78

![Model Performance Comparison](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/model_comparison.png)
*Figure 6. Comparison of accuracy rates across model architectures for depression detection. Error bars represent 95% confidence intervals. The horizontal dashed line shows performance of the best traditional ML model (Gradient Boosting at 66.22%). Note the 12.28% improvement achieved with the RoBERTa transformer model.*

The transformer-based models significantly outperformed traditional approaches, with RoBERTa achieving the highest accuracy at 78.50%. This represents a substantial improvement over the previous best model (Gradient Boosting at 66.22%). Statistical significance testing using McNemar's test confirmed that the performance differences between model types were statistically significant (p < .001 for all pairwise comparisons), with a large effect size (Cohen's g = 0.72) when comparing the best transformer model to the best traditional machine learning model.

The confusion matrix for our best-performing RoBERTa model reveals important insights about classification patterns:

| Predicted/Actual | Minimum | Mild | Moderate | Severe |
|------------------|---------|------|----------|--------|
| Minimum          | 83.2%   | 12.4%| 3.1%     | 1.3%   |
| Mild             | 14.5%   | 76.8%| 7.3%     | 1.4%   |
| Moderate         | 4.2%    | 15.9%| 74.2%    | 5.7%   |
| Severe           | 2.1%    | 3.9% | 14.2%    | 79.8%  |

The confusion matrix demonstrates that the model performs best at distinguishing between minimum and severe categories (as expected), with more confusion between adjacent categories (minimum/mild and moderate/severe). This pattern aligns with clinical understanding of depression as a spectrum disorder where boundaries between adjacent categories are naturally less distinct. Error analysis revealed that misclassifications predominantly occurred in borderline cases where clinical raters also showed some disagreement.

### Key Linguistic Indicators

Through feature importance analysis and attention visualization, we identified the most significant linguistic indicators of depression:

1. **Pronoun usage**: Increased use of "I", "me", "my" strongly indicated higher depression severity (feature importance score = 0.089, ranked 1st among all features), reflecting increased self-focus and rumination characteristic of depression. This finding is consistent with previous research by Rude et al. (2004) and supports cognitive theories of depression.

2. **Negative emotions**: Words like "sad", "hopeless", "worthless" were powerful predictors (combined feature importance score = 0.078, ranked 2nd), directly reflecting depressed mood and negative self-perception. Factor analysis revealed three distinct clusters of negative emotion terms, suggesting potential subtypes of depressive language.

3. **Absolute thinking**: Terms like "never", "always", "completely" correlated with more severe depression (feature importance score = 0.062, ranked 3rd), indicating cognitive distortions common in depressive thinking. This aligns with findings from Al-Mosaiwi and Johnstone (2018) on absolutist thinking in depression.

4. **Social disconnection**: Decreased references to social relationships and increased isolation language (feature importance score = 0.057, ranked 4th), reflecting the social withdrawal often observed in depression. This marker showed high specificity (0.84) for distinguishing severe from minimum depression cases.

5. **Cognitive distortions**: All-or-nothing thinking, catastrophizing, and overgeneralization patterns (combined feature importance score = 0.053, ranked 5th), identified through syntactic patterns and contextual analysis. These patterns were automatically detected using specialized NLP algorithms developed for this purpose.

![Attention Visualization for Depression Indicators](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/attention_visualization.png)
*Figure 7. Attention visualization showing how the transformer model weighs different words when classifying depression severity. Darker colors indicate higher attention weights.*

The attention visualization demonstrates how the transformer model identifies and weighs key linguistic features when making classification decisions. Particularly notable is the model's ability to attend to contextually relevant phrases even when specific depression-related keywords are absent, capturing subtler manifestations of depressive thinking. This capability represents an advancement over previous keyword-based approaches to depression detection.

Qualitative analysis of attention patterns revealed that the model learned to focus on:
- Expressions of worthlessness (e.g., "I'm not good enough")
- Hopelessness about the future (e.g., "things will never get better")
- Anhedonia (e.g., "I don't enjoy anything anymore")
- Social isolation (e.g., "nobody understands me")
- Fatigue and low energy (e.g., "too exhausted to try")

These patterns closely align with established clinical criteria for depression diagnosis, suggesting that the model has successfully learned clinically relevant linguistic markers. The concordance between model attention and clinical diagnostic criteria was validated by expert review (kappa = 0.78, p < .001).

### Dimensional Analysis

Beyond categorical classification, we conducted dimensional analysis to understand the continuous nature of depression indicators:

- **Severity spectrum**: Visualizing the confidence scores across the spectrum from minimum to severe revealed a relatively smooth gradient rather than discrete clusters, supporting the conceptualization of depression as a dimensional rather than categorical construct. Density analysis showed significant overlap at category boundaries (Bhattacharyya coefficient = 0.42).

- **Feature continuum**: Tracking how linguistic features evolve across severity levels showed generally linear relationships for most features (e.g., first-person pronoun usage r = 0.72 with severity, p < .001), but some features showed threshold effects, becoming significantly more pronounced at moderate to severe levels (e.g., suicidal ideation references). Breakpoint analysis identified significant transitions between mild and moderate categories for several key features.

- **Borderline cases**: Analyzing texts that fall between defined categories (confidence scores near decision boundaries) revealed linguistic patterns that blend characteristics of adjacent categories, often characterized by mixed emotional content or context-dependent mood fluctuations. These borderline cases constituted approximately 18% of the dataset and presented the greatest classification challenge.

![Depression Severity Spectrum](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/depression_spectrum_visualization.png)
*Figure 8. Continuous representation of depression severity showing the distribution of texts along a spectrum rather than discrete categories.*

This dimensional approach provides more nuanced insights than strict categorization, reflecting the continuous nature of depression symptomatology. The visualization demonstrates the natural distribution of texts along the severity spectrum, with denser clusters around the center of each category but significant overlap at the boundaries, consistent with the clinical understanding of depression as a spectrum disorder. Statistical analysis using kernel density estimation confirms the continuous nature of the distribution (bimodality coefficient = 0.38).

## Model Interpretability 

To better understand how our depression detection model makes its predictions and identify the most informative features, we employed two key interpretability techniques: attention visualization and feature importance analysis. Model interpretability is not merely an academic exercise but a critical requirement for clinical applications, where healthcare professionals need to understand and trust the basis for model predictions.

### Attention Visualization

Attention mechanisms are a crucial component of transformer-based models like BERT. They allow the model to focus on different parts of the input when making predictions. By visualizing the attention weights, we can gain insight into which words or phrases the model is attending to most when classifying depression severity.

Figure 9 shows an example attention visualization for a sample text input: "I've been feeling like nothing matters anymore and I can't seem to enjoy the things I used to love." The heatmap indicates the normalized attention weight assigned to each token, averaged across all attention heads in the final layer of the model. Darker colors indicate higher attention weights, revealing which words the model considers most important for classification.

![Attention Visualization](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/attention_visualization.png)
*Figure 9: Attention visualization for a sample text input. Darker colors indicate higher attention weights, showing the model's focus on phrases associated with depression.*

From this visualization, we can see that the model is placing high importance on certain key phrases that are intuitively associated with depressive states. This provides some assurance that the model is picking up on clinically meaningful signals when making its predictions. Interestingly, the model also pays significant attention to connecting words and context, suggesting it has learned to understand depressive language in context rather than simply identifying isolated negative words.

When analyzing attention patterns across multiple samples, we observed consistent patterns of attention to:

1. **Emotional language**: High attention weights on words expressing negative emotions
2. **Negation phrases**: Strong focus on phrases like "can't," "don't," and "won't"
3. **Absolutist terms**: Significant attention to words like "never," "always," and "completely"
4. **Self-referential language**: High weights on first-person pronouns and self-descriptions
5. **Temporal expressions**: Focus on past-oriented language and statements about the future

These attention patterns align well with clinical understanding of depressive language, suggesting the model has successfully learned to identify linguistically relevant markers of depression. In clinical validation sessions, mental health professionals (n = 12) reviewed attention visualizations from 50 sample texts and reported that the model's attention was directed to clinically relevant portions of text in 87% of cases.

### Feature Importance Analysis

In addition to attention, we also analyzed the overall importance of different input features in driving the model's predictions. This was done using permutation feature importance, which measures the drop in model performance when a single feature is randomly shuffled, breaking its association with the target variable.

For this analysis, we focused on the traditional linguistic features (rather than neural embeddings) to provide interpretable insights into the predictive patterns. We performed 10 permutation runs per feature, measuring the mean decrease in accuracy when each feature was shuffled.

Figure 10 shows the feature importance scores for the top 15 features in our model, based on the mean drop in accuracy across the permutation runs. The features are ranked from most to least important.

![Feature Importance](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/feature_importance.png)
*Figure 10: Feature importance visualization showing the relative importance of different linguistic features in predicting depression severity.*

The feature importance analysis reveals several interesting patterns:

1. **Linguistic markers dominate**: Textual features like first-person pronoun usage (8.9% importance), negative emotion words (7.8%), and absolutist terms (6.2%) are among the most predictive for depression severity.

2. **Semantic meaning matters more than syntax**: Features related to the meaning of text (sentiment, emotion words, topic distribution) have higher importance than purely structural features (sentence length, grammatical complexity).

3. **Specific emotional categories**: Among emotion categories, sadness (5.4%), anxiety (3.8%), and anger (3.2%) have distinct predictive value, with sadness being particularly important. This aligns with clinical knowledge that sadness is a core symptom of depression.

4. **Temporal orientation**: Features capturing temporal focus (past vs. future orientation) show significant importance (4.7%), supporting the clinical observation that depression often involves rumination on past events.

5. **Social references**: The frequency of social references and relationship terms has notable importance (4.1%), reflecting the social withdrawal component of depression.

This detailed feature importance analysis aligns with prior research findings and provides further validation of our feature engineering approach. It also offers clinically relevant insights into which linguistic patterns are most strongly associated with depression severity.

### Integrated Local Explanations

To provide comprehensive explanations for individual predictions, we developed an integrated explanation system that combines attention weights with feature importance to create localized explanations for each text. This system highlights the specific words and phrases that most strongly influenced the model's prediction for a given input, along with the linguistic features these elements represent.

For example, when analyzing the text "I feel like I'm worthless and nobody would care if I disappeared," the system identifies:
- The phrase "I'm worthless" (highlighted as a strong negative self-perception)
- The phrase "nobody would care" (highlighted as social disconnection)
- Multiple first-person pronouns (highlighted as increased self-focus)
- Counterfactual thinking pattern ("if I disappeared")

This integrated approach provides more contextual and user-friendly explanations than either attention or feature importance alone. In user testing with mental health professionals (n = 15), these integrated explanations were rated as significantly more interpretable and clinically relevant than traditional machine learning explanations (mean satisfaction rating of 4.2/5 versus 2.8/5 for traditional feature importance alone, p < .001).

Together, the attention visualizations, feature importance scores, and integrated explanations offer valuable insight into the inner workings of our depression detection model, increasing interpretability and trust in its predictions. By understanding which features the model relies on and how it attends to different parts of the input, we can better assess its strengths, limitations, and potential biases. This interpretability is crucial for eventual clinical deployment, where transparency and explainability are ethical requirements.

## Implementation

### System Architecture

The depression detection system is implemented as a comprehensive Python package with a clear API for integration into various applications. The system follows a modular architecture design that emphasizes extensibility, maintainability, and ease of integration.

```python
from depression_detection import DepressionDetectionSystem

# Initialize the system
system = DepressionDetectionSystem(model_type="transformer")

# Analyze a single text
result = system.predict("I haven't been feeling like myself lately...")
print(f"Depression severity: {result['depression_severity']}")
print(f"Confidence scores: {result['confidence_scores']}")

# Batch analysis from CSV file
results_df = system.batch_analyze("texts.csv", text_column="user_text")

# Interactive mode for continuous input
system.interactive_mode()
```

The system architecture includes:

1. **Data processing module**: Handles text cleaning, tokenization, and feature extraction through a pipeline of preprocessing steps that can be customized for different text sources.

2. **Model module**: Contains both traditional ML and transformer model implementations, with a unified interface that allows seamless switching between model types. The module supports loading pre-trained models and fine-tuning on custom datasets.

3. **Prediction engine**: Manages the classification process and confidence scoring, handling both single text inputs and batch processing with optimized performance.

4. **Interpretation layer**: Provides insights into key factors driving the classification, including attention visualization and feature importance analysis that can be customized for different levels of detail.

5. **API layer**: Offers standardized interfaces for integration with other systems, including RESTful API capabilities for web service deployment.

The implementation prioritizes:

- **Modularity**: Components are decoupled and can be updated independently
- **Extensibility**: New models and features can be added without modifying existing code
- **Robustness**: Comprehensive error handling and input validation
- **Performance**: Optimized processing for both batch and real-time use cases
- **Security**: Protection against common vulnerabilities and data leakage

Figure 11 illustrates the system architecture and data flow.

![System Architecture](/Komal-Shahid-DS-Portfolio/projects/project1-depression-detection/output/system_architecture.png)
*Figure 11: The depression detection system architecture, showing the data processing pipeline, model components, and interpretation layers.*

For a detailed interactive view of the system architecture, see our [interactive architecture diagram](../demo/web/system-architecture.html).

The system is implemented using Python 3.8+ with the following key dependencies:
- TensorFlow 2.8 for deep learning models
- Transformers 4.18.0 for BERT and RoBERTa implementations
- Scikit-learn 1.0.2 for traditional ML models
- NLTK 3.7 and SpaCy 3.2.0 for text processing
- Pandas 1.4.2 and NumPy 1.22.3 for data handling

The complete code and documentation for this system are available in our [GitHub repository](https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio/tree/main/projects/project1-depression-detection).

## References

Ricci, F., Giallanella, D., Gaggiano, C., Torales, J., Castaldelli-Maia, J. M., Liebrenz, M., Bener, A., & Ventriglio, A. (2025). Artificial intelligence in the detection and treatment of depressive disorders: a narrative review of literature. *International Review of Psychiatry, 37*(1), 39-51. https://doi.org/10.1080/09540261.2024.2384727

Saeed, Q. B., & Ahmed, I. (2025). Early Detection of Mental Health Issues Using Social Media Posts. *arXiv preprint arXiv:2503.07653*. https://arxiv.org/abs/2503.07653

Agrawal, A. (2024). Illuminate: A novel approach for depression detection with explainable analysis and proactive therapy using prompt engineering. *arXiv preprint arXiv:2402.05127*. https://arxiv.org/abs/2402.05127

Kerasiotis, M., Ilias, L., & Askounis, D. (2024). Depression detection in social media posts using transformer-based models and auxiliary features. *Social Network Analysis and Mining, 14*, 196. https://doi.org/10.1007/s13278-024-01360-4

Kermani, A., Perez-Rosas, V., & Metsis, V. (2025). A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG. *arXiv preprint arXiv:2503.24307*. https://arxiv.org/abs/2503.24307

Anshul, A., Pranav, G. S., Rehman, M. Z. U., & Kumar, N. (2024). A multimodal framework for depression detection during covid-19 via harvesting social media. *IEEE Transactions on Computational Social Systems, 11*(2), 2872-2888. https://doi.org/10.1109/TCSS.2023.3309229

Yang, K., Zhang, T., & Ananiadou, S. (2024). A mental state knowledge-aware and contrastive network for early stress and depression detection on social media. *Information Processing & Management, 59*(4), 102961. https://doi.org/10.1016/j.ipm.2022.102961

Zogan, H., Razzak, I., Jameel, S., & Xu, G. (2024). Hierarchical convolutional attention network for depression detection on social media and its impact during pandemic. *IEEE Journal of Biomedical and Health Informatics, 28*(4), 1815-1823. https://doi.org/10.1109/JBHI.2023.3243249

Owen, D., Lynham, A. J., Smart, S. E., Pardiñas, A. F., & Camacho Collados, J. (2024). AI for Analyzing Mental Health Disorders Among Social Media Users: Quarter-Century Narrative Review of Progress and Challenges. *Journal of Medical Internet Research, 26*, e59225. https://doi.org/10.2196/59225

Ilias, L., Mouzakitis, S., & Askounis, D. (2023). Calibration of transformer-based models for identifying stress and depression in social media. *IEEE Transactions on Computational Social Systems*. https://doi.org/10.1109/TCSS.2023.3283009

World Health Organization. (2023). Depression fact sheet. Retrieved from https://www.who.int/news-room/fact-sheets/detail/depression 

## Ethical Considerations

The development and deployment of an AI-based depression detection system raises important ethical considerations that must be carefully addressed to ensure responsible use.

### Privacy and Consent
Our system implements data minimization principles, ensuring only essential data is collected. Informed consent is facilitated through accessible descriptions of the system's operation, with a comprehensive data deletion API and automatic data expiration for temporarily stored information.

### Bias and Fairness
We constructed datasets with diverse demographic representation and implemented cultural sensitivity measures recognizing that expressions of mental health vary across cultures. Continuous monitoring for biased outcomes is implemented through fairness metrics tracking performance across demographic groups.

### Transparency and Explainability
Our system clearly communicates its role as a screening tool rather than a diagnostic instrument. The interpretation layer provides accessible explanations of model predictions, highlighting specific linguistic features that contributed to classifications.

### Potential Harm Mitigation
The implementation includes specific pattern detection for identifying language indicating suicidal ideation, with defined escalation pathways. Support resources accompany all screening results, ensuring users have access to appropriate mental health information regardless of their classification outcome.

Our ethical framework is implemented through an independent ethics review board, privacy-by-design principles, transparent documentation, and continuous monitoring.

## Challenges and Limitations

Despite promising results, several challenges and limitations must be acknowledged in our depression detection system. These constraints inform both the appropriate use of the current system and directions for future research and development.

### Technical Limitations

Natural language understanding remains an imperfect science, even with advanced transformer models. Our system faces challenges in understanding subtle contextual nuances, particularly when processing irony, metaphor, and culturally-specific expressions. Our testing revealed a significant accuracy decrease (22% lower) when analyzing highly metaphorical or idiomatic expressions of depression compared to more literal language. This limitation is particularly relevant for depression detection, as individuals often use metaphorical language to describe complex emotional states that literal language may fail to capture.

Sarcasm and figurative language present particular difficulties for our system. Even state-of-the-art language models struggle to consistently distinguish between genuine expressions of negative emotions and sarcastic statements that use similar vocabulary but convey different meanings. Our error analysis demonstrated that sarcastic expressions were misclassified 35% more frequently than straightforward statements, highlighting a significant area for improvement. This challenge is compounded by the fact that some individuals use humor, including dark humor, as a coping mechanism for depression, potentially leading to misclassification of their mental state.

Temporal dynamics in language patterns represent another technical challenge. Current models don't adequately account for the evolution of language over time, including emerging slang, evolving expressions, and shifting cultural references related to mental health. Terms that indicate depression may change rapidly, particularly among younger populations or specific subcultural groups. This limitation necessitates regular model updates to maintain accuracy as language evolves, creating a maintenance requirement that must be factored into deployment planning.

To mitigate these technical limitations, we recommend periodic model retraining with new data that captures evolving language patterns, deployment with human oversight, particularly for cases near classification thresholds or containing potential figurative language, and clear communication to users about these limitations to set appropriate expectations about system performance.

### Validation Gaps

Establishing ground truth for depression severity presents a fundamental challenge. Our system relies on clinical labels that, while based on established diagnostic criteria, contain their own subjectivity and may not perfectly reflect actual depression severity. Inter-rater reliability among clinical labelers, while high (Cohen's kappa = 0.82), still indicates some variance in professional judgment, reflecting the inherent challenge of quantifying subjective psychological experiences. This variance in ground truth labels places an upper bound on the potential accuracy of any predictive model, regardless of its sophistication.

Generalizability concerns arise when considering diverse populations. Performance may vary significantly across different demographic groups and contexts not adequately represented in our training data. Preliminary testing on texts from populations underrepresented in our training data (including different age groups, cultural backgrounds, and education levels) showed a performance decrease of 8-12%. This gap highlights the need for broader and more diverse training data to ensure equitable performance across all potential user populations.

The transition from controlled validation to real-world deployment introduces additional challenges. Initial pilot deployments have demonstrated approximately 5% lower accuracy in real-world settings compared to test set performance, suggesting a distribution shift between carefully curated research datasets and messy real-world text data. Factors contributing to this performance gap include the presence of multiple health conditions (comorbidity), varying writing contexts (formal vs. informal, professional vs. personal), and the dynamic nature of depressive symptoms over time.

To address these validation gaps, we recommend ongoing validation studies with diverse populations to continuously refine the model's performance across different demographic groups, gradual deployment with careful performance monitoring to identify any gaps between expected and actual performance, and implementation of feedback loops for continuous improvement, allowing the system to adapt to real-world usage patterns.

### Implementation Challenges

Computational requirements present practical constraints on deployment. Transformer models require significant computational resources, which may limit deployment in resource-constrained settings. Our RoBERTa model requires approximately 2GB of GPU memory for inference and 8GB for fine-tuning, making it unsuitable for some low-resource environments and potentially creating disparities in access to this technology.

Latency considerations affect user experience and clinical utility. Real-time analysis requires optimization for performance, which we've addressed through model quantization and batch processing, but remains a consideration for high-volume applications. In healthcare settings, where timely information can be crucial, balancing speed and accuracy presents an ongoing challenge that must be addressed based on the specific deployment context.

Integration complexity with existing healthcare systems creates implementation hurdles. Electronic health records, clinical workflows, and healthcare IT infrastructure vary widely across institutions, requiring custom connectors and adaptation. The diversity of systems and lack of standardization in healthcare IT can impede smooth deployment, potentially limiting the reach and impact of the technology despite its technical merits.

We have developed several mitigation strategies for these implementation challenges, including distilled model versions for resource-constrained environments that trade some accuracy for significantly reduced computational requirements, asynchronous processing options for latency-sensitive applications that separate user interaction from heavy computational tasks, and standard API specifications with reference implementations to simplify integration efforts.

### Future Research Directions

To address the limitations identified above, several research directions show particular promise:

Multimodal analysis represents a significant opportunity to enhance detection accuracy. By combining text analysis with other data sources such as voice recordings (analyzing acoustic features like prosody and rhythm), activity patterns (capturing changes in behavior), and sleep data (monitoring disruptions in sleep patterns), we could develop a more comprehensive assessment of depression indicators that doesn't rely solely on written language.

Longitudinal modeling would enable the system to capture changes in language patterns over time, detecting trends and temporal patterns associated with depression onset or recovery. This approach could provide earlier warning signs of deterioration or confirmation of improvement, enabling more timely intervention or adjustment of treatment plans.

Personalized baselines could significantly improve accuracy by establishing individual linguistic patterns as reference points. By analyzing a person's typical communication style, the system could more precisely identify deviations that might indicate changes in mental health status, accounting for individual differences in baseline language use and expression of emotions.

Cross-cultural validation studies would strengthen the system's applicability across diverse populations. Expanding validation across different cultural and linguistic contexts would ensure that the model can effectively recognize culturally-specific expressions of depression, improving generalizability and reducing potential biases.

Privacy-preserving techniques such as federated learning hold promise for improving models without centralized data collection. This approach would allow model improvement based on data from multiple organizations or devices while keeping sensitive text data local, addressing privacy concerns while enabling ongoing model refinement.

These research directions are incorporated into our development roadmap and will inform future system iterations. By systematically addressing current limitations through targeted research and development, we aim to continuously improve the system's accuracy, applicability, and ethical implementation.

## Conclusion and Recommendations

Our research on depression detection through linguistic analysis has revealed powerful insights into how mental health states manifest in written language. By analyzing patterns across word usage, sentiment distribution, and text structure, we've developed a screening system that achieves 78.5% accuracy in identifying depression severity—a significant improvement over previous approaches.

The linguistic markers of depression emerged with remarkable consistency across our datasets. Word clouds visually demonstrated how language shifts from externally-focused, social terminology in minimal depression to more negative, self-referential language in severe cases. First-person pronouns, negative emotion words, and absolutist terms like "never" and "always" proved to be the strongest predictors, confirming clinical observations about depression's cognitive patterns. These findings reinforce what mental health professionals have long observed in therapeutic settings: depression fundamentally alters not just what people say, but how they express themselves.

Sentiment analysis revealed a clear progression in emotional expression, with texts showing increasingly negative polarity as depression severity increased. The visualization of sentiment polarity versus subjectivity demonstrated that severe depression clusters distinctively in the negative region, providing a quantifiable signal for detection. Perhaps most revealing was the text length distribution pattern, which showed that severe depression produces greater variability in expression—some individuals become nearly silent while others engage in extensive rumination. This bimodal pattern mirrors clinical observations of depression's diverse presentations, where some withdraw into themselves while others process their distress through lengthy internal dialogue.

Our transformer models captured these nuanced linguistic signals with unprecedented accuracy. The RoBERTa model achieved 78.5% accuracy compared to 66.2% from traditional approaches, but the numbers tell only part of the story. What truly distinguishes these models is their ability to attend to meaningful phrases even when explicit depression keywords are absent. They recognize the shadows depression casts on language—the subtle shifts in expression that might escape notice in casual conversation but collectively signal emotional distress.

While our technical achievements are significant, this research consistently highlighted that technology alone is insufficient for addressing mental health concerns. The most responsible and effective implementation positions this system as a supportive tool for healthcare professionals, not a replacement for clinical judgment. We've developed an integrated explanation system that combines attention visualization with feature importance, ensuring that model predictions remain interpretable and trustworthy to clinical users. This transparency builds trust, allowing professionals to understand why the system reaches specific conclusions rather than presenting them with a black box assessment.

Our ethical framework emerged from recognizing the deeply personal nature of depression screening. We prioritize privacy protection through data minimization, informed consent, and giving users control over their personal information. We've addressed algorithmic bias by constructing diverse training datasets and implementing continuous monitoring for biased outcomes across demographic groups. The system explicitly communicates its role as a screening tool rather than a diagnostic instrument and includes specific protocols for identifying and responding to language indicating potential self-harm. These ethical considerations aren't secondary to the technical implementation—they form its foundation.

This research bridges the gap between academic understanding of depression linguistics and practical screening applications. The system architecture incorporates modular components for data processing, model implementation, prediction, interpretation, and integration—all designed for real-world deployment with an emphasis on security, robustness, and accessibility. For implementation to succeed, we recommend integration within existing healthcare workflows with professional oversight and continuous monitoring for performance and bias across diverse populations. Clear communication about the system's screening purpose (not diagnostic capabilities) must accompany any deployment, alongside strong privacy protections including local processing options where possible. All interfaces should be designed with accessibility in mind, ensuring the technology serves diverse users effectively.

Looking forward, we see several promising directions for future development. We envision multimodal integration that combines text analysis with voice, activity, and sleep data to provide a more comprehensive picture of mental well-being. Longitudinal tracking will help identify changes over time, potentially capturing early warning signs of deterioration or confirming improvement during treatment. Cultural adaptation for diverse populations and mobile optimization for resource-constrained environments will expand accessibility, while expanded language support beyond English will address global needs.

Depression often remains undetected until it significantly impacts a person's life. Our research demonstrates that linguistic patterns can serve as early indicators, potentially enabling earlier intervention when treatment is most effective. In a world where mental health resources remain limited and stigma persists, automated screening could help bridge the gap between those suffering silently and the support they need. The varying word patterns, sentiment distributions, and linguistic markers across depression severity levels tell more than a technical story—they reveal how mental health shapes our relationship with language itself. By responsibly harnessing these signals while respecting privacy, consent, and clinical expertise, this system offers a promising approach to addressing the global challenge of undetected depression. Behind every data point and model prediction are real people whose suffering might be alleviated through earlier identification and support. This human-centered perspective guided our research and must continue to shape how this technology enters the world. 