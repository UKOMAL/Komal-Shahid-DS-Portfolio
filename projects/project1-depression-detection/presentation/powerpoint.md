---
marp: true
theme: default
paginate: true
---

<!-- 
This is a Markdown file optimized for PowerPoint import.
You can also use this with Marp, Slidev, or other Markdown slide tools.
-->

# Depression Detection System

Using NLP to Identify Signs of Depression

Komal Shahid
Master's in Data Science, Bellevue University

<!-- Speaker Notes:
Good afternoon everyone. Today, I'm presenting my work on a Depression Detection System that uses Natural Language Processing and Machine Learning to identify potential signs of depression in written text. This project sits at the intersection of artificial intelligence and mental health, with the goal of creating an accessible screening tool that could help identify those who might need professional support.
-->

---

# The Challenge

- Depression affects 264+ million people globally
- Average 10-year delay between onset and treatment
- Subtle symptoms often go unrecognized
- Stigma prevents many from seeking help

<!-- Speaker Notes:
Depression affects over 264 million people globally according to the WHO, yet many cases go undiagnosed. The average delay between onset of symptoms and treatment is approximately 10 years. This delay occurs for multiple reasons: stigma around mental health, lack of accessible screening tools, and the often subtle progression of symptoms that individuals might not recognize in themselves. Our challenge was to develop a system that could serve as an initial, non-invasive screening tool to help identify potential depression indicators from natural language.
-->

---

# Project Objectives

1. Develop accurate NLP model for depression detection
2. Balance sensitivity and specificity
3. Create accessible, integration-ready interface
4. Ensure privacy and ethical data handling
5. Validate against clinical screening tools

<!-- Speaker Notes:
The primary objectives of this project were to: First, develop an accurate NLP model to detect linguistic patterns associated with depression. Second, create a system that balances sensitivity and specificity - we need to identify potential cases without excessive false alarms. Third, design an accessible interface that could be integrated into various platforms like telehealth systems, patient portals, or even social media. Fourth, ensure privacy and ethical handling of all data, maintaining HIPAA compliance. And finally, validate the system against established clinical screening tools to ensure it provides meaningful insights.
-->

---

# Data Acquisition & Preparation

- **Reddit dataset**: 30,000+ anonymized posts from depression-related subreddits
- **DAIC-WOZ**: Transcribed clinical interviews
- **Synthetic data**: Augmenting underrepresented linguistic patterns

*All data anonymized, cleaned, and balanced*

<!-- Speaker Notes:
For this project, we utilized multiple data sources: A carefully curated dataset of Reddit posts from depression-related and control subreddits, with over 30,000 anonymized text samples; the DAIC-WOZ dataset containing transcribed clinical interviews; and a synthetic dataset created to augment specific linguistic patterns underrepresented in the main datasets. All data was anonymized, cleaned, and balanced to ensure no demographic biases. The preprocessing pipeline included tokenization, removing personally identifiable information, and handling class imbalance through techniques like SMOTE.
-->

---

# Technical Approach

- **Core**: DistilBERT transformer architecture
- **Features**: Semantic content + linguistic depression markers
- **Training**: Transfer learning with fine-tuning
- **Explainability**: SHAP values for interpretation

<!-- Speaker Notes:
Our approach combined multiple NLP techniques: We implemented a DistilBERT-based transformer model as our core architecture. Feature extraction focused on both semantic content and linguistic markers known to correlate with depression, such as increased use of first-person singular pronouns, absolute terms like "always" and "never", negative emotional content, and linguistic indicators of social isolation. The model was fine-tuned using transfer learning from pre-trained language models. We incorporated explainability techniques like SHAP values to make the model's decisions interpretable for healthcare professionals.
-->

---

# Results

- 87% accuracy
- 89% sensitivity (recall)
- 85% specificity
- F1 score: 0.88

*Consistent performance across demographics and writing styles*

<!-- Speaker Notes:
Our final model achieved 87% accuracy on the test dataset, 89% sensitivity (recall), which was prioritized to minimize missed cases, 85% specificity, balancing false positives, and an F1 score of 0.88. The model performed consistently across different demographics and writing styles. Importantly, we found that combining multiple linguistic features significantly outperformed models that relied solely on semantic content or emotional word detection.
-->

---

# Ethics & Responsible Implementation

- Screening tool only, not diagnostic replacement
- Confidence intervals and factor explanations
- Differential privacy for data protection
- Human-in-the-loop review process
- Regular bias audits

<!-- Speaker Notes:
Throughout this project, ethical considerations were paramount: The system is designed as a screening tool only, not a diagnostic replacement. All predictions include confidence intervals and explanations of the contributing factors. We implemented differential privacy techniques to protect user data. A human-in-the-loop approach ensures healthcare professionals review flagged content. Regular bias audits ensure the system performs consistently across different demographic groups. And we maintain clear communication to users about how their data is being used and the limitations of the system.
-->

---

# System Demonstration

*Live demonstration of web interface*

<!-- Speaker Notes:
Here you can see the web interface we've developed. The system accepts text input - which could be journal entries, social media posts, or responses to specific prompts. The analysis runs in real-time, providing an assessment of depression indicators along with the confidence level and specific linguistic patterns identified. Healthcare providers can use this dashboard to review multiple patients and track changes over time. Let me walk you through a quick demonstration of how the system handles different types of text input and what insights it provides.
-->

---

# Looking Forward

- Multi-language support
- Age-specific models
- Mobile applications
- EHR integration
- Multimodal inputs
- Longitudinal validation studies

<!-- Speaker Notes:
While the current system shows promising results, we have several directions for enhancement: Expanding to multiple languages beyond English; developing specialized models for different age groups, particularly adolescents; creating mobile applications for continuous passive monitoring with user consent; integrating with EHR systems for streamlined clinical workflows; incorporating multimodal inputs such as speech patterns and vocal biomarkers; and conducting longitudinal studies to validate the system's effectiveness as an early screening tool.
-->

---

# Thank You

Questions?

**Contact**: komalshahid@example.com
**GitHub**: github.com/UKOMAL

<!-- Speaker Notes:
To conclude, the Depression Detection System demonstrates how NLP can be leveraged as a screening tool for mental health conditions. While technology cannot replace human connection and professional care, it can serve as an accessible first step in identifying those who might benefit from further assessment. By combining advanced machine learning with careful ethical implementation, we hope this work contributes to earlier intervention and better outcomes for individuals experiencing depression. I welcome any questions you might have about the technical implementation, ethical considerations, or potential applications of this system.
--> 