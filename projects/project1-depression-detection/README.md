# AI-Powered Depression Detection System

## Overview

This project implements an AI-powered system for detecting indicators of depression from written text. The system analyzes linguistic patterns and contextual cues to classify text according to depression severity levels (minimum, mild, moderate, severe).

The project includes:
- Traditional machine learning models (Random Forest, SVM, Gradient Boosting)
- Advanced transformer-based models (BERT, RoBERTa)
- A comprehensive API for text analysis
- Interactive demo application
- Visualizations of linguistic patterns and model performance

## Project Structure

```
Project1/
├── docs/                      # Documentation
├── output/                    # Visualizations and analysis outputs
├── presentation/              # Presentation files
│   ├── depression_detection_presentation.md
│   └── audio_recording_instructions.md
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   │   └── sample/            # Sample data for testing
│   ├── models/                # Model implementations
│   │   └── transformer_model.py
│   ├── utils/                 # Utility functions
│   ├── visualization/         # Visualization functions
│   ├── depression_detection.py # Main system interface
│   └── main.py                # Demo application
├── white_paper_draft.md       # Original project white paper
├── white_paper_updated.md     # Updated white paper with transformer models
└── README.md                  # This file
```

## Key Features

- **Multi-model approach**: Combines traditional ML and transformer-based models
- **Severity classification**: Categorizes text into minimum, mild, moderate, or severe depression indicators
- **Confidence scoring**: Provides confidence scores for each severity category
- **Interpretability**: Highlights key linguistic features driving the classification
- **Batch processing**: Ability to analyze multiple texts from CSV files
- **Interactive mode**: Command-line interface for real-time text analysis

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.5+
- Transformers library
- Pandas, NumPy, Matplotlib, Seaborn
- NLTK, spaCy

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download required NLTK and spaCy resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   
   import spacy
   spacy.cli.download("en_core_web_sm")
   ```

### Usage

#### Command-line interface

```bash
# Single text analysis
python src/main.py --mode single

# Batch analysis
python src/main.py --mode batch

# Interactive mode
python src/main.py --mode interactive
```

#### Python API

```python
from src.depression_detection import DepressionDetectionSystem

# Initialize the system
system = DepressionDetectionSystem(model_type="transformer")

# Analyze a single text
result = system.predict("I haven't been feeling like myself lately...")
print(f"Depression severity: {result['depression_severity']}")
print(f"Confidence scores: {result['confidence_scores']}")

# Batch analysis
results_df = system.batch_analyze("texts.csv", text_column="text")
```

## Ethical Considerations

This system is designed as a screening tool, not a diagnostic tool. It should be used with the following considerations:

- **Not a replacement for professional assessment**: The system should complement, not replace, professional mental health evaluation.
- **Privacy protection**: Minimize data storage and implement strong security measures when deploying.
- **Informed consent**: Users should understand how their text is analyzed and what the results mean.
- **Support resources**: Clear pathways to professional help should accompany all screening results.

## Model Performance

The system includes multiple models with varying performance:

| Model | Accuracy |
|-------|----------|
| Random Forest | 62.18% |
| Support Vector Machine | 63.45% |
| Gradient Boosting | 66.22% |
| Logistic Regression | 61.03% |
| LSTM with GloVe | 70.34% |
| BiLSTM with attention | 72.18% |
| Fine-tuned BERT-base | 75.92% |
| Fine-tuned RoBERTa | 78.50% |

## Documentation

For more detailed information, please refer to:
- [White Paper](white_paper_updated.md): Comprehensive documentation of methodology, results, and implications
- [Presentation](presentation/depression_detection_presentation.md): Summary presentation of the project

## License

This project is for educational and research purposes only.

## Acknowledgments

This project was developed as part of DSC680 at Bellevue University.
