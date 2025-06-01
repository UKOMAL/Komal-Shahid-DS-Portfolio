# Project 1: AI-Powered Depression Detection System

## Overview

This project implements an AI-powered system for detecting indicators of depression from written text using Natural Language Processing (NLP) and Deep Learning, specifically transformer models (DistilBERT) from HuggingFace with rule-based enhancements.

For a detailed explanation of the methodology, findings, and ethical considerations, please refer to the [White Paper](./docs/white_paper.md).

## Features

- Analyzes text to classify depression severity (minimum, mild, moderate, severe).
- Utilizes a HuggingFace transformer model with rule-based enhancements.
- Provides confidence scores for each severity category.
- Offers a command-line interface (CLI) for single text analysis, batch processing from CSV files, and case studies.
- Includes visualizations for depression severity distribution and confidence scores.

## Project Structure

```
project1-depression-detection/
├── README.md               # This file: Instructions for setup and usage
├── requirements.txt        # Python package dependencies
├── data/                   # Directory for input data
├── docs/
│   ├── white_paper.md      # Detailed project documentation
│   └── original_documents/ # Backup of original DOCX/PDF files (not tracked by Git)
├── models/                 # Directory for saving/loading models
│   └── transformer/        # HuggingFace model files
├── output/                 # Directory for generated outputs (e.g., plots)
│   ├── word_frequency_by_category.png
│   ├── sentiment_distribution.png
│   ├── interactive_model_comparison.png
│   ├── attention_visualization.png
│   ├── depression_spectrum_visualization.png
│   └── transformer_confusion_matrix.png
├── src/                    # Source code modules
    ├── app/                # Application logic
    ├── data/               # Data loading/preprocessing functions
    ├── models/             # Model definitions
    │   └── transformer_model.py # HuggingFace model implementation
    ├── utils/              # Utility functions
    ├── visualization/      # Visualization functions
    └── depression_detection.py # Main script for depression detection
```

## Setup and Installation

It is highly recommended to use a virtual environment to manage dependencies.

1.  **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This includes PyTorch, HuggingFace Transformers, scikit-learn, pandas, etc.* 

3.  **Download NLP Resources (if needed by specific functions):** 
    ```python
    # Example if using NLTK
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
    # Example if using spaCy
    import spacy
    spacy.cli.download("en_core_web_sm")
    ```

## Usage

The main script `src/depression_detection.py` provides a command-line interface.

```bash
# Navigate to the project directory
cd projects/project1-depression-detection

# Run the case studies demonstration
python3 src/depression_detection.py --case-studies

# Analyze text from a CSV file (batch mode)
python3 src/depression_detection.py --file data/input_texts.csv --text-col text_column_name --output output/results.csv --visualize
```

## Model

The system uses a DistilBERT model from HuggingFace with rule-based enhancements for improved depression severity classification. The enhanced model uses linguistic features such as:

- Keyword presence for different severity levels
- First-person pronoun density
- Negation patterns
- Text length and complexity

This hybrid approach combines the power of transformer models with clinically-relevant linguistic markers of depression.

## Contributing

This project was developed for DSC680. Contributions or suggestions are welcome via issues or pull requests on the main portfolio repository.
