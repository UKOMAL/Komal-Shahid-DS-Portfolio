# Project 1: AI-Powered Depression Detection System

## Overview

This project implements an AI-powered system for detecting indicators of depression from written text using Natural Language Processing (NLP) and Deep Learning, specifically transformer models (BERT).

For a detailed explanation of the methodology, findings, and ethical considerations, please refer to the [White Paper](./docs/white_paper.md).

## Features

- Analyzes text to classify depression severity (minimum, mild, moderate, severe).
- Utilizes a fine-tuned BERT transformer model.
- Provides confidence scores for each severity category.
- Offers a command-line interface (CLI) for single text analysis, batch processing from CSV files, and interactive mode.
- Includes functionality to save and load trained models.

## Project Structure

```
project1-depression-detection/
├── README.md               # This file: Instructions for setup and usage
├── depression_detector.py  # Main consolidated Python script
├── requirements.txt       # Python package dependencies
├── data/
│   └── sample/           # Sample input data and expected results
├── docs/
│   ├── white_paper.md  # Detailed project documentation
│   ├── depression_detection_presentation.md # Project presentation slides (Markdown)
│   └── original_documents/ # Backup of original DOCX/PDF files (not tracked by Git)
├── models/                 # Default directory for saving/loading trained models
├── output/                 # Directory for generated outputs (e.g., plots)
│   ├── word_frequency_by_category.png  # Placeholder*
│   ├── sentiment_distribution.png     # Placeholder*
│   ├── interactive_model_comparison.png # Placeholder*
│   ├── attention_visualization.png    # Placeholder*
│   ├── depression_spectrum_visualization.png # Placeholder*
│   └── transformer_confusion_matrix.png # Placeholder*
└── src/                    # Source code modules (imported by depression_detector.py)
    ├── app/                # Application logic (if separated)
    ├── data/               # Data loading/preprocessing functions
    ├── models/             # Model definitions
    ├── utils/              # Utility functions
    └── visualization/      # Visualization functions
```
*Note: Image files in the `output/` directory are placeholders. You need to run the script to generate the actual visualizations based on your data and model training.* 

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
    *Note: This includes TensorFlow, Transformers, scikit-learn, pandas, etc.* 

3.  **Download NLP Resources (if needed by specific functions):** 
    The current transformer model doesn't strictly require NLTK/spaCy downloads, but if you extend functionality, you might need:
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

The main script `depression_detector.py` provides a command-line interface.

```bash
# Navigate to the project directory
cd projects/project1-depression-detection

# Display help message for options
python3 depression_detector.py --help

# Analyze a single piece of text
python3 depression_detector.py --mode single --text "I have been feeling quite down and unmotivated lately."

# Analyze text from a CSV file (batch mode)
# Assumes a CSV file named 'input_texts.csv' in the 'data/' directory with a column named 'text_column_name'
python3 depression_detector.py --mode batch --input_file data/input_texts.csv --text_column text_column_name --output_file output/batch_results.csv

# Enter interactive mode for continuous analysis
python3 depression_detector.py --mode interactive

# Train the model (if training data is prepared)
# python3 depression_detector.py --mode train --train_data path/to/train.csv --val_data path/to/val.csv --model_dir models/my_trained_model

# Evaluate the model (if test data is prepared)
# python3 depression_detector.py --mode evaluate --test_data path/to/test.csv --model_dir models/my_trained_model --output_dir output/
```

*(Note: Training and evaluation data preparation steps are not fully detailed here but the script supports these modes).* 

## Model

The primary model used is a BERT-based transformer fine-tuned for depression severity classification. The script includes functionality to train, evaluate, save, and load models.

## Contributing

This project was developed for DSC680. Contributions or suggestions are welcome via issues or pull requests on the main portfolio repository.
