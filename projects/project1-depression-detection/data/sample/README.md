# Sample Data

This directory contains small sample datasets for demonstration purposes. The full datasets used in the research are not included in this repository due to privacy considerations and size constraints.

## Files

- `sample_depression_data.csv` - A small subset (100 records) of the depression text data used for demonstration
- `sample_results.csv` - Example model prediction results

## Data Structure

### sample_depression_data.csv

This file contains the following columns:

- `text_id`: Unique identifier for each text sample
- `text`: The text content being analyzed
- `depression_level`: The depression severity level (minimum, mild, moderate, severe)
- `word_count`: Number of words in the text
- `sentiment_score`: Calculated sentiment polarity score (-1 to 1)

### sample_results.csv

This file contains example results from model predictions:

- `text_id`: Identifier matching to the original text
- `true_label`: Actual depression severity level
- `predicted_label`: Model's predicted severity level
- `confidence`: Model's confidence score for the prediction
- `key_indicators`: Top linguistic features that influenced the prediction

## Usage Notes

These samples are provided for demonstration purposes only and should not be used for training production models. The data has been anonymized and modified to protect privacy while maintaining the general patterns found in the original research.

To use the full dataset for research purposes, please contact the author for information about data access protocols and ethical usage guidelines. 