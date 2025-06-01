#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System - Consolidated Application
Version: 1.0.0 (Final Release)
Date: April 6, 2025
Authors: Komal Shahid
License: MIT

This module implements a comprehensive AI-based depression detection system 
that analyzes text inputs to identify indicators of depression and classify 
their severity. The system uses transformer-based models (BERT variants) to 
detect linguistic patterns associated with depression.

Key components:
- TransformerDepressionModel: Core ML implementation using BERT
- DepressionDetectionSystem: User-facing interface for the detection system
- Visualization and reporting capabilities
- Command-line interface for various usage modes

IMPORTANT: This system is intended for research and screening purposes only,
not for clinical diagnosis. All predictions should be reviewed by qualified
mental health professionals.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define constants and default paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)

#-------------------------------------------------------------------------
# Removed TransformerDepressionModel Implementation (Now in src/models/)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Removed DepressionDataset Implementation (Now in src/models/)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# User Interface
#-------------------------------------------------------------------------

class DepressionDetectionSystem:
    """
    Main interface for the depression detection system.
    Uses the HuggingFace Transformer model from src/models/transformer_model.py
    """
    
    def __init__(self, model_type="huggingface", model_path=None):
        """
        Initialize the detection system using the HuggingFace model.
        
        Args:
            model_type (str): Should be 'huggingface'.
            model_path (str): Path to a saved HuggingFace model directory (optional, defaults to project standard).
        """
        # Ensure we explicitly use the HF model from src/models
        from models.transformer_model import TransformerDepressionModel 
        
        self.model_type = model_type 
        # Determine the model directory path
        if model_path:
             self.model_dir = model_path
        else:
             # Default path relative to this file's location
             project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             self.model_dir = os.path.join(project_root, "models", "transformer")
             
        self.model: Optional[TransformerDepressionModel] = None # Type hint for HF model
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the HuggingFace model by loading from directory."""
        from models.transformer_model import TransformerDepressionModel
        
        try:
            print(f"Loading HuggingFace model from {self.model_dir}...")
            # The HF TransformerDepressionModel now handles loading in its __init__
            self.model = TransformerDepressionModel(model_dir=self.model_dir)
            if not self.model.model or not self.model.tokenizer:
                 raise ValueError("Model or tokenizer failed to load within TransformerDepressionModel init.")
                 
        except Exception as e:
            logger.error(f"Fatal error initializing HuggingFace model: {str(e)}")
            # If init fails, self.model will remain None or could raise, stopping execution.
            # Consider how to handle this - maybe allow running without a model? For now, let it fail.
            raise # Re-raise the exception to indicate critical failure

    def predict(self, text):
        """
        Analyze a single text input using the HuggingFace model.
        Returns a dictionary compatible with the rest of the system.
        """
        if not self.model:
            return {'error': "Model is not initialized.", 'disclaimer': "..."}
            
        if not text or not isinstance(text, str) or not text.strip():
            # Handle empty input consistently
            return {
                'depression_severity': self.model.classes[0] if self.model.classes else 'unknown', # Predict least severe or unknown
                'confidence': 1.0,
                'confidence_scores': {cls_name: (1.0 if i == 0 else 0.0) for i, cls_name in enumerate(self.model.classes)} if self.model.classes else {},
                'warning': 'Empty or whitespace-only text provided',
                'disclaimer': "This analysis is for informational purposes only..."
            }
        
        try:
            # Use the HF model's predict method
            predicted_label, confidence_scores_array = self.model.predict(text)
            
            # Format results into the expected dictionary structure
            confidence_scores_dict = dict(zip(self.model.classes, confidence_scores_array))
            predicted_confidence = confidence_scores_dict.get(predicted_label, 0.0)
            
            result = {
                'depression_severity': predicted_label,
                'confidence': float(predicted_confidence), # Ensure float
                'confidence_scores': {k: float(v) for k, v in confidence_scores_dict.items()}, # Ensure float
                # Get attention scores if the model supports it
                'attention': self._get_hf_attention(text) 
            }
            
            # Add guidance (remains the same logic)
            severity = result['depression_severity']
            if severity == 'moderate' or severity == 'severe':
                result['guidance'] = "This text shows indicators of significant depression. " \
                                    "If this is your own text, please consider speaking with a " \
                                    "mental health professional."
            elif severity == 'mild':
                result['guidance'] = "This text shows some indicators of mild depression. " \
                                    "Consider self-care practices and monitoring your mental well-being."
            else:
                result['guidance'] = "This text shows minimal indicators of depression."
            
            result['disclaimer'] = "This analysis is for informational purposes only and " \
                                  "is not a clinical diagnosis."
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict method (HF Model): {str(e)}")
            # import traceback
            # traceback.print_exc()
            return {
                'error': str(e),
                'disclaimer': "This analysis is for informational purposes only and is not a clinical diagnosis."
            }

    def _get_hf_attention(self, text: str) -> Optional[Dict]:
        """Helper to extract attention if available from the HF model."""
        if not self.model or not self.model.model or not self.model.tokenizer:
            return None
        
        # Check if the loaded model is configured to output attentions
        # Note: This requires the model itself to support and be configured for it.
        # We might need to reload the model with `output_attentions=True` if needed.
        if not getattr(self.model.model.config, 'output_attentions', False):
             # print("Model not configured to output attentions.") # Optional debug
             return None

        try:
            inputs = self.model.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
            self.model.model.eval()
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_attentions=True)
            
            # Process attentions (example: last layer, average heads, CLS token)
            if hasattr(outputs, 'attentions') and outputs.attentions:
                 last_layer_attention = outputs.attentions[-1].cpu().numpy()
                 # Average over attention heads [batch, head, seq_len, seq_len]
                 attention_scores = last_layer_attention.mean(axis=1)[0] # Get first item in batch, avg heads
                 cls_attention = attention_scores[0, :] # Attention from CLS token to all tokens
                 
                 tokens = self.model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                 
                 # Filter padding/special tokens for visualization
                 viz_data = []
                 for token, score in zip(tokens, cls_attention):
                      if token not in [self.model.tokenizer.cls_token, self.model.tokenizer.sep_token, self.model.tokenizer.pad_token]:
                           viz_data.append({"token": token, "score": float(score)})
                           
                 return {"visualization_data": viz_data}
            else:
                 return None
                 
        except Exception as e:
             logger.warning(f"Could not extract attention scores: {e}")
             return None

    def batch_analyze(self, file_path, text_column="text", output_file=None):
        """
        Analyze multiple texts using the HuggingFace model.
        """
        if not self.model:
            print("Error: Model not initialized for batch analysis.")
            return None
            
        try:
            # Load data
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            data = pd.read_csv(file_path)
            
            if text_column not in data.columns:
                print(f"Column '{text_column}' not found in the data at {file_path}")
                return None
            
            # Get texts, handle potential NaN values
            texts = data[text_column].fillna("").astype(str).tolist()
            
            if not texts:
                print("No valid texts found in the specified column.")
                return None
            
            print(f"Starting batch analysis of {len(texts)} texts with HuggingFace model...")
            # Use the HF model's predict method for batch processing
            # It returns lists: list of labels, list of probability arrays
            predicted_labels, list_of_confidences = self.model.predict(texts)
            print("Batch prediction complete.")
            
            # Create results dataframe
            results_df = data.copy()
            results_df['predicted_severity'] = predicted_labels
            
            # Add confidence scores for each class
            num_classes = len(self.model.classes)
            for i, cls_name in enumerate(self.model.classes):
                 # Extract the confidence for this class from each array in the list
                 results_df[f'confidence_{cls_name}'] = [conf_array[i] if len(conf_array) == num_classes else 0.0 for conf_array in list_of_confidences]
            
            # Save results if output file is specified
            if output_file:
                output_dir = os.path.dirname(output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                results_df.to_csv(output_file, index=False)
                print(f"Batch analysis results saved to {output_file}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error in batch_analyze method (HF Model): {str(e)}")
            # import traceback
            # traceback.print_exc()
            return None
    
    def interactive_mode(self):
        """
        Run an interactive session (adapted for HF model output)
        """
        print("\n=== Depression Detection Interactive Mode (HF Model) ===\n")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'help' for more information")
        print("DISCLAIMER: This is not a diagnostic tool. Results are for informational purposes only.")
        
        while True:
            try:
                print("\nEnter text to analyze (or 'exit' to quit):")
                text = input("> ")
                
                if text.lower() in ['exit', 'quit']:
                    print("Exiting interactive mode")
                    break
                
                if text.lower() == 'help':
                    print("\nHelp Information:")
                    print("- Enter any text to analyze it for depression indicators")
                    print("- The system will classify the text into one of four severity levels:")
                    print("  minimum, mild, moderate, or severe")
                    print("- The system also provides confidence scores for each category")
                    print("- This is NOT a diagnostic tool and should not replace professional advice")
                    print("- Type 'exit' or 'quit' to end the session")
                    continue
                
                if not text or not text.strip():
                    print("Please enter some text.")
                    continue
                
                # Analyze the text using the updated predict method
                result = self.predict(text)
                
                # Display results (check for errors first)
                if 'error' in result:
                     print(f"Error: {result['error']}")
                     continue
                    
                print("\nAnalysis Results:")
                print(f"Predicted Severity: {result['depression_severity']}")
                print(f"Confidence: {result['confidence']:.3f}")
                
                print("\nConfidence Scores per Class:")
                for label, score in result['confidence_scores'].items():
                    print(f"  {label}: {score:.3f}")
                
                if 'guidance' in result:
                    print(f"\nGuidance: {result['guidance']}")
                
                print(f"\n{result['disclaimer']}")
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting interactive mode.")
                break
            except Exception as e:
                print(f"Interactive mode error: {str(e)}")
    
    def save_model(self, save_path=None):
        """
        Save the current HuggingFace model
        
        Args:
            save_path (str): Directory to save the HF model (optional)
        """
        if not self.model:
             print("Error: No model loaded to save.")
             return False
            
        if save_path is None:
            save_path = self.model_dir # Default to saving where it was loaded from
        
        print(f"Saving HuggingFace model to {save_path}...")
        return self.model.save(save_path)

#-------------------------------------------------------------------------
# Demo Functions (Need updates to use the HF model system)
#-------------------------------------------------------------------------

def demo_single_text():
    """Demonstrate single text analysis using the HF model system"""
    print("\n=== Single Text Analysis Demo (HF Model) ===\n")
    
    # Initialize the system (will use HF model by default)
    system = DepressionDetectionSystem()
    
    # Example text
    text = "I haven't been feeling like myself lately. It's hard to get out of bed in the morning."
    
    # Analyze
    print(f"Analyzing text: \"{text}\"")
    result = system.predict(text)
    
    # Display results
    if 'error' in result:
         print(f"Error: {result['error']}")
         return None
        
    print("\nResult:")
    print(f"Predicted Severity: {result['depression_severity']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nConfidence Scores:")
    for label, score in result['confidence_scores'].items():
        print(f"  {label}: {score:.3f}")
    
    if 'guidance' in result:
        print(f"\nGuidance: {result['guidance']}")
    
    print(f"\n{result['disclaimer']}")
    return result

def demo_batch_analysis():
    """Demonstrate batch analysis using the HF model system"""
    print("\n=== Batch Analysis Demo (HF Model) ===\n")
    
    # Initialize the system
    system = DepressionDetectionSystem()
    
    # Sample data path
    sample_path = os.path.join(DEFAULT_DATA_DIR, "sample", "sample_texts_hf.csv") # Use a diff name maybe
    output_path = os.path.join(DEFAULT_OUTPUT_DIR, "batch_results_hf_demo.csv")
    
    # Check if sample data exists, create it if not
    if not os.path.exists(sample_path):
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'text': [
                "Feeling great and positive!",
                "Feeling a bit down, stressed.",
                "No joy, always tired, can't focus.",
                "Completely worthless, struggle daily, want pain to stop."
            ]
        })
        sample_data.to_csv(sample_path, index=False)
        print(f"Created sample HF data at {sample_path}")
    
    # Analyze
    print(f"Analyzing texts from: {sample_path} using HF model")
    results = system.batch_analyze(sample_path, text_column="text", output_file=output_path)
    
    # Display summary
    if results is not None:
        print("\nBatch Analysis Summary (HF Model):")
        if 'predicted_severity' in results.columns:
             severity_counts = results['predicted_severity'].value_counts()
             print(severity_counts)
        else:
             print("Could not find 'predicted_severity' column in results.")
        return results
    
    return None

def print_usage():
    """Print usage information"""
    print("\nDepression Detection System - Usage")
    print("-----------------------------------")
    print("Usage: python depression_detector.py [options]")
    print("\nOptions:")
    print("  --mode MODE              Specify the mode: single, batch, or interactive")
    print("  --text \"TEXT\"            Text to analyze (for single mode)")
    print("  --file PATH              Path to CSV file (for batch mode)")
    print("  --column NAME            Column name containing text (for batch mode, default: text)")
    print("  --output PATH            Path to save output (for batch mode)")
    print("  --model PATH             Path to a saved HuggingFace model directory")
    print("  --save PATH              Save the HuggingFace model after analysis (specify dir)")
    print("  --help                   Show this help message and exit")
    print("\nExamples:")
    print("  python depression_detector.py --mode single --text \"I've been feeling sad lately\"")
    print("  python depression_detector.py --mode batch --file data.csv --output results.csv")
    print("  python depression_detector.py --mode interactive")

def main():
    """Main function adapted for HF model usage"""
    parser = argparse.ArgumentParser(description="Depression Detection System (HF Model)")
    parser.add_argument("--mode", choices=["single", "batch", "interactive", "demo"], 
                        default="demo", help="Operation mode")
    parser.add_argument("--text", type=str, help="Text to analyze (for single mode)")
    parser.add_argument("--file", type=str, help="Path to CSV file (for batch mode)")
    parser.add_argument("--column", type=str, default="text", 
                        help="Column name containing text (for batch mode)")
    parser.add_argument("--output", type=str, help="Path to save output (for batch mode)")
    parser.add_argument("--model", type=str, help="Path to a saved HuggingFace model directory")
    parser.add_argument("--save", type=str, help="Save the HuggingFace model after analysis (specify dir)")
    
    args = parser.parse_args()
    
    # Print banner (adjusted title)
    print("\n" + "="*60)
    print("Depression Detection System (HF Model) v1.0.0")
    print("="*60)
    print("DISCLAIMER: This system is for demonstration and research purposes only.")
    print("It is not a diagnostic tool and should not be used for clinical decisions.")
    print("="*60 + "\n")
    
    try:
        # Initialize the system, passing the HF model path if provided
        system = DepressionDetectionSystem(model_type="huggingface", model_path=args.model)
        
        # Run the specified mode (logic inside modes relies on updated system methods)
        if args.mode == "single":
            text = args.text
            if not text:
                print("Error: Text must be provided in single mode")
                print_usage()
                return
            
            result = system.predict(text)
            print("\nResult:")
            print(f"Predicted Severity: {result['depression_severity']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'guidance' in result:
                print(f"\nGuidance: {result['guidance']}")
        
        elif args.mode == "batch":
            file_path = args.file
            if not file_path:
                print("Error: File path must be provided in batch mode")
                print_usage()
                return
            
            results = system.batch_analyze(file_path, text_column=args.column, output_file=args.output)
            if results is not None:
                print("\nBatch Analysis Summary (HF Model):")
                if 'predicted_severity' in results.columns:
                     severity_counts = results['predicted_severity'].value_counts()
                     print(severity_counts)
                else:
                     print("Could not find 'predicted_severity' column in results.")
            
        elif args.mode == "interactive":
            system.interactive_mode()
            
        elif args.mode == "demo":
            print("Running HF model demonstration...")
            demo_single_text()
            demo_batch_analysis()
        
        # Save the model if requested
        if args.save:
            print(f"Saving HF model to {args.save}...")
            system.save_model(args.save)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print usage information
    print_usage()

if __name__ == "__main__":
    main() 