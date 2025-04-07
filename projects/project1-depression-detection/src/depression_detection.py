#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System

This module provides a unified interface for depression detection
using various models including traditional machine learning and
deep learning approaches.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import models
from models.transformer_model import TransformerDepressionModel

# Default paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

class DepressionDetectionSystem:
    """
    Unified interface for depression detection using multiple models
    """
    
    def __init__(self, model_type="transformer", model_path=None):
        """
        Initialize the depression detection system
        
        Args:
            model_type: Type of model to use ('transformer', 'lstm', or 'gradient_boosting')
            model_path: Path to pre-trained model (if None, will use default path)
        """
        self.model_type = model_type
        self.model = None
        self.model_path = model_path or os.path.join(DEFAULT_MODEL_DIR, model_type)
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the appropriate model based on model_type"""
        if self.model_type == "transformer":
            # Try to load pre-trained model, or create a new one
            if os.path.exists(self.model_path):
                print(f"Loading pre-trained transformer model from {self.model_path}")
                self.model = TransformerDepressionModel.from_saved(self.model_path)
            else:
                print("Creating new transformer model")
                self.model = TransformerDepressionModel()
        
        elif self.model_type == "lstm":
            # This would be implemented similarly for LSTM model
            print("LSTM model not yet implemented")
            # self.model = LSTMDepressionModel.from_saved(self.model_path)
        
        elif self.model_type == "gradient_boosting":
            # This would be implemented for traditional ML models
            print("Gradient Boosting model not yet implemented")
            # self.model = GradientBoostingModel.from_saved(self.model_path)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(self, text):
        """
        Analyze text for depression indicators
        
        Args:
            text: Text to analyze, can be a single string or list of strings
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized or loaded")
        
        # Get prediction
        severity, confidence = self.model.predict(text)
        
        # Format result
        if isinstance(text, str):
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "depression_severity": severity,
                "confidence_scores": {
                    label: float(score) for label, score in 
                    zip(self.model.classes, confidence)
                },
                "timestamp": datetime.now().isoformat(),
                "model_type": self.model_type
            }
        else:
            result = [{
                "text": t[:100] + "..." if len(t) > 100 else t,
                "depression_severity": s,
                "confidence_scores": {
                    label: float(score) for label, score in 
                    zip(self.model.classes, conf)
                },
                "timestamp": datetime.now().isoformat(),
                "model_type": self.model_type
            } for t, s, conf in zip(text, severity, confidence)]
        
        return result
    
    def batch_analyze(self, file_path, text_column="text", output_file=None):
        """
        Analyze a batch of texts from a CSV file
        
        Args:
            file_path: Path to CSV file containing texts
            text_column: Name of column containing text to analyze
            output_file: Path to save results (if None, will return results)
            
        Returns:
            DataFrame with prediction results (if output_file is None)
        """
        # Load data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        
        if text_column not in df.columns:
            print(f"Column '{text_column}' not found in file. Available columns: {df.columns.tolist()}")
            return None
        
        # Get predictions
        texts = df[text_column].tolist()
        print(f"Analyzing {len(texts)} texts...")
        
        # Process in batches to avoid memory issues with large files
        batch_size = 32
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.predict(batch_texts)
            all_results.extend(batch_results)
            print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} texts")
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add original data
        for col in df.columns:
            if col != text_column:  # Avoid duplicate text column
                results_df[col] = df[col].values
        
        # Save results if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results_df
    
    def interactive_mode(self):
        """
        Run an interactive session for depression detection
        """
        print("\n=== Depression Detection System - Interactive Mode ===")
        print(f"Using model: {self.model_type}")
        print("Enter text to analyze, or 'q' to quit.")
        print("Disclaimer: This is a screening tool only and not a diagnostic system.\n")
        
        while True:
            text = input("\nEnter text: ")
            if text.lower() in ('q', 'quit', 'exit'):
                break
            
            if not text.strip():
                print("Please enter some text to analyze.")
                continue
            
            try:
                result = self.predict(text)
                
                # Format and display result
                print("\n----- Analysis Result -----")
                print(f"Depression Severity: {result['depression_severity']}")
                print("\nConfidence Scores:")
                for label, score in result['confidence_scores'].items():
                    print(f"  {label}: {score:.2f}")
                
                print("\nNote: This analysis is for screening purposes only and")
                print("should not replace professional medical evaluation.")
            except Exception as e:
                print(f"Error analyzing text: {e}")
    
    def save_model(self, save_path=None):
        """Save the current model"""
        save_path = save_path or self.model_path
        if self.model is not None:
            self.model.save(save_path)
        else:
            print("No model to save")


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Depression Detection System")
    parser.add_argument("--model", choices=["transformer", "lstm", "gradient_boosting"], 
                        default="transformer", help="Model type to use")
    parser.add_argument("--model_path", help="Path to pre-trained model")
    parser.add_argument("--mode", choices=["interactive", "file"], default="interactive",
                        help="Operation mode")
    parser.add_argument("--input_file", help="Input CSV file for batch processing")
    parser.add_argument("--text_column", default="text", 
                        help="Column name containing text in input file")
    parser.add_argument("--output_file", help="Output file path for batch results")
    
    args = parser.parse_args()
    
    # Initialize system
    system = DepressionDetectionSystem(model_type=args.model, model_path=args.model_path)
    
    # Run in selected mode
    if args.mode == "interactive":
        system.interactive_mode()
    
    elif args.mode == "file":
        if not args.input_file:
            print("Error: Input file required for file mode")
            return
        
        output_file = args.output_file or os.path.join(
            DEFAULT_OUTPUT_DIR, 
            f"depression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        system.batch_analyze(
            args.input_file,
            text_column=args.text_column,
            output_file=output_file
        )

if __name__ == "__main__":
    main() 