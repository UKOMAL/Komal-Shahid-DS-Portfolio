#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System - Consolidated Application

This module combines the core functionality for the depression detection system
including model definition, data processing, and application interface.
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
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

# Default paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

#-------------------------------------------------------------------------
# Model Implementation
#-------------------------------------------------------------------------

class TransformerDepressionModel:
    """
    BERT-based transformer model for depression detection
    
    This class implements a complete pipeline for depression detection using 
    a transformer-based approach
    """
    
    def __init__(self, num_classes=4):
        """
        Initialize the transformer model
        
        Args:
            num_classes: Number of depression severity categories
        """
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = None
        self.classes = None
        self.metadata = {
            "model_type": "BERT-based Transformer",
            "accuracy": None,
            "classes": ["minimum", "mild", "moderate", "severe"],
            "parameters": {
                "bert_variant": "small_bert/bert_en_uncased_L-4_H-512_A-8",
                "learning_rate": 3e-5
            }
        }
    
    def build(self):
        """
        Build the BERT-based transformer model
        
        Returns:
            Compiled TensorFlow model
        """
        # Load BERT preprocessing model from TF Hub
        preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            name="preprocessing"
        )
        
        # Load BERT encoder from TF Hub
        encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
            trainable=True,
            name="BERT_encoder"
        )
        
        # Input layer
        text_input = Input(shape=(), dtype=tf.string, name='text_input')
        
        # Preprocess data
        encoder_inputs = preprocessor(text_input)
        
        # Apply BERT encoder
        outputs = encoder(encoder_inputs)
        
        # Extract pooled output
        pooled_output = outputs["pooled_output"]
        
        # Add dropout layer
        x = Dropout(0.1)(pooled_output)
        
        # Add dense layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(self.num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
        
        # Create model
        model = Model(inputs=text_input, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=3e-5),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=16, epochs=5, callbacks=None):
        """
        Train the transformer model
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs
            callbacks: Optional callbacks for training
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build()
        
        # Encode labels if they're not already numeric
        if not isinstance(y_train[0], (int, np.integer)):
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            self.classes = self.label_encoder.classes_
            self.metadata["classes"] = self.classes.tolist()
            
            if X_val is not None and y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
                validation_data = (X_val, y_val_encoded)
            else:
                validation_data = None
        else:
            y_train_encoded = y_train
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        history = self.model.fit(
            X_train, y_train_encoded,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        Evaluate the transformer model
        
        Args:
            X_test: Test text data
            y_test: Test labels
            output_dir: Directory to save evaluation results
            
        Returns:
            Evaluation metrics
        """
        if not isinstance(y_test[0], (int, np.integer)) and self.label_encoder is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test_encoded)
        
        # Update metadata
        self.metadata["accuracy"] = float(accuracy)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1) if y_pred_prob.shape[1] > 1 else (y_pred_prob > 0.5).astype(int)
        
        # Create confusion matrix
        cm = tf.math.confusion_matrix(y_test_encoded, y_pred).numpy()
        
        # Plot confusion matrix if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=self.classes, yticklabels=self.classes)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Transformer Model Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'transformer_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Calculate metrics
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def predict(self, text):
        """
        Make predictions on new text data
        
        Args:
            text: Text to analyze, can be a single string or list of strings
            
        Returns:
            Tuple of (predicted_severity, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train or load a model first.")
        
        # Handle single string input
        is_single = isinstance(text, str)
        if is_single:
            text = [text]
        
        # Get raw predictions
        predictions = self.model.predict(text)
        
        # Get predicted classes and confidence
        if predictions.shape[1] > 1:  # Multi-class
            confidences = predictions
            predicted_ids = np.argmax(predictions, axis=1)
        else:  # Binary
            confidences = np.hstack([1 - predictions, predictions])
            predicted_ids = (predictions > 0.5).astype(int).flatten()
        
        # Convert to class names
        if self.classes is not None:
            predicted_labels = self.classes[predicted_ids]
        else:
            predicted_labels = predicted_ids
        
        # Return results
        if is_single:
            return predicted_labels[0], confidences[0]
        else:
            return predicted_labels, confidences
    
    def save(self, save_dir):
        """
        Save the model and its metadata
        
        Args:
            save_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(save_dir, 'model'))
        
        # Save metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save label encoder classes if available
        if self.classes is not None:
            with open(os.path.join(save_dir, 'classes.json'), 'w') as f:
                json.dump(self.classes.tolist(), f)
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def from_saved(cls, load_dir):
        """
        Load a saved model
        
        Args:
            load_dir: Directory where the model is saved
            
        Returns:
            Loaded TransformerDepressionModel instance
        """
        # Create a new instance
        instance = cls()
        
        # Load model
        instance.model = tf.keras.models.load_model(
            os.path.join(load_dir, 'model'),
            custom_objects={'KerasLayer': hub.KerasLayer}
        )
        
        # Load metadata
        with open(os.path.join(load_dir, 'metadata.json'), 'r') as f:
            instance.metadata = json.load(f)
        
        # Load classes if available
        try:
            with open(os.path.join(load_dir, 'classes.json'), 'r') as f:
                instance.classes = np.array(json.load(f))
        except FileNotFoundError:
            instance.classes = np.array(instance.metadata.get("classes", ["minimum", "mild", "moderate", "severe"]))
        
        return instance

#-------------------------------------------------------------------------
# Application Interface
#-------------------------------------------------------------------------

class DepressionDetectionSystem:
    """
    Unified interface for depression detection
    """
    
    def __init__(self, model_type="transformer", model_path=None):
        """
        Initialize the depression detection system
        
        Args:
            model_type: Type of model to use (currently only 'transformer' is fully implemented)
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

#-------------------------------------------------------------------------
# Demo Functions
#-------------------------------------------------------------------------

def demo_single_text():
    """Demo the system on a single text input"""
    print("\n=== Single Text Analysis Demo ===")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Example text for analysis
    text = "I haven't been feeling like myself lately. It's hard to get out of bed in the morning."
    
    # Analyze the text
    print(f"Analyzing text: \"{text}\"")
    result = system.predict(text)
    
    # Display results
    print("\nResult:")
    print(f"Depression Severity: {result['depression_severity']}")
    print("\nConfidence Scores:")
    for label, score in result['confidence_scores'].items():
        print(f"  {label}: {score:.2f}")

def demo_batch_analysis():
    """Demo the system on a batch of texts from a sample file"""
    print("\n=== Batch Analysis Demo ===")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Path to sample data
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              "data", "sample", "sample_depression_data.csv")
    
    if not os.path.exists(sample_path):
        print(f"Sample data not found at {sample_path}")
        return
    
    # Analyze the batch of texts
    print(f"Analyzing texts from: {sample_path}")
    results = system.batch_analyze(
        sample_path,
        text_column="text",
        output_file=os.path.join(DEFAULT_OUTPUT_DIR, "sample_results.csv")
    )
    
    # Display summary
    if results is not None:
        print("\nAnalysis Summary:")
        severity_counts = results['depression_severity'].value_counts()
        print(severity_counts)
        
        # Plot severity distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='depression_severity', data=results, 
                      order=['minimum', 'mild', 'moderate', 'severe'],
                      palette='viridis')
        plt.title('Depression Severity Distribution')
        plt.xlabel('Severity')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, "severity_distribution.png")
        plt.savefig(output_path)
        print(f"\nSeverity distribution plot saved to: {output_path}")

def print_usage():
    """Print usage information"""
    print("\nDepression Detection System - Usage")
    print("-----------------------------------")
    print("1. Single text analysis mode:")
    print("   python depression_detector.py --mode single")
    print("\n2. Batch analysis mode:")
    print("   python depression_detector.py --mode batch")
    print("\n3. Interactive mode:")
    print("   python depression_detector.py --mode interactive")
    print("\nFor more options:")
    print("   python depression_detector.py --help")

#-------------------------------------------------------------------------
# Main Function
#-------------------------------------------------------------------------

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Depression Detection System")
    parser.add_argument("--mode", choices=["single", "batch", "interactive"], 
                        default="interactive", help="Operation mode")
    parser.add_argument("--input_file", help="Input CSV file for batch processing")
    parser.add_argument("--text_column", default="text", 
                        help="Column name containing text in input file")
    parser.add_argument("--output_file", help="Output file path for batch results")
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("Depression Detection System - AI-Powered Screening Tool")
    print("="*60)
    print("DISCLAIMER: This system is for demonstration and research purposes only.")
    print("It is not a diagnostic tool and should not be used for clinical decisions.")
    print("="*60 + "\n")
    
    # Run in selected mode
    if args.mode == "single":
        demo_single_text()
    
    elif args.mode == "batch":
        if args.input_file:
            system = DepressionDetectionSystem(model_type="transformer")
            system.batch_analyze(
                args.input_file,
                text_column=args.text_column,
                output_file=args.output_file
            )
        else:
            demo_batch_analysis()
    
    elif args.mode == "interactive":
        system = DepressionDetectionSystem(model_type="transformer")
        system.interactive_mode()
    
    print_usage()

if __name__ == "__main__":
    main() 