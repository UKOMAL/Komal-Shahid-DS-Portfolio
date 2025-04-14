#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Model for Depression Detection
Implementation of the BERT-based transformer model used for depression severity classification
"""

import os
import json
import numpy as np
import torch # Use PyTorch for HuggingFace models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Needed for saving metadata timestamps potentially
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
from torch.utils.data import Dataset, DataLoader # For potential training dataset structure
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # For evaluation
from sklearn.preprocessing import LabelEncoder

# Configure logging (can be defined here or imported)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TransformerDepressionModel:
    """
    HuggingFace Transformer-based model for depression detection.
    Loads model/tokenizer from a directory (HF cache format expected).
    """
    
    def __init__(self, model_dir: str = None, num_labels: int = 4):
        """
        Initialize the transformer model by loading from a directory.
        
        Args:
            model_dir: Path to the directory containing the HF model files 
                       (e.g., the copied cache dir like 'models/transformer').
                       If None, tries a default path.
            num_labels: Number of classification labels (e.g., 4 for severity).
        """
        # Default model directory within the project structure
        DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "transformer")
        
        self.model_dir = model_dir if model_dir else DEFAULT_MODEL_DIR
        self.num_labels = num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.config = {}
        self.classes = ["minimum", "mild", "moderate", "severe"] # Default

        self.load(self.model_dir) # Load model during initialization

    def load(self, model_path: str):
        """
        Load a saved HuggingFace model and tokenizer from a directory.
        
        Args:
            model_path: Path to the directory containing model files.
                      (e.g., config.json, pytorch_model.bin or model.safetensors, tokenizer_config.json etc.)
        """
        try:
            if not os.path.isdir(model_path):
                 raise FileNotFoundError(f"Model directory not found: {model_path}")
                 
            logger.info(f"Loading HuggingFace model and tokenizer from {model_path}")
            
            # Load configuration first to get label mappings etc.
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                # Try to infer classes from config (id2label is common)
                if 'id2label' in self.config:
                     id2label = {int(k): v for k, v in self.config['id2label'].items()} # Ensure keys are ints
                     self.classes = [id2label[i] for i in sorted(id2label.keys())]
                     self.num_labels = len(self.classes)
                     logger.info(f"Inferred classes from config: {self.classes}")
                elif 'classes' in self.config:
                     self.classes = self.config['classes']
                     self.num_labels = len(self.classes)
                     logger.info(f"Loaded classes from config: {self.classes}")
                else:
                     logger.warning("Could not infer class labels from config.json. Using default.")
            else:
                 logger.warning(f"config.json not found in {model_path}. Cannot infer class labels.")
                 # Keep default classes if config is missing
                 self.num_labels = len(self.classes)
            
            # Load tokenizer and model using from_pretrained with the directory path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                 model_path, 
                 num_labels=self.num_labels # Ensure model output matches labels
            )
            self.model.to(self.device)
            logger.info("HuggingFace model and tokenizer loaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error loading HuggingFace model from {model_path}: {str(e)}")
            # Fallback or re-raise
            self.tokenizer = None
            self.model = None
            # Optionally, raise the error to prevent using an uninitialized model
            raise ValueError(f"Failed to load model from {model_path}") from e

    def save(self, save_dir: str):
        """
        Save the HuggingFace model and tokenizer to disk.
        
        Args:
            save_dir: Directory to save the model files.
        """
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not loaded. Cannot save.")
            return False
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"Saving HuggingFace model and tokenizer to {save_dir}")
            
            # Save model and tokenizer
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            
            # Optionally save the config/metadata explicitly if needed
            # config_path = os.path.join(save_dir, "config.json")
            # with open(config_path, 'w') as f:
            #     json.dump(self.config, f, indent=2)
            
            logger.info("Model saved successfully.")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_dir}: {str(e)}")
            return False

    def predict(self, text: Union[str, List[str]]) -> Union[Tuple[str, np.ndarray], Tuple[List[str], List[np.ndarray]]]:
        """
        Make predictions using the loaded HuggingFace model.
        
        Args:
            text: Text to analyze (single string or list of strings).
            
        Returns:
            If single string: (predicted_label, confidence_scores_array)
            If list of strings: (list_of_predicted_labels, list_of_confidence_scores_arrays)
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not loaded. Cannot predict.")

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            # Prepare inputs
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.get('max_length', 512), # Use max_length from config or default
                return_tensors="pt"
            ).to(self.device)

            # Get predictions
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Move probabilities to CPU and convert to numpy
            probabilities_np = probabilities.cpu().numpy()
            
            # Get predicted class indices
            predicted_ids = np.argmax(probabilities_np, axis=1)
            
            # Map indices to class labels
            predicted_labels = [self.classes[i] for i in predicted_ids]
            
            # Return based on input type
            if is_single:
                return predicted_labels[0], probabilities_np[0]
            else:
                return predicted_labels, [prob for prob in probabilities_np] # Return list of arrays

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return something indicative of an error or re-raise
            # Depending on how the caller (DepressionDetectionSystem) handles it
            if is_single:
                 # Need to define how errors are returned, maybe raise exception
                 raise RuntimeError(f"Prediction failed for text: {text}") from e
            else:
                 # For batch, maybe return empty lists or raise
                 raise RuntimeError(f"Batch prediction failed.") from e

    # --- Methods removed or needing adaptation --- 
    # build() -> Not needed, model loaded via from_pretrained
    # fit() -> Would need reimplementation using HF Trainer or custom PyTorch loop
    # evaluate() -> Needs reimplementation using predict() and sklearn metrics
    # from_saved() -> Replaced by load() method using from_pretrained

    # Add a simple train method placeholder if needed for structure
    def train(self, *args, **kwargs):
         logger.warning("Training method not fully implemented in this version.")
         print("To train, use HuggingFace Trainer API or a custom PyTorch loop.")
         # Placeholder for potential future implementation
         pass

    # Add evaluate placeholder
    def evaluate(self, test_texts: List[str], test_labels: List[int], output_dir=None):
        logger.warning("Evaluation method using HF model needs implementation.")
        print("Predicting on test data...")
        try:
             predicted_labels, _ = self.predict(test_texts)
             
             # Convert string labels back to indices if necessary for metrics
             # This requires a consistent label encoding
             le = LabelEncoder()
             le.fit(self.classes) # Fit on the known classes
             true_indices = le.transform([self.classes[i] for i in test_labels]) # Assuming test_labels are indices
             pred_indices = le.transform(predicted_labels)

             accuracy = accuracy_score(true_indices, pred_indices)
             report = classification_report(true_indices, pred_indices, target_names=self.classes, output_dict=True)
             cm = confusion_matrix(true_indices, pred_indices)

             metrics = {
                 'accuracy': accuracy,
                 'classification_report': report,
                 'confusion_matrix': cm.tolist()
             }
             print(f"Accuracy: {accuracy:.4f}")
             print("Classification Report:\n", classification_report(true_indices, pred_indices, target_names=self.classes))

             # Optional: Plot confusion matrix
             if output_dir is not None:
                 os.makedirs(output_dir, exist_ok=True)
                 plt.figure(figsize=(10, 8))
                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=self.classes, yticklabels=self.classes)
                 plt.xlabel('Predicted')
                 plt.ylabel('True')
                 plt.title('Confusion Matrix (HuggingFace Model)')
                 plt.tight_layout()
                 plt.savefig(os.path.join(output_dir, 'hf_confusion_matrix.png'), dpi=300, bbox_inches='tight')
                 plt.close()
                 print(f"Confusion matrix saved to {output_dir}")

             return metrics
        except Exception as e:
             logger.error(f"Error during evaluation: {e}")
             return {"error": str(e)}

# Note: The DepressionDataset class would be needed if implementing training
class DepressionDataset(Dataset):
    """
    Simple PyTorch Dataset for text classification.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure encodings contain tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Ensure labels are tensors
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def demo():
    """
    Demonstrate the HuggingFace transformer model loading and prediction.
    """
    print("\n=== HuggingFace Transformer Model Demo ===")
    # Assume model files are in 'models/transformer' relative to project root
    # Adjust this path if needed
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_directory = os.path.join(project_root, "models", "transformer")
    
    try:
        print(f"Attempting to load model from: {model_directory}")
        # Initialize by loading the model
        model = TransformerDepressionModel(model_dir=model_directory)
        
        # Create sample data
        samples = [
            "I feel great today and am looking forward to the future.",
            "I've been feeling a bit down lately, but it's not too bad.",
            "I find it hard to concentrate and sleep these days.",
            "Nothing matters anymore. I feel completely hopeless."
        ]
        expected_labels = ["minimum", "mild", "moderate", "severe"]
        
        print("\nMaking sample predictions...")
        predicted_labels, confidences = model.predict(samples)
        
        for i, sample in enumerate(samples):
            print(f"\nSample text: \"{sample}\"")
            print(f"  Expected: {expected_labels[i]}")
            print(f"  Predicted: {predicted_labels[i]}")
            print(f"  Confidence Scores: {dict(zip(model.classes, confidences[i].round(3)))}")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        print("Ensure the model files exist in the specified directory and are compatible.")

if __name__ == "__main__":
    demo() 