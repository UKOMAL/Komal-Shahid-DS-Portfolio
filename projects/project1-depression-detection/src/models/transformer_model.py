#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer Model for Depression Detection
Implementation of the BERT-based transformer model used for depression severity classification
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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

def demo():
    """
    Demonstrate the transformer model on sample data
    """
    # Create sample data
    samples = [
        "I feel great today and am looking forward to the future.",
        "I've been feeling a bit down lately, but it's not too bad.",
        "I find it hard to concentrate and sleep these days.",
        "Nothing matters anymore. I feel completely hopeless."
    ]
    labels = ["minimum", "mild", "moderate", "severe"]
    
    # Create and train model
    model = TransformerDepressionModel()
    model.build()
    
    # Simulate training (in practice, real training would be performed)
    print("Model would be trained on real data. For demonstration, we'll simulate predictions.")
    
    # Make predictions (this would normally use a trained model)
    # For demo purposes, we'll just show the API
    print("\nSample predictions (demonstration only):")
    for i, sample in enumerate(samples):
        print(f"\nSample text: \"{sample}\"")
        print(f"Expected severity: {labels[i]}")
        print(f"Model would predict severity and confidence scores")
    
    print("\nIn a real application, the model would be trained on depression text data")
    print("and would make actual predictions based on the trained parameters.")

if __name__ == "__main__":
    demo() 