#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System Demo
This script demonstrates the key functionality of the depression detection system
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from depression_detection import DepressionDetectionSystem
from models.transformer_model import TransformerDepressionModel
from visualization.attention import plot_attention_weights
from visualization.feature_importance import plot_feature_importance

def analyze_text(text):
    """Analyze a single text and display the results"""
    print(f"\nAnalyzing text: \"{text}\"")
    
    # Initialize the depression detection system
    system = DepressionDetectionSystem(model_type="transformer")
    
    # Analyze the text
    result = system.predict(text)
    
    # Display results
    print("\nResult:")
    print(f"Depression Severity: {result['depression_severity']}")
    print("\nConfidence Scores:")
    for label, score in result['confidence_scores'].items():
        print(f"  {label}: {score:.2f}")
    
    # Plot attention weights
    attention_weights = result['attention_weights']
    plot_attention_weights(text, attention_weights, 
                           output_file=os.path.join("demo", "output", "attention_weights.png"))

def visualize_results(results_file):
    """Visualize the results from a batch analysis"""
    print(f"\nVisualizing results from: {results_file}")
    
    # Load results
    results = pd.read_csv(results_file)
    
    # Plot severity distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='depression_severity', data=results, 
                  order=['minimum', 'mild', 'moderate', 'severe'],
                  palette='viridis')
    plt.title('Depression Severity Distribution')
    plt.xlabel('Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join("demo", "output", "severity_distribution.png"))
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='depression_severity', y='sentiment_score', data=results,
                   order=['minimum', 'mild', 'moderate', 'severe'],
                   palette='viridis')
    plt.title('Sentiment Distribution by Depression Severity')
    plt.xlabel('Depression Severity')
    plt.ylabel('Sentiment Polarity')
    plt.tight_layout() 
    plt.savefig(os.path.join("demo", "output", "sentiment_distribution.png"))
    
    # Calculate evaluation metrics
    y_true = results['depression_severity']
    y_pred = results['predicted_severity']
    
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\nEvaluation Metrics:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}") 
    print(f"F1 Score: {f1:.2f}")
    
    # Plot feature importance
    plot_feature_importance(results, 
                            output_file=os.path.join("demo", "output", "feature_importance.png"))

def main():
    """Main demo function"""
    # Print banner
    print("="*60)
    print("Depression Detection System Demo")
    print("="*60)
    
    # Analyze individual texts
    texts = [
        "I feel like I'm stuck in a rut and can't get out.",
        "I'm so excited about my upcoming vacation!",
        "I don't know why I even bother trying anymore.",
        "I'm really proud of what I accomplished today."
    ]
    for text in texts:
        analyze_text(text)
    
    # Visualize batch results
    results_file = os.path.join("output", "sample_results.csv")
    visualize_results(results_file)

if __name__ == "__main__":
    main() 