#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Importance Visualization Generator

This script creates a feature importance visualization for the depression detection system
showing the relative importance of different linguistic features in predicting depression severity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to system path to allow imports from src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)

def generate_feature_importance_visualization():
    """
    Generates a feature importance visualization showing the relative importance
    of different linguistic features in predicting depression severity.
    """
    # Define the top features with their importance scores
    # These values are taken from the white paper
    features = {
        'First-person pronouns': 0.089,
        'Negative emotion words': 0.078,
        'Absolutist terms': 0.062,
        'Social disconnection': 0.057,
        'Cognitive distortions': 0.053,
        'Sadness indicators': 0.054,
        'Temporal focus (past)': 0.047,
        'Social references': 0.041,
        'Anxiety indicators': 0.038,
        'Anger indicators': 0.032,
        'Hopelessness expressions': 0.028,
        'Self-focus language': 0.025,
        'Worthlessness indicators': 0.022,
        'Fatigue references': 0.019,
        'Sleep disruption': 0.015
    }
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Importance': list(features.values())
    })
    
    # Sort by importance
    df = df.sort_values('Importance', ascending=True)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=df, palette='viridis')
    
    # Customize the plot
    plt.title('Feature Importance for Depression Detection', fontsize=16, fontweight='bold')
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Linguistic Feature', fontsize=12)
    
    # Add the values at the end of the bars
    for i, v in enumerate(df['Importance']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10)
    
    # Add a grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Feature importance visualization saved to {output_path}")
    return output_path

if __name__ == "__main__":
    generate_feature_importance_visualization() 