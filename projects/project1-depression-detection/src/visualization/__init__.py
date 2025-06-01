"""
Depression Detection Visualization Package

This package contains functions for visualizing depression detection results and models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

def plot_attention(attention_weights, input_text, output_file):
    """Plot the attention weights for a given input text.
    
    Args:
        attention_weights (numpy.ndarray): 2D array of attention weights 
            with shape (num_heads, sequence_length).
        input_text (str): The input text corresponding to the attention weights.
        output_file (str): File path to save the attention plot.
    """
    # Average attention weights across all heads
    avg_attention = attention_weights.mean(axis=0)

    # Normalize attention weights 
    norm_attention = avg_attention / avg_attention.max()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(np.expand_dims(norm_attention, axis=0), 
                cmap="Blues", vmax=1, vmin=0, 
                xticklabels=input_text.split(), yticklabels=["Attention"],
                ax=ax)
    ax.set_title("Attention Visualization")
    ax.set_xlabel("Input Text")
    ax.set_ylabel("Attention Weight")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200)

def plot_feature_importance(model, X, y, feature_names, output_file):
    """Plot the feature importance scores for the model.
    
    Args:
        model: Trained model object.
        X (numpy.ndarray): Input features.
        y (numpy.ndarray): Target labels. 
        feature_names (list): List of feature names.
        output_file (str): File path to save the feature importance plot.
    """
    # Calculate permutation feature importance
    importance = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    
    sorted_idx = importance.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.boxplot(importance.importances[sorted_idx].T, 
               labels=[feature_names[i] for i in sorted_idx],
               vert=False)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    fig.savefig(output_file, dpi=200) 