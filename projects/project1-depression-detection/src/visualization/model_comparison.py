#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Comparison Visualization
Generate a bar chart comparing the performance of different model architectures
for depression detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance_comparison(output_file=None):
    """
    Plot a comparison of model performance from the research findings.
    
    Args:
        output_file (str): Path to save the visualization
    """
    # Model names and their accuracy scores from the white paper
    models = [
        'Logistic Regression',
        'Random Forest',
        'Support Vector Machine',
        'Gradient Boosting',
        'LSTM with GloVe',
        'BiLSTM with Attention',
        'Fine-tuned BERT-base',
        'Fine-tuned RoBERTa'
    ]
    
    # Accuracy scores and confidence intervals from the white paper
    accuracies = [
        61.03,  # Logistic Regression
        62.18,  # Random Forest
        63.45,  # Support Vector Machine
        66.22,  # Gradient Boosting
        70.34,  # LSTM with GloVe
        72.18,  # BiLSTM with Attention
        75.92,  # Fine-tuned BERT-base
        78.50   # Fine-tuned RoBERTa
    ]
    
    # Confidence interval lower and upper bounds
    ci_lower = [
        60.26,  # Logistic Regression
        61.42,  # Random Forest
        62.70,  # Support Vector Machine
        65.48,  # Gradient Boosting
        69.62,  # LSTM with GloVe
        71.47,  # BiLSTM with Attention
        75.24,  # Fine-tuned BERT-base
        77.84   # Fine-tuned RoBERTa
    ]
    
    ci_upper = [
        61.80,  # Logistic Regression
        62.94,  # Random Forest
        64.20,  # Support Vector Machine
        66.96,  # Gradient Boosting
        71.06,  # LSTM with GloVe
        72.89,  # BiLSTM with Attention
        76.60,  # Fine-tuned BERT-base
        79.16   # Fine-tuned RoBERTa
    ]
    
    # Calculate error bars for plt.bar
    yerr = np.zeros((2, len(models)))
    yerr[0, :] = np.array(accuracies) - np.array(ci_lower)
    yerr[1, :] = np.array(ci_upper) - np.array(accuracies)
    
    # Create color palette - gradient from light blue to dark blue
    # Traditional ML models in blue, Deep Learning models in purple
    colors = sns.color_palette("Blues", 4)[::-1] + sns.color_palette("Purples", 4)[::-1]
    
    # Setup the figure
    plt.figure(figsize=(14, 8))
    
    # Create bar chart
    bars = plt.bar(models, accuracies, color=colors, width=0.6)
    
    # Add error bars
    plt.errorbar(models, accuracies, yerr=yerr, fmt='none', ecolor='black', capsize=5, elinewidth=1.5, capthick=1.5)
    
    # Add model performance values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7, 
                 f"{accuracies[i]:.2f}%", 
                 ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add plot labels and title
    plt.title('Depression Detection Model Performance Comparison', fontsize=16, pad=20)
    plt.xlabel('Model Architecture', fontsize=14, labelpad=10)
    plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
    
    # Add a horizontal line at the best traditional ML model performance
    plt.axhline(y=66.22, color='lightgray', linestyle='--', alpha=0.7)
    plt.text(0, 66.7, 'Best Traditional ML', fontsize=10, alpha=0.7)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Y-axis from 55 to 85 to show differences better
    plt.ylim(55, 85)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Divide the chart with categories
    plt.axvline(x=3.5, color='lightgray', linestyle='-', alpha=0.5)
    plt.text(1.5, 83, 'Traditional ML Models', ha='center', fontsize=12, color='gray')
    plt.text(5.5, 83, 'Deep Learning Models', ha='center', fontsize=12, color='gray')
    
    # Add annotations for improvement
    plt.annotate('', xy=(7, 78.5), xytext=(3, 66.22),
                arrowprops=dict(arrowstyle='fancy', color='green', alpha=0.5, 
                                connectionstyle="arc3,rad=.2"))
    plt.text(5, 73, '12.28% improvement', fontsize=11, color='green', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot if output file is specified
    if output_file:
        dir_path = os.path.dirname(output_file)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Model comparison visualization saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Default output location
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
        "model_comparison.png"
    )
    
    plot_model_performance_comparison(output_path) 