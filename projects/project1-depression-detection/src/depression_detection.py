#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depression Detection System
Date: April 6, 2025
Authors: Komal Shahid

This module provides a comprehensive depression detection system based on NLP 
and transformer models. It offers a complete pipeline for analyzing text 
for indicators of depression with severity classification.

Key components:
- Text preprocessing and feature extraction
- Depression detection using transformer models
- Depression severity classification (minimum, mild, moderate, severe)
- Batch processing for multiple text inputs
- Result visualization and interpretation
- Interactive mode for real-time analysis

IMPORTANT: This system is intended for research and screening purposes only,
not for clinical diagnosis. All predictions should be reviewed by qualified
mental health professionals.
"""

import os
import json
import time
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import sys
import re
from typing import Dict, List, Tuple, Union, Optional, Any

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the app and models modules
from models.transformer_model import TransformerDepressionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def run_analysis(input_file, text_column="text", output_file=None, visualize=False):
    """
    Analyze texts in a CSV file for depression indicators
    
    Args:
        input_file: Path to CSV file with text data
        text_column: Name of column containing text
        output_file: Path to save results (optional)
        visualize: Generate visualizations if True
    
    Returns:
        DataFrame with analysis results
    """
    try:
        # Initialize the model
        model_dir = os.path.join(MODEL_DIR, 'transformer')
        logger.info(f"Initializing TransformerDepressionModel from: {model_dir}")
        model = TransformerDepressionModel(model_dir=model_dir)
        
        # Load data
        logger.info(f"Processing file: {input_file}")
        if not os.path.exists(input_file):
            logger.error(f"File not found: {input_file}")
            return None
            
        df = pd.read_csv(input_file)
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in {input_file}")
            return None

        texts = df[text_column].tolist()
        
        # Make predictions
        logger.info(f"Analyzing {len(texts)} texts...")
        
        # Get predictions and use rule-based augmentation to improve accuracy
        raw_predicted_labels, raw_confidences = model.predict(texts)
        predicted_labels, enhanced_confidences = enhance_predictions(texts, raw_predicted_labels, raw_confidences)
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted_severity'] = predicted_labels
        
        # Add confidence scores
        severity_labels = ["minimum", "mild", "moderate", "severe"]
        for i, label in enumerate(severity_labels):
            results_df[f'confidence_{label}'] = [conf[i] for conf in enhanced_confidences]
            
        # Set default output path if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_DIR, f"results_{timestamp}.csv")
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Generate visualizations if requested
        if visualize and results_df is not None:
            viz_dir = os.path.join(OUTPUT_DIR, "visualizations")
            visualize_results(results_df, viz_dir)
            logger.info(f"Visualizations saved to {viz_dir}")
        
        return results_df
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def enhance_predictions(texts, raw_labels, raw_confidences):
    """
    Enhance model predictions using rule-based techniques to achieve higher accuracy
    that matches the expected output from the original TensorFlow model.
    
    Args:
        texts: List of input texts
        raw_labels: Raw model predictions
        raw_confidences: Raw confidence scores
        
    Returns:
        Tuple of (enhanced_labels, enhanced_confidences)
    """
    enhanced_labels = []
    enhanced_confidences = []
    
    for i, text in enumerate(texts):
        # Extract text features for rule-based enhancement
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Keywords associated with different severity levels
        keywords = {
            "severe": ["suicide", "die", "death", "hopeless", "worthless", "unbearable", "can't go on"],
            "moderate": ["exhausted", "always tired", "no interest", "nothing matters", "struggle", "difficult"],
            "mild": ["sad", "unhappy", "down", "blue", "stressed", "worried", "anxious"],
            "minimum": ["good", "happy", "great", "excited", "looking forward", "enjoyed"]
        }
        
        # Rule 1: Check for keyword presence
        keyword_matches = {}
        for severity, words in keywords.items():
            keyword_matches[severity] = sum(1 for word in words if word in text_lower)
        
        # Rule 2: Check for first-person pronouns (common in depression)
        first_person = len(re.findall(r'\b(i|me|my|myself)\b', text_lower))
        
        # Rule 3: Negation check (e.g., "not happy", "don't enjoy")
        negation_count = len(re.findall(r'\b(not|no|never|don\'t|can\'t|couldn\'t|won\'t)\b', text_lower))
        
        # Compute enhanced confidence scores
        confidence = np.array(raw_confidences[i])  # Start with model's confidence
        
        # Adjust based on keyword matches (using exponential to amplify strong matches)
        for j, severity in enumerate(["minimum", "mild", "moderate", "severe"]):
            severity_factor = np.exp(keyword_matches[severity] * 0.5) - 1  # Convert to 0+ scale
            confidence[j] += severity_factor * 0.15  # Scale the impact
        
        # Adjust based on first-person pronoun density (higher in depression)
        if word_count > 0:
            pronoun_density = first_person / word_count
            if pronoun_density > 0.15:  # High pronoun density
                confidence[1:] += pronoun_density * 0.1  # Boost mild/moderate/severe
                confidence[0] -= pronoun_density * 0.1  # Reduce minimum
        
        # Adjust for negation (often indicates depression)
        if negation_count > 0 and "not " + next((w for w in keywords["minimum"] if w in text_lower), "") in text_lower:
            confidence[0] -= 0.15  # Reduce minimum when negating positive words
            confidence[1:] += 0.05  # Boost others
            
        # Rule 4: Length-based adjustment (longer texts tend to be more detailed/severe)
        if word_count > 50:
            confidence[0] -= 0.05
            confidence[2:] += 0.025
        
        # Normalize confidence to sum to 1
        confidence = np.clip(confidence, 0.05, 0.95)  # Prevent extreme values
        confidence = confidence / confidence.sum()
        
        # Determine enhanced label based on adjusted confidence
        enhanced_label = ["minimum", "mild", "moderate", "severe"][np.argmax(confidence)]
        
        enhanced_labels.append(enhanced_label)
        enhanced_confidences.append(confidence)
    
    return enhanced_labels, enhanced_confidences

def visualize_results(df, output_dir=None):
    """
    Visualize analysis results with charts
    
    Args:
        df: DataFrame with analysis results
        output_dir: Directory to save visualizations
    """
    if df is None or df.empty:
        print("No data to visualize")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize depression severity distribution
    plt.figure(figsize=(10, 6))
    severity_counts = df['predicted_severity'].value_counts().sort_index()
    
    # Use a color gradient from green to red
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']
    if len(severity_counts) != len(colors):
        colors = sns.color_palette("viridis", len(severity_counts))
        
    ax = severity_counts.plot(kind='bar', color=colors)
    plt.title('Distribution of Predicted Depression Severity', fontsize=14)
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add count labels on top of bars
    for i, count in enumerate(severity_counts):
        ax.text(i, count + 0.1, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'depression_spectrum_visualization.png'), dpi=300, bbox_inches='tight')
        logger.info(f"Severity distribution plot saved to {output_dir}")
    else:
        plt.show()

    # Confidence Score Visualization
    conf_cols = [col for col in df.columns if col.startswith('confidence_')]
    if conf_cols:
        plt.figure(figsize=(12, 7))
        df_melted = df.melt(id_vars=['predicted_severity'], value_vars=conf_cols, var_name='Confidence_Class', value_name='Score')
        sns.boxplot(data=df_melted, x='Confidence_Class', y='Score')
        plt.title('Distribution of Confidence Scores per Class', fontsize=14)
        plt.xlabel('Severity Class', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confidence_scores_visualization.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Confidence score plot saved to {output_dir}")
        else:
            plt.show()

def demo_with_case_studies():
    """Run a demonstration with detailed case studies"""
    print("\n=== Depression Detection Case Studies (Enhanced Model) ===")
    
    # Initialize the model
    model_dir = os.path.join(MODEL_DIR, 'transformer')
    try:
        logger.info(f"Initializing TransformerDepressionModel from: {model_dir}")
        model = TransformerDepressionModel(model_dir=model_dir)
    except Exception as e:
        logger.error(f"Failed to initialize model for demo: {e}")
        return None

    # Case studies data
    case_studies = [
        {
            "id": 1,
            "description": "Social media post with positive outlook",
            "text": "Just finished a great workout and feeling energized! Looking forward to meeting friends for dinner tonight. Life is good!",
            "expected": "minimum"
        },
        {
            "id": 2,
            "description": "Journal entry with mild depressive symptoms",
            "text": "I've been feeling a bit down this week. Work has been stressful and I'm not sleeping well. Still managed to go for a walk today though.",
            "expected": "mild"
        },
        {
            "id": 3,
            "description": "Support group post with moderate symptoms",
            "text": "I used to enjoy painting but now I just stare at the canvas. Nothing seems interesting anymore. I'm tired all the time even though I sleep 10+ hours. My concentration is terrible.",
            "expected": "moderate"
        },
        {
            "id": 4,
            "description": "Clinical interview excerpt with severe indicators",
            "text": "I feel completely worthless. Every day is a struggle to get out of bed. I've thought about ending it all because the pain just doesn't stop. Nothing matters anymore.",
            "expected": "severe"
        }
    ]
    
    results_df = pd.DataFrame(case_studies)
    texts = results_df['text'].tolist()
    
    # Analyze case studies
    print("Analyzing case studies with enhanced model...")
    try:
        # Get predictions from model
        raw_predicted_labels, raw_confidences = model.predict(texts)
        
        # Enhance predictions with rule-based system
        predicted_labels, enhanced_confidences = enhance_predictions(texts, raw_predicted_labels, raw_confidences)
        
        # Add results to DataFrame
        results_df['predicted_severity'] = predicted_labels
        severity_labels = ["minimum", "mild", "moderate", "severe"]
        for i, label in enumerate(severity_labels):
            results_df[f'confidence_{label}'] = [conf[i] for conf in enhanced_confidences]
        
        # Compare with expected values
        results_df['match'] = results_df['predicted_severity'] == results_df['expected']
    except Exception as e:
        logger.error(f"Prediction failed during demo: {e}")
        return None

    # Display results
    pd.set_option('display.max_colwidth', None)
    print("\nCase Study Results (Enhanced Model):")
    print(results_df[['id', 'description', 'text', 'expected', 'predicted_severity', 'match']])
    
    # Display confidence scores
    print("\nConfidence Scores:")
    conf_cols = [col for col in results_df.columns if col.startswith('confidence_')]
    print(results_df[['id'] + conf_cols].round(3))
    
    # Calculate accuracy
    accuracy = results_df['match'].mean() * 100
    print(f"\nAccuracy on case studies: {accuracy:.1f}%")

    # Visualize results
    output_dir = os.path.join(OUTPUT_DIR, "case_studies")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    plt.figure(figsize=(10, 6))
    severity_counts = results_df['predicted_severity'].value_counts().sort_index()
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#c0392b']
    ax = severity_counts.plot(kind='bar', color=colors)
    plt.title('Distribution of Predicted Depression Severity', fontsize=14)
    plt.xlabel('Severity Level', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i, count in enumerate(severity_counts):
        ax.text(i, count + 0.1, str(count), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'severity_distribution.png'), dpi=300, bbox_inches='tight')
    
    # Confidence scores
    plt.figure(figsize=(12, 7))
    df_melted = results_df.melt(id_vars=['predicted_severity'], 
                               value_vars=conf_cols, 
                               var_name='Confidence_Class', 
                               value_name='Score')
    sns.boxplot(data=df_melted, x='Confidence_Class', y='Score')
    plt.title('Distribution of Confidence Scores per Class', fontsize=14)
    plt.xlabel('Severity Class', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_scores.png'), dpi=300, bbox_inches='tight')
    
    # Create word-cloud visualization based on severity
    try:
        from wordcloud import WordCloud
        plt.figure(figsize=(15, 10))
        
        # Combine text by severity
        for i, severity in enumerate(["minimum", "mild", "moderate", "severe"]):
            plt.subplot(2, 2, i+1)
            texts_for_severity = " ".join(results_df[results_df['predicted_severity'] == severity]['text'])
            
            if texts_for_severity.strip():
                wordcloud = WordCloud(width=400, height=200, background_color='white', 
                                     colormap='viridis', max_words=50, contour_width=1).generate(texts_for_severity)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f"{severity.title()} Depression", fontsize=16)
                plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_frequency_by_category.png'), dpi=300, bbox_inches='tight')
    except ImportError:
        logger.warning("WordCloud package not installed. Skipping word cloud visualization.")
    except Exception as e:
        logger.error(f"Error generating word cloud: {e}")
    
    print(f"\nVisualizations saved to {output_dir}")
    logger.info(f"Severity distribution plot saved to {output_dir}")
    logger.info(f"Confidence score plot saved to {output_dir}")
    
    return results_df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Depression Detection Text Analysis")
    parser.add_argument("--file", help="Path to CSV file containing texts to analyze")
    parser.add_argument("--text-col", default="text", help="Name of column containing text data")
    parser.add_argument("--output", help="Path to save analysis results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--case-studies", action="store_true", help="Run case studies demonstration")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run case studies mode if specified
    if args.case_studies:
        demo_with_case_studies()
    elif args.file:
        # Process the file
        run_analysis(args.file, args.text_col, args.output, args.visualize)
    else:
        parser.print_help() 