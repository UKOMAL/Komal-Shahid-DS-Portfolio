#!/usr/bin/env python3
"""
Colorful Canvas: AI Art Studio - Full Pipeline

This script runs the complete pipeline for the Colorful Canvas project:
1. Fetches data from APIs
2. Performs exploratory data analysis with visualizations
3. Trains the models, and generates 3D anamorphic results from 2D images

Usage:
    python run_pipeline.py --data_source [all|optical|depth|fashion|color|social|advertising]
                          --train [all|illusion|depth|performance]
                          --generate_samples [num_samples]
                          --no_cache
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from tqdm import tqdm
import seaborn as sns
from PIL import Image, ImageDraw

# Import local modules
from data.data_loader import DataLoader
from models.train_models import IllusionPredictor, DepthEstimator, PerformancePredictor

# Output directories
OUTPUT_DIR = Path("./output")
MODELS_DIR = Path("./models/ml")
FIGURES_DIR = Path("./output/figures")
ANAMORPHIC_DIR = Path("./output/anamorphic")

def setup_directories():
    """Create necessary directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ANAMORPHIC_DIR, exist_ok=True)
    print(f"Setup directories: {OUTPUT_DIR}, {MODELS_DIR}, {FIGURES_DIR}, {ANAMORPHIC_DIR}")

def fetch_datasets(loader, args):
    """Fetch all datasets based on arguments"""
    datasets = {}
    
    if args.data_source in ['all', 'optical']:
        print("\n===== Fetching Optical Illusions Dataset =====")
        datasets['optical'] = loader.fetch_optical_illusions_dataset(max_samples=args.max_samples)
    
    if args.data_source in ['all', 'depth']:
        print("\n===== Fetching 3D Depth Dataset =====")
        datasets['depth'] = loader.fetch_3d_depth_dataset(max_samples=args.max_samples)
    
    if args.data_source in ['all', 'fashion']:
        print("\n===== Fetching 3D Fashion Dataset =====")
        datasets['fashion'] = loader.fetch_fashion_3d_dataset(max_samples=args.max_samples)
    
    if args.data_source in ['all', 'color']:
        print("\n===== Fetching Color Psychology Dataset =====")
        datasets['color'] = loader.fetch_color_psychology_dataset(max_samples=args.max_samples)
    
    if args.data_source in ['all', 'social']:
        print("\n===== Fetching Social Media Engagement Dataset =====")
        datasets['social'] = loader.fetch_social_media_engagement_data(max_samples=args.max_samples)
    
    if args.data_source in ['all', 'advertising']:
        print("\n===== Fetching Advertising Performance Dataset =====")
        datasets['advertising'] = loader.fetch_advertising_performance_data(max_samples=args.max_samples)
    
    return datasets

def perform_eda(datasets):
    """Perform exploratory data analysis with visualizations"""
    print("\n===== Performing Exploratory Data Analysis =====")
    
    # Create a results dictionary
    eda_results = {}
    
    # Analyze optical illusions dataset
    if 'optical' in datasets:
        print("\nAnalyzing optical illusions dataset...")
        optical = datasets['optical']
        
        # Distribution of categories
        categories = [f['category'] for f in optical['features']]
        category_counts = pd.Series(categories).value_counts()
        
        plt.figure(figsize=(10, 6))
        category_counts.plot(kind='bar')
        plt.title('Distribution of Optical Illusion Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'optical_categories.png')
        
        # Depth score distribution
        depth_scores = [f['depth_score'] for f in optical['features']]
        
        plt.figure(figsize=(10, 6))
        plt.hist(depth_scores, bins=20, alpha=0.7)
        plt.title('Distribution of Depth Scores in Optical Illusions')
        plt.xlabel('Depth Score')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'optical_depth_scores.png')
        
        # Sample images
        if len(optical['images']) > 0:
            fig, axes = plt.subplots(1, min(5, len(optical['images'])), figsize=(15, 5))
            for i, ax in enumerate(axes):
                if i < len(optical['images']):
                    ax.imshow(optical['images'][i])
                    ax.set_title(f"{optical['features'][i]['category']}")
                    ax.axis('off')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / 'optical_samples.png')
        
        # Store summary statistics
        eda_results['optical'] = {
            'count': len(optical['images']),
            'categories': category_counts.to_dict(),
            'depth_score_mean': np.mean(depth_scores),
            'depth_score_std': np.std(depth_scores)
        }
    
    # Analyze 3D depth dataset
    if 'depth' in datasets:
        print("\nAnalyzing 3D depth dataset...")
        depth = datasets['depth']
        
        # Number of scenes and images
        n_scenes = len(depth['scenes'])
        n_images = sum(len(scene['images']) for scene in depth['scenes'])
        
        # Feature distributions
        perspective_distortion = [scene['features']['perspective_distortion'] for scene in depth['scenes']]
        lighting_intensity = [scene['features']['lighting_intensity'] for scene in depth['scenes']]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(perspective_distortion, bins=15, alpha=0.7)
        plt.title('Perspective Distortion')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(lighting_intensity, bins=15, alpha=0.7)
        plt.title('Lighting Intensity')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'depth_features.png')
        
        # Sample images and depth maps
        if n_scenes > 0:
            scene = depth['scenes'][0]
            if len(scene['images']) > 0:
                fig, axes = plt.subplots(2, min(3, len(scene['images'])), figsize=(15, 8))
                for i in range(min(3, len(scene['images']))):
                    axes[0, i].imshow(scene['images'][i])
                    axes[0, i].set_title(f"Image {i+1}")
                    axes[0, i].axis('off')
                    
                    axes[1, i].imshow(scene['depth_maps'][i], cmap='viridis')
                    axes[1, i].set_title(f"Depth Map {i+1}")
                    axes[1, i].axis('off')
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / 'depth_samples.png')
        
        # Store summary statistics
        eda_results['depth'] = {
            'n_scenes': n_scenes,
            'n_images': n_images,
            'perspective_distortion_mean': np.mean(perspective_distortion),
            'lighting_intensity_mean': np.mean(lighting_intensity)
        }
    
    # Analyze fashion 3D dataset
    if 'fashion' in datasets:
        print("\nAnalyzing fashion 3D dataset...")
        fashion = datasets['fashion']
        
        # Category distribution
        categories = [model['category'] for model in fashion['models']]
        category_counts = pd.Series(categories).value_counts()
        
        plt.figure(figsize=(10, 6))
        category_counts.plot(kind='bar')
        plt.title('Distribution of 3D Fashion Model Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fashion_categories.png')
        
        # Feature distributions
        complexity = [model['features']['complexity'] for model in fashion['models']]
        visual_appeal = [model['features']['visual_appeal'] for model in fashion['models']]
        realistic_score = [model['features']['realistic_score'] for model in fashion['models']]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(complexity, bins=15, alpha=0.7)
        plt.title('Complexity')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.hist(visual_appeal, bins=15, alpha=0.7)
        plt.title('Visual Appeal')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.hist(realistic_score, bins=15, alpha=0.7)
        plt.title('Realistic Score')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'fashion_features.png')
        
        # Store summary statistics
        eda_results['fashion'] = {
            'count': len(fashion['models']),
            'categories': category_counts.to_dict(),
            'complexity_mean': np.mean(complexity),
            'visual_appeal_mean': np.mean(visual_appeal),
            'realistic_score_mean': np.mean(realistic_score)
        }
    
    # Analyze color psychology dataset
    if 'color' in datasets:
        print("\nAnalyzing color psychology dataset...")
        color_df = datasets['color']
        
        # Color distribution
        plt.figure(figsize=(12, 6))
        color_counts = color_df['color'].value_counts()
        color_counts.plot(kind='bar')
        plt.title('Color Distribution')
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'color_distribution.png')
        
        # Emotion-Color heatmap
        plt.figure(figsize=(14, 8))
        emotion_color_counts = pd.crosstab(color_df['emotion'], color_df['color'])
        sns.heatmap(emotion_color_counts, annot=True, cmap='YlGnBu', fmt='d')
        plt.title('Emotion-Color Associations')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'color_emotion_heatmap.png')
        
        # Association strength by context
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='context', y='association_strength', data=color_df)
        plt.title('Association Strength by Context')
        plt.xlabel('Context')
        plt.ylabel('Association Strength')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'color_context_association.png')
        
        # Store summary statistics
        eda_results['color'] = {
            'count': len(color_df),
            'colors': color_counts.to_dict(),
            'association_strength_mean': color_df['association_strength'].mean(),
            'memory_retention_mean': color_df['memory_retention'].mean()
        }
    
    # Analyze social media dataset
    if 'social' in datasets:
        print("\nAnalyzing social media engagement dataset...")
        social_df = datasets['social']
        
        # Platform distribution
        plt.figure(figsize=(10, 6))
        platform_counts = social_df['platform'].value_counts()
        platform_counts.plot(kind='bar')
        plt.title('Distribution of Social Media Platforms')
        plt.xlabel('Platform')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'social_platforms.png')
        
        # Engagement by 3D effect
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(x='has_3d_effect', y='engagement_rate', data=social_df)
        plt.title('Engagement Rate by 3D Effect')
        plt.xlabel('Has 3D Effect')
        plt.ylabel('Engagement Rate')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='has_depth_illusion', y='engagement_rate', data=social_df)
        plt.title('Engagement Rate by Depth Illusion')
        plt.xlabel('Has Depth Illusion')
        plt.ylabel('Engagement Rate')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'social_engagement_by_effect.png')
        
        # Platform metrics comparison
        metrics = ['engagement_rate', 'likes', 'shares', 'comments', 'views']
        platform_metrics = social_df.groupby('platform')[metrics].mean().reset_index()
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            sns.barplot(x='platform', y=metric, data=platform_metrics)
            plt.title(f'Average {metric.replace("_", " ").title()} by Platform')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'social_platform_metrics.png')
        
        # Store summary statistics
        eda_results['social'] = {
            'count': len(social_df),
            'platforms': platform_counts.to_dict(),
            'with_3d_effect': social_df['has_3d_effect'].sum(),
            'engagement_with_3d': social_df[social_df['has_3d_effect']]['engagement_rate'].mean(),
            'engagement_without_3d': social_df[~social_df['has_3d_effect']]['engagement_rate'].mean()
        }
    
    # Analyze advertising dataset
    if 'advertising' in datasets:
        print("\nAnalyzing advertising performance dataset...")
        ad_df = datasets['advertising']
        
        # Campaign type distribution
        plt.figure(figsize=(10, 6))
        campaign_counts = ad_df['campaign_type'].value_counts()
        campaign_counts.plot(kind='bar')
        plt.title('Distribution of Campaign Types')
        plt.xlabel('Campaign Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ad_campaign_types.png')
        
        # Performance by depth effect
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        sns.boxplot(x='depth_effect_used', y='ctr', data=ad_df)
        plt.title('CTR by Depth Effect')
        plt.xlabel('Depth Effect Used')
        plt.ylabel('CTR')
        
        plt.subplot(1, 3, 2)
        sns.boxplot(x='depth_effect_used', y='conversion_rate', data=ad_df)
        plt.title('Conversion Rate by Depth Effect')
        plt.xlabel('Depth Effect Used')
        plt.ylabel('Conversion Rate')
        
        plt.subplot(1, 3, 3)
        sns.boxplot(x='depth_effect_used', y='roas', data=ad_df)
        plt.title('ROAS by Depth Effect')
        plt.xlabel('Depth Effect Used')
        plt.ylabel('ROAS')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ad_performance_by_depth.png')
        
        # Visual appeal correlation with performance
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='visual_appeal_score', y='conversion_rate', 
                        hue='depth_effect_used', size='budget', 
                        sizes=(20, 200), alpha=0.7, data=ad_df)
        plt.title('Conversion Rate vs Visual Appeal')
        plt.xlabel('Visual Appeal Score')
        plt.ylabel('Conversion Rate')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ad_conversion_vs_appeal.png')
        
        # Performance across industries
        metrics = ['ctr', 'conversion_rate', 'roas', 'roi']
        industry_metrics = ad_df.groupby('industry')[metrics].mean().reset_index()
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            sns.barplot(x='industry', y=metric, data=industry_metrics)
            plt.title(f'Average {metric.upper()} by Industry')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'ad_industry_performance.png')
        
        # Store summary statistics
        eda_results['advertising'] = {
            'count': len(ad_df),
            'campaign_types': campaign_counts.to_dict(),
            'with_depth_effect': ad_df['depth_effect_used'].sum(),
            'ctr_with_depth': ad_df[ad_df['depth_effect_used']]['ctr'].mean(),
            'ctr_without_depth': ad_df[~ad_df['depth_effect_used']]['ctr'].mean(),
            'avg_roas': ad_df['roas'].mean()
        }
    
    # Save EDA results to JSON
    import json
    with open(OUTPUT_DIR / 'eda_results.json', 'w') as f:
        json.dump(eda_results, f, indent=2)
    
    print(f"EDA completed. Results saved to {OUTPUT_DIR / 'eda_results.json'}")
    print(f"Figures saved to {FIGURES_DIR}")
    
    return eda_results

def train_models(datasets, args):
    """Train models based on the datasets"""
    print("\n===== Training Models =====")
    
    trained_models = {}
    
    # Train illusion predictor
    if args.train in ['all', 'illusion'] and 'optical' in datasets:
        print("\nTraining Illusion Effectiveness Predictor...")
        illusion_model = IllusionPredictor()
        illusion_metrics = illusion_model.train(datasets['optical'], test_size=0.2)
        
        # Save model
        model_path = MODELS_DIR / "illusion_predictor.joblib"
        illusion_model.save(model_path)
        
        print(f"Illusion predictor trained with accuracy: {illusion_metrics['accuracy']:.4f}")
        print(f"Model saved to {model_path}")
        
        trained_models['illusion'] = illusion_model
    
    # Train depth estimator
    if args.train in ['all', 'depth'] and 'depth' in datasets:
        print("\nTraining Depth Estimator...")
        depth_model = DepthEstimator()
        depth_metrics = depth_model.train(datasets['depth'], test_size=0.2)
        
        # Save model
        model_path = MODELS_DIR / "depth_estimator.joblib"
        depth_model.save(model_path)
        
        print(f"Depth estimator trained with MSE: {depth_metrics['mse']:.4f}, R²: {depth_metrics['r2']:.4f}")
        print(f"Model saved to {model_path}")
        
        trained_models['depth'] = depth_model
    
    # Train performance predictor
    if args.train in ['all', 'performance'] and 'advertising' in datasets:
        print("\nTraining Performance Predictor...")
        performance_model = PerformancePredictor()
        performance_metrics = performance_model.train(datasets['advertising'], test_size=0.2)
        
        # Save model
        model_path = MODELS_DIR / "performance_predictor"
        performance_model.save(model_path)
        
        print("Performance predictor trained with metrics:")
        for target, metric in performance_metrics.items():
            print(f"  {target}: MSE={metric['mse']:.4f}, R²={metric['r2']:.4f}")
        print(f"Model saved to {model_path}")
        
        trained_models['performance'] = performance_model
    
    return trained_models

def generate_anamorphic_results(datasets, trained_models, args):
    """Generate 3D anamorphic results from 2D images"""
    print("\n===== Generating 3D Anamorphic Results =====")
    
    # Check if we have the necessary models
    if 'illusion' not in trained_models or 'depth' not in trained_models:
        print("Cannot generate anamorphic results without illusion and depth models.")
        return
    
    illusion_model = trained_models['illusion']
    depth_model = trained_models['depth']
    
    # Get images to process
    images = []
    if 'optical' in datasets and len(datasets['optical']['images']) > 0:
        images.extend(datasets['optical']['images'][:args.generate_samples])
    
    # If we don't have enough images, create some
    while len(images) < args.generate_samples:
        # Create a simple gradient image
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Add random shapes with gradients
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, 256), random.randint(0, 256)
            x2, y2 = random.randint(0, 256), random.randint(0, 256)
            
            # Create gradient colors
            r1, g1, b1 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            r2, g2, b2 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            
            # Draw gradient rectangle
            for i in range(min(x1, x2), max(x1, x2)):
                for j in range(min(y1, y2), max(y1, y2)):
                    # Interpolate color
                    r = int(r1 + (r2 - r1) * (i - min(x1, x2)) / max(1, max(x1, x2) - min(x1, x2)))
                    g = int(g1 + (g2 - g1) * (j - min(y1, y2)) / max(1, max(y1, y2) - min(y1, y2)))
                    b = int(b1 + (b2 - b1) * ((i + j) / 2) / 256)
                    
                    # Draw pixel
                    if 0 <= i < 256 and 0 <= j < 256:
                        img.putpixel((i, j), (r, g, b))
        
        images.append(img)
    
    # Process each image
    for i, image in enumerate(images[:args.generate_samples]):
        print(f"\nProcessing image {i+1}/{args.generate_samples}")
        
        # Predict illusion effectiveness
        effectiveness = illusion_model.predict(image)
        print(f"Predicted effectiveness: {'Effective' if effectiveness == 1 else 'Not effective'}")
        
        # Generate depth map
        depth_map = depth_model.predict(image)
        depth_img = Image.fromarray(depth_map)
        
        # Create anamorphic 3D representation
        # In a real implementation, this would use the depth map to create a 3D model
        # For now, we'll just create a simple visualization
        
        # Combine original and depth map
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(depth_img, cmap='viridis')
        axes[1].set_title("Depth Map")
        axes[1].axis('off')
        
        # Create a simple "3D" visualization (just for demonstration)
        # In a real implementation, this would be a proper 3D render
        axes[2].imshow(image)
        depth_contour = axes[2].contour(depth_map, cmap='Reds', alpha=0.6, levels=10)
        axes[2].set_title("Pseudo-3D Visualization")
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = ANAMORPHIC_DIR / f"anamorphic_result_{i+1}.png"
        plt.savefig(output_path)
        print(f"Saved anamorphic result to {output_path}")
        
        # In a real implementation, we would also save the 3D model file
        # output_path_3d = ANAMORPHIC_DIR / f"anamorphic_model_{i+1}.obj"
        # print(f"Saved 3D model to {output_path_3d}")
    
    print(f"\nGenerated {args.generate_samples} anamorphic results in {ANAMORPHIC_DIR}")

def main():
    """Main function to run the full pipeline"""
    parser = argparse.ArgumentParser(description="Run the full Colorful Canvas AI Art Studio pipeline")
    
    parser.add_argument("--data_source", type=str, default="all",
                        choices=["all", "optical", "depth", "fashion", "color", "social", "advertising"],
                        help="Which data sources to fetch")
    
    parser.add_argument("--train", type=str, default="all",
                        choices=["all", "illusion", "depth", "performance", "none"],
                        help="Which models to train")
    
    parser.add_argument("--generate_samples", type=int, default=5,
                        help="Number of anamorphic samples to generate")
    
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to fetch from each data source")
    
    parser.add_argument("--no_cache", action="store_true",
                        help="Don't use cached data")
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Start timer
    start_time = time.time()
    
    # Create data loader
    loader = DataLoader(use_cache=not args.no_cache)
    
    # Fetch datasets
    datasets = fetch_datasets(loader, args)
    
    # Perform EDA
    eda_results = perform_eda(datasets)
    
    # Train models (if requested)
    trained_models = {}
    if args.train != "none":
        trained_models = train_models(datasets, args)
    
    # Generate anamorphic results (if requested)
    if args.generate_samples > 0 and len(trained_models) > 0:
        generate_anamorphic_results(datasets, trained_models, args)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\n===== Pipeline completed in {elapsed_time:.2f} seconds =====")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import random
    main() 