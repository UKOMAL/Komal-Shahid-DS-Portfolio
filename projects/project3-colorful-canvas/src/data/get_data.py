#!/usr/bin/env python3
"""
This script downloads datasets for the Colorful Canvas AI Art Studio
and demonstrates how to use them with our models.

Usage:
    python get_data.py --dataset [optical_illusions|depth_perception|industry_performance|all]
"""

import os
import argparse
import matplotlib.pyplot as plt
from data_loader import DataLoader

def show_optical_illusions_samples(dataset, n_samples=3):
    """Show samples from optical illusions dataset"""
    if len(dataset["images"]) < n_samples:
        n_samples = len(dataset["images"])
    
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 5, 5))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        img = dataset["images"][i]
        feature = dataset["features"][i]
        
        axes[i].imshow(img)
        axes[i].set_title(f"Category: {feature['category']}\nDepth Score: {feature['depth_score']:.1f}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def show_depth_perception_samples(dataset, n_samples=2):
    """Show samples from depth perception dataset"""
    if len(dataset["scenes"]) < n_samples:
        n_samples = len(dataset["scenes"])
    
    for i in range(n_samples):
        scene = dataset["scenes"][i]
        
        n_images = min(2, len(scene["images"]))
        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 5, 10))
        
        for j in range(n_images):
            # Display image
            axes[0, j].imshow(scene["images"][j])
            axes[0, j].set_title(f"Image {j+1}")
            axes[0, j].axis("off")
            
            # Display depth map
            axes[1, j].imshow(scene["depth_maps"][j], cmap="viridis")
            axes[1, j].set_title(f"Depth Map {j+1}")
            axes[1, j].axis("off")
        
        plt.suptitle(f"Scene: {scene['name']}")
        plt.tight_layout()
        plt.show()

def show_industry_performance_samples(dataset, n_samples=5):
    """Show samples from industry performance dataset"""
    if len(dataset) < n_samples:
        n_samples = len(dataset)
    
    # Show dataframe
    print(dataset.head(n_samples))
    
    # Plot engagement rate vs depth perception score
    plt.figure(figsize=(10, 6))
    
    # Color by depth effect used
    colors = dataset["depth_effect_used"].map({True: "green", False: "red"})
    
    plt.scatter(
        dataset["depth_perception_score"],
        dataset["engagement_rate"],
        c=colors,
        alpha=0.6
    )
    
    plt.xlabel("Depth Perception Score")
    plt.ylabel("Engagement Rate")
    plt.title("Engagement Rate vs Depth Perception Score")
    plt.grid(True, alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlGn"), 
                 label="Depth Effect Used", 
                 ticks=[0, 1])
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Download and test datasets")
    
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["optical_illusions", "depth_perception", "industry_performance", "all"],
                        help="Which dataset to download and test")
    
    parser.add_argument("--max_samples", type=int, default=10,
                        help="Maximum number of samples to download")
    
    parser.add_argument("--no_cache", action="store_true",
                        help="Don't use cached data")
    
    parser.add_argument("--no_visualize", action="store_true",
                        help="Don't visualize the data")
    
    args = parser.parse_args()
    
    # Create data loader
    loader = DataLoader(use_cache=not args.no_cache)
    
    # Load and show datasets
    if args.dataset == "optical_illusions" or args.dataset == "all":
        print("\n=== Loading Optical Illusions Dataset ===")
        optical_illusions = loader.load_optical_illusions(max_samples=args.max_samples)
        print(f"Loaded {len(optical_illusions['images'])} optical illusion images")
        
        if not args.no_visualize:
            show_optical_illusions_samples(optical_illusions)
    
    if args.dataset == "depth_perception" or args.dataset == "all":
        print("\n=== Loading 3D Depth Perception Dataset ===")
        depth_perception = loader.load_depth_perception(max_samples=args.max_samples)
        print(f"Loaded {len(depth_perception['scenes'])} 3D depth perception scenes")
        
        if not args.no_visualize:
            show_depth_perception_samples(depth_perception)
    
    if args.dataset == "industry_performance" or args.dataset == "all":
        print("\n=== Loading Industry Performance Dataset ===")
        industry_performance = loader.load_industry_performance(max_samples=args.max_samples)
        print(f"Loaded {len(industry_performance)} industry performance records")
        
        if not args.no_visualize:
            show_industry_performance_samples(industry_performance)
    
    print("\nDataset loading complete!")

if __name__ == "__main__":
    main() 