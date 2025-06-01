#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorful Canvas: AI Art Studio - Main Entry Point
A comprehensive toolkit for creating stunning 3D visual illusions from 2D images

This script serves as the main entry point for the application, providing a unified
interface to all the functionality in the project.

Usage:
    python run.py --mode [data|train|generate|demo] [options]
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Colorful Canvas: AI Art Studio - Create 3D visual illusions from 2D images"
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True,
        choices=["data", "train", "generate", "demo"],
        help="Operation mode: data (fetch/prepare data), train (train models), generate (create illusions), demo (run examples)"
    )
    
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to input image (for generate mode)"
    )
    
    parser.add_argument(
        "--effect", 
        type=str, 
        choices=["shadow_box", "screen_pop", "anamorphic"],
        help="Type of 3D effect to generate"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--strength", 
        type=float, 
        default=1.5,
        help="Effect strength (0.5-3.0)"
    )
    
    return parser.parse_args()

def main():
    """Main function that dispatches to the appropriate module based on the mode."""
    args = parse_arguments()
    
    if args.mode == "data":
        # Import data modules only when needed
        from src.data.get_data import fetch_github_data
        from src.data.data_loader import preprocess_images
        
        # Fetch and prepare data
        print("Fetching data from GitHub...")
        fetch_github_data()
        print("Data fetched successfully")
        
    elif args.mode == "train":
        # Import training module only when needed
        from src.models.train_models import train_model
        
        # Train models
        print("Training models...")
        train_model()
        print("Models trained successfully")
        
    elif args.mode == "generate":
        # Import generation modules only when needed
        from src.colorful_canvas_ai import (
            generate_depth_map,
            create_shadow_box_effect,
            create_screen_pop_effect,
            create_anamorphic_billboard,
            load_image,
            save_image
        )
        
        # Validate arguments
        if not args.image:
            print("Error: --image argument is required for generate mode")
            sys.exit(1)
            
        if not args.effect:
            print("Error: --effect argument is required for generate mode")
            sys.exit(1)
            
        # Set output directory
        output_dir = args.output if args.output else "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
        if image is None:
            sys.exit(1)
            
        # Generate depth map
        depth_map = generate_depth_map(args.image)
        
        # Apply selected effect
        if args.effect == "shadow_box":
            result = create_shadow_box_effect(image, depth_map, light_angle=45)
            output_path = os.path.join(output_dir, "shadow_box_effect.png")
        elif args.effect == "screen_pop":
            result = create_screen_pop_effect(image, depth_map, strength=args.strength)
            output_path = os.path.join(output_dir, "screen_pop_effect.png")
        elif args.effect == "anamorphic":
            result = create_anamorphic_billboard(image, depth_map, strength=args.strength)
            output_path = os.path.join(output_dir, "anamorphic_effect.png")
            
        # Save result
        save_image(result, output_path)
        print(f"Effect generated successfully: {output_path}")
        
    elif args.mode == "demo":
        # Import pipeline module only when needed
        from src.run_pipeline import run_demo
        
        # Run demo
        print("Running demo...")
        run_demo()
        print("Demo completed successfully")
        
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

if __name__ == "__main__":
    main() 