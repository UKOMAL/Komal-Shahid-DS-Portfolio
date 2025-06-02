#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorful Canvas Demo Script
A command-line interactive demo for the Colorful Canvas AI Art Studio.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Project: Colorful Canvas AI Art Studio
"""

import os
import sys
import argparse
from pathlib import Path
import time
from PIL import Image

# Add the parent directory to sys.path to import the project modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the ColorfulCanvasAI class
from src.colorful_canvas import ColorfulCanvasAI

def print_header():
    """Print a colorful header for the demo"""
    print("\n" + "=" * 80)
    print("🎨  COLORFUL CANVAS AI ART STUDIO - INTERACTIVE DEMO  🎨")
    print("=" * 80)
    print("Create stunning anamorphic illusions with AI-powered image processing")
    print("=" * 80 + "\n")

def list_sample_images():
    """List available sample images"""
    samples_dir = Path(parent_dir) / "data" / "samples"
    if not samples_dir.exists():
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"⚠️  No sample images found in {samples_dir}")
        print("   Please add some sample images to this directory.")
        return []
    
    sample_files = [f for f in samples_dir.glob("*.jpg") or samples_dir.glob("*.png")]
    
    if not sample_files:
        print(f"⚠️  No sample images found in {samples_dir}")
        print("   Please add some .jpg or .png images to this directory.")
        return []
    
    print("📸  Available sample images:")
    for i, file in enumerate(sample_files, 1):
        print(f"  {i}. {file.name}")
    
    return sample_files

def run_interactive_demo():
    """Run the interactive demo"""
    print_header()
    
    # Initialize the AI
    ai = ColorfulCanvasAI()
    print("✅  AI system initialized\n")
    
    # List sample images
    sample_files = list_sample_images()
    
    if not sample_files:
        return
    
    # Select image
    while True:
        try:
            choice = input("\n🖼️  Select an image number (or 'q' to quit): ")
            
            if choice.lower() == 'q':
                print("\n👋  Thanks for trying Colorful Canvas AI!")
                return
            
            choice = int(choice)
            if 1 <= choice <= len(sample_files):
                selected_file = sample_files[choice - 1]
                break
            else:
                print(f"⚠️  Please enter a number between 1 and {len(sample_files)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
    
    print(f"\n🔍  Selected image: {selected_file.name}")
    
    # Select effect
    effects = {
        "1": {"name": "shadow_box", "display": "Shadow Box Illusion"},
        "2": {"name": "screen_pop", "display": "Screen Pop-out Effect"},
        "3": {"name": "seoul_corner", "display": "Seoul Corner Projection"}
    }
    
    print("\n🎭  Available effects:")
    for key, effect in effects.items():
        print(f"  {key}. {effect['display']}")
    
    while True:
        effect_choice = input("\n✨  Select an effect number: ")
        if effect_choice in effects:
            selected_effect = effects[effect_choice]["name"]
            display_effect = effects[effect_choice]["display"]
            break
        else:
            print(f"⚠️  Please enter a number between 1 and {len(effects)}")
    
    print(f"\n🔮  Selected effect: {display_effect}")
    
    # Process the image
    print("\n⏳  Processing image...")
    
    # Load the image
    input_image = ai.load_image(selected_file)
    
    # Generate depth map
    print("🏔️  Generating depth map...")
    depth_map = ai.generate_depth_map(input_image)
    
    # Apply selected effect
    print(f"✨  Applying {display_effect}...")
    if selected_effect == "shadow_box":
        result_image = ai.create_shadow_box_effect(input_image, depth_map)
    elif selected_effect == "screen_pop":
        result_image = ai.create_screen_pop_effect(input_image, depth_map)
    elif selected_effect == "seoul_corner":
        result_image = ai.create_seoul_corner_projection(input_image, depth_map)
    
    # Save output
    output_dir = Path(parent_dir) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    output_file = output_dir / f"{selected_effect}_{timestamp}.png"
    
    ai.save_image(result_image, output_file)
    
    print(f"\n✅  Processing complete!")
    print(f"📁  Output saved to: {output_file}")
    print("\n👁️  View the image from the correct angle to see the 3D illusion effect!")
    
    # Viewing instructions
    if selected_effect == "shadow_box":
        print("\n👀  Viewing instructions: Look directly at the image from the front.")
    elif selected_effect == "screen_pop":
        print("\n👀  Viewing instructions: View from a 45-degree angle for best effect.")
    elif selected_effect == "seoul_corner":
        print("\n👀  Viewing instructions: Position image in a corner and view from 30-degrees.")
    
    print("\n👋  Thanks for trying Colorful Canvas AI!")

def main():
    """Main function for the demo script"""
    parser = argparse.ArgumentParser(description="Colorful Canvas AI Art Studio Demo")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    
    args = parser.parse_args()
    
    if args.interactive or len(sys.argv) == 1:
        run_interactive_demo()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 