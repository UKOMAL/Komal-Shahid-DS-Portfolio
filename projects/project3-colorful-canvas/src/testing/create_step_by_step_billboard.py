#!/usr/bin/env python3
"""
Step-by-Step Billboard Generator with PNG Saves
Saves intermediate results at each step to watch the progression
"""

import subprocess
import sys
import os
from pathlib import Path

def run_blender_step_by_step(image_path, output_dir="output/step_by_step"):
    """
    Run the billboard generation with step-by-step PNG saves
    """
    print("🎬 STEP-BY-STEP BILLBOARD GENERATION")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the steps we want to capture
    steps = [
        (1, "Environment Setup"),
        (2, "3D Content Creation"), 
        (3, "Camera Positioning"),
        (4, "Billboard Geometry"),
        (5, "UV Projection"),
        (6, "Texture Baking"),
        (7, "Vibrant Texture Application"),
        (8, "Dimensional Lighting"),
        (9, "Final Render")
    ]
    
    print(f"📷 Input: {image_path}")
    print(f"💾 Step outputs: {output_dir}")
    print(f"🎯 Total steps: {len(steps)}")
    print()
    
    for step_num, step_name in steps:
        print(f"🚀 EXECUTING STEP {step_num}: {step_name}")
        print("-" * 40)
        
        # Set output path for this step
        step_output = f"{output_dir}/step_{step_num:02d}_{step_name.replace(' ', '_').lower()}.png"
        
        # Run the billboard generator for this step only
        cmd = [
            "./run_anamorphic_with_deps.sh",
            "--image", image_path,
            "--output", step_output
        ]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Check if file was created
            if os.path.exists(step_output):
                file_size = os.path.getsize(step_output) / 1024 / 1024  # MB
                print(f"✅ Step {step_num} completed: {step_output} ({file_size:.1f}MB)")
            else:
                print(f"⚠️  Step {step_num} completed but no output file found")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Step {step_num} failed: {e}")
            print("STDOUT:", e.stdout[-500:] if e.stdout else "None")
            print("STDERR:", e.stderr[-500:] if e.stderr else "None")
            break
        except Exception as e:
            print(f"❌ Unexpected error in step {step_num}: {e}")
            break
        
        print()
    
    # Create summary
    print("📊 STEP-BY-STEP SUMMARY")
    print("=" * 60)
    created_files = []
    for step_num, step_name in steps:
        step_output = f"{output_dir}/step_{step_num:02d}_{step_name.replace(' ', '_').lower()}.png"
        if os.path.exists(step_output):
            file_size = os.path.getsize(step_output) / 1024 / 1024
            created_files.append((step_num, step_name, step_output, file_size))
            print(f"✅ Step {step_num:2d}: {step_name:<25} -> {file_size:6.1f}MB")
        else:
            print(f"❌ Step {step_num:2d}: {step_name:<25} -> MISSING")
    
    print(f"\n🎉 Generated {len(created_files)}/{len(steps)} step images")
    return created_files

def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python3 create_step_by_step_billboard.py <image_path>")
        print("Available images:")
        for img in Path("data/input").glob("*.jpg"):
            print(f"  - {img}")
        return
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Extract image name for output directory
    image_name = Path(image_path).stem
    output_dir = f"output/step_by_step_{image_name}"
    
    # Run step-by-step generation
    created_files = run_blender_step_by_step(image_path, output_dir)
    
    if created_files:
        print(f"\n📁 All step images saved in: {output_dir}")
        print("🎬 You can now watch the progression by viewing each step PNG!")

if __name__ == "__main__":
    main() 