#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations Runner

This script serves as a central entry point to generate all visualizations
for the federated healthcare AI project.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to run all visualizations."""
    print("Federated Healthcare AI - Visualization Generator")
    print("=" * 60)
    
    # Get absolute paths
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output" / "visualizations"
    scripts_dir = output_dir
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Visualizations will be saved to: {output_dir}")
    print("-" * 60)
    
    # List of visualization scripts
    visualization_scripts = [
        "model_convergence.py",
        "privacy_budget_tradeoff.py",
        "institution_performance.py",
        "accuracy_by_modality.py",
        "communication_efficiency.py",
        "privacy_attack_success.py",
        "client_contribution.py",
        # Diverse visualization types
        "privacy_radar.py",
        "performance_heatmap.py",
        "network_visualization.py",
        # Advanced visualizations
        "model_complexity_3d.py",
        "convergence_analysis.py"
    ]
    
    # Run each visualization script
    success_count = 0
    for script in visualization_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"Running {script}...")
            try:
                result = subprocess.run([sys.executable, str(script_path)], 
                                       check=True, 
                                       capture_output=True,
                                       text=True)
                print(f"✅ Successfully generated visualization from {script}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"❌ Error running {script}: {e}")
                print(f"Error output: {e.stderr}")
        else:
            print(f"⚠️ Script {script} not found at {script_path}")
        
        print("-" * 60)
    
    # Copy visualization files to docs directory for the presentation
    docs_images_dir = project_root / "docs" / "images"
    os.makedirs(docs_images_dir, exist_ok=True)
    
    print(f"Copying visualization images to {docs_images_dir}")
    
    # Find all image files in the visualizations directory (PNG and GIF)
    image_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.gif"))
    
    if image_files:
        import shutil
        for img_file in image_files:
            target_path = docs_images_dir / img_file.name
            # Use shutil.copy2 to preserve metadata
            shutil.copy2(img_file, target_path)
            print(f"✅ Copied {img_file.name} to {target_path}")
            
        print(f"\nSuccessfully copied {len(image_files)} visualization images to {docs_images_dir}")
    else:
        print("No visualization images found to copy")
    
    print("\nSummary:")
    print(f"- Total visualization scripts: {len(visualization_scripts)}")
    print(f"- Successfully generated: {success_count}")
    print(f"- Failed: {len(visualization_scripts) - success_count}")
    
    print("\nVisualization process complete!")

if __name__ == "__main__":
    main() 