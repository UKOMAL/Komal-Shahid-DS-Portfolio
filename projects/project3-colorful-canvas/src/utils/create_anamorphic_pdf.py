#!/usr/bin/env python3
"""
Script to create anamorphic billboard PDF documentation for ColorfulCanvas project
"""
import os
import sys
import shutil
from pathlib import Path

def copy_script_to_project():
    """Copy the anamorphic billboard script to the project directory"""
    # Source file path
    source_file = os.path.join(os.path.expanduser("~"), "Downloads", "working_anamorphic_billboard-4.py")
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Destination paths
    code_dir = os.path.join(project_root, "src", "core")
    os.makedirs(code_dir, exist_ok=True)
    
    dest_file = os.path.join(code_dir, "anamorphic_billboard.py")
    
    # Copy the file
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_file)
        print(f"Copied script to: {dest_file}")
        return dest_file
    else:
        print(f"Error: Source file not found at {source_file}")
        return None

def generate_html_files(script_path):
    """Generate HTML files for the script"""
    if not script_path or not os.path.exists(script_path):
        print("Error: Invalid script path")
        return
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the HTML generator
    html_generator = os.path.join(script_dir, "generate_code_html.py")
    
    if not os.path.exists(html_generator):
        print(f"Error: HTML generator script not found at {html_generator}")
        return
    
    # Execute the generator script using python3 instead of python
    os.system(f"python3 {html_generator} {script_path}")
    
    print("\nHTML files generated for PDF creation")
    print("\nTo create PDFs:")
    print("1. Open the generated HTML files in Chrome or any modern browser")
    print("2. Use Print (Cmd+P or Ctrl+P) and select 'Save as PDF'")
    print("3. Set margins to 'None' for best results")
    print("4. Save the PDF to the project's docs/final directory")

if __name__ == "__main__":
    print("Creating anamorphic billboard documentation for ColorfulCanvas project...")
    
    # 1. Copy script to project
    script_path = copy_script_to_project()
    
    # 2. Generate HTML files for PDF creation
    if script_path:
        generate_html_files(script_path)
    
    print("\nProcess completed. Check the docs/final directory for output files.") 