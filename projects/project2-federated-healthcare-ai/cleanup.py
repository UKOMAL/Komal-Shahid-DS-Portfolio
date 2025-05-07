#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository Cleanup Script

This script cleans up unnecessary files that shouldn't be included in the final GitHub repository.
It removes test files, analysis scripts, and other miscellaneous files not needed for the portfolio.
"""

import os
import shutil
from pathlib import Path

# Files and directories to keep in the final repository
ESSENTIAL_FILES = [
    # Core project files
    "README.md",
    "PORTFOLIO.md",
    "INSTALLATION.md",
    "requirements.txt",
    ".gitignore",
    "LICENSE",
    
    # Core directories
    "src/",
    "docs/",
    "models/",
    "data/",
    "config/",
    "output/visualizations/",
]

# Files to specifically delete (relative to the project root)
FILES_TO_DELETE = [
    # Temporary files
    "*.tmp",
    "*.temp",
    "*.log",
    "*.pyc",
    "__pycache__",
    
    # Analysis and test files
    "src/run_new_visualizations.py",
    "uploadAssignment",
    "uploadAssignment_milestone3",
    "dsc-680-project-milestone-3-rubric-v2.pdf",
    
    # Unused output directories
    "output/test_results/",
    "output/visualizations/unused/",
]

def is_essential(path, project_root):
    """Check if a path is essential and should be kept."""
    rel_path = os.path.relpath(path, project_root)
    
    # Check if path is directly in the list of essential files
    if rel_path in ESSENTIAL_FILES:
        return True
    
    # Check if path is a subdirectory of an essential directory
    for essential in ESSENTIAL_FILES:
        if essential.endswith("/") and rel_path.startswith(essential):
            return True
    
    return False

def should_delete(path, project_root):
    """Check if a path should be deleted."""
    rel_path = os.path.relpath(path, project_root)
    
    # Check exact matches
    if rel_path in FILES_TO_DELETE:
        return True
    
    # Check for patterns with wildcards
    for pattern in FILES_TO_DELETE:
        if pattern.startswith("*") and rel_path.endswith(pattern[1:]):
            return True
    
    # Check for directories specified with wildcards
    for pattern in FILES_TO_DELETE:
        if pattern.endswith("/") and rel_path.startswith(pattern):
            return True
    
    return False

def main():
    """Main function to clean up the repository."""
    print("Repository Cleanup Starting...")
    
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir  # Assuming this script is in the root directory
    
    # Create a directory for files to be moved (instead of deleted)
    backup_dir = os.path.join(project_root, "_cleanup_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Walk through all files and directories
    for root, dirs, files in os.walk(project_root, topdown=False):
        # Skip the backup directory itself
        if root.startswith(backup_dir):
            continue
        
        # Process files
        for file in files:
            file_path = os.path.join(root, file)
            
            # Skip this script itself
            if file_path == os.path.abspath(__file__):
                continue
                
            if should_delete(file_path, project_root) and not is_essential(file_path, project_root):
                print(f"Moving to backup: {os.path.relpath(file_path, project_root)}")
                rel_path = os.path.relpath(file_path, project_root)
                backup_path = os.path.join(backup_dir, rel_path)
                
                # Create directory structure in backup
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Move the file instead of deleting it
                shutil.move(file_path, backup_path)
        
        # Process directories (empty ones only)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            
            # Skip the backup directory
            if dir_path.startswith(backup_dir):
                continue
                
            # Remove empty directories
            if not os.listdir(dir_path):
                print(f"Removing empty directory: {os.path.relpath(dir_path, project_root)}")
                os.rmdir(dir_path)
    
    print("\nRepository cleanup complete!")
    print(f"Backup of removed files saved to: {backup_dir}")
    print("Review the backup directory to ensure no important files were moved.")
    print("If everything looks good, you can delete the backup directory.")

if __name__ == "__main__":
    main() 