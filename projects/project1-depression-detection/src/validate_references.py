#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Validation Script
Validates all URLs in the white paper and generates a report.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to make imports work when script is run directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import the utility functions
from src.utils.reference_manager import validate_and_update_references, generate_markdown_report

def main():
    """Validate references in the white paper."""
    # Define paths
    white_paper_path = os.path.join(
        project_root,
        "docs",
        "white_paper.md"
    )
    output_dir = os.path.join(
        project_root,
        "output",
        "reference_validation"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate references
    print(f"Validating references in {white_paper_path}")
    print(f"Results will be saved to {output_dir}")
    
    validation_results, csv_report_path = validate_and_update_references(
        file_path=white_paper_path,
        output_dir=output_dir,
        update_file=False,
        timeout=15,
        max_workers=5
    )
    
    # Generate markdown report
    if csv_report_path:
        md_report_path = generate_markdown_report(validation_results, csv_report_path)
        print(f"Markdown report saved to {md_report_path}")
    
    # Print summary
    valid_count = sum(1 for r in validation_results if r.get('valid', False))
    invalid_count = len(validation_results) - valid_count
    
    print("\nValidation Summary:")
    print(f"- Total References: {len(validation_results)}")
    print(f"- Valid References: {valid_count} ({valid_count/len(validation_results)*100:.1f}% if available)")
    print(f"- Invalid References: {invalid_count} ({invalid_count/len(validation_results)*100:.1f}% if available)")
    
    if invalid_count > 0:
        print("\nInvalid References:")
        for i, ref in enumerate([r for r in validation_results if not r.get('valid', False)]):
            print(f"{i+1}. {ref.get('URL', '')}")
            print(f"   Status: {ref.get('status', 'Unknown')}")
            print(f"   Error: {ref.get('error', '')}")
            print()

if __name__ == "__main__":
    main() 