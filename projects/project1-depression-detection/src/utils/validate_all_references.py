#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate All References
This script runs URL validation on the white paper and optionally updates references.
"""

import os
import argparse
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local modules
try:
    from utils.url_validator import validate_references
    from utils.update_references import update_references, VERIFIED_REFERENCES
except ImportError:
    print("Error importing local modules. Make sure you're running from the project root.")
    sys.exit(1)

def generate_report(validation_results: pd.DataFrame, 
                    output_path: str = "reference_validation_report.md") -> None:
    """
    Generate a detailed markdown report from validation results.
    
    Args:
        validation_results: DataFrame with validation results
        output_path: Path to save the report
    """
    valid_count = validation_results["Valid"].sum()
    total_count = len(validation_results)
    
    report = f"""# Reference Validation Report
    
## Summary
- **Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total References:** {total_count}
- **Valid URLs:** {valid_count} ({valid_count/total_count*100:.1f}%)
- **Invalid URLs:** {total_count - valid_count} ({(total_count-valid_count)/total_count*100:.1f}%)

## Details

"""
    
    # Add valid URLs
    if valid_count > 0:
        report += "### Valid References\n\n"
        for _, row in validation_results[validation_results["Valid"]].iterrows():
            report += f"1. [{row['URL']}]({row['URL']})\n   - {row['Reference']}\n\n"
    
    # Add invalid URLs
    if valid_count < total_count:
        report += "### Invalid References\n\n"
        for _, row in validation_results[~validation_results["Valid"]].iterrows():
            report += f"1. {row['URL']} - Error: {row['Error']}\n   - {row['Reference']}\n\n"
    
    # Add section for potential fixes if update_references module is available
    report += """## Recommended Actions

1. Check each invalid URL manually to determine if:
   - The URL has a typo
   - The reference has been moved to a new location
   - The reference is no longer available online

2. Consider using the `update_references.py` script to update references with verified alternatives.

"""
    
    # Write the report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {output_path}")

def main():
    """Main function to validate references and generate a report."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Validate and update references in markdown files")
    parser.add_argument("--file", default="../docs/white_paper.md", 
                         help="Path to the markdown file (default: ../docs/white_paper.md)")
    parser.add_argument("--timeout", type=int, default=10, 
                         help="Timeout for URL validation in seconds (default: 10)")
    parser.add_argument("--update", action="store_true", 
                         help="Update invalid references with verified ones")
    parser.add_argument("--report", default=None,
                         help="Path to save the validation report (default: white_paper_validation_report.md)")
    
    args = parser.parse_args()
    
    # Normalize and validate file path
    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Set default report path if not specified
    if args.report is None:
        base_name = os.path.basename(file_path).split('.')[0]
        args.report = f"{base_name}_validation_report.md"
    
    # Validate references
    print(f"Validating references in: {file_path}")
    results_df = validate_references(file_path, args.timeout)
    
    if results_df.empty:
        print("No references found to validate.")
        sys.exit(0)
    
    # Generate validation report
    generate_report(results_df, args.report)
    
    # Update references if requested
    if args.update:
        print("\nUpdating references...")
        success = update_references(file_path)
        if success:
            print("References updated successfully.")
            
            # Re-validate after update
            print("\nRe-validating references after update...")
            updated_results = validate_references(file_path, args.timeout)
            
            # Generate updated report
            updated_report = f"updated_{args.report}"
            generate_report(updated_results, updated_report)
            print(f"Updated validation report: {updated_report}")
        else:
            print("Failed to update references.")
    
    # Print final message
    print("\nValidation process complete.")

if __name__ == "__main__":
    main() 