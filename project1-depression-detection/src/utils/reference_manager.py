#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Manager Module
Functions for validating and updating references in markdown files.
"""

import os
import re
import csv
import json
import shutil
import datetime
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Import from the same package using relative imports
from .url_validator import extract_urls_from_markdown, validate_urls

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reference_manager')

def validate_and_update_references(
    file_path: str, 
    output_dir: Optional[str] = None, 
    update_file: bool = False,
    timeout: int = 10,
    max_workers: int = 5
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Validate references in a markdown file and generate a report.
    
    Args:
        file_path: Path to the markdown file
        output_dir: Directory to save the validation report
        update_file: Whether to update broken links in the file
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent workers
    
    Returns:
        Tuple containing the validation results and path to the report
    """
    # Extract references
    logger.info(f"Extracting references from {file_path}")
    references = extract_urls_from_markdown(file_path)
    logger.info(f"Found {len(references)} references")
    
    if not references:
        logger.warning("No references found in the file")
        return [], ""
    
    # Validate references
    logger.info(f"Validating {len(references)} references")
    validation_results = validate_urls(references, timeout=timeout, max_workers=max_workers)
    
    # Generate report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"reference_validation_{Path(file_path).stem}_{timestamp}.csv"
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, report_filename)
    else:
        report_path = os.path.join(os.path.dirname(file_path), report_filename)
    
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['URL', 'Title', 'Type', 'Valid', 'Status', 'Response Time (s)', 'Error'])
        
        for result in validation_results:
            writer.writerow([
                result.get('URL', ''),
                result.get('Title', ''),
                result.get('Type', ''),
                result.get('valid', False),
                result.get('status', 'Unknown'),
                result.get('response_time', 0),
                result.get('error', '')
            ])
    
    logger.info(f"Validation report saved to {report_path}")
    
    # Count valid and invalid references
    valid_count = sum(1 for r in validation_results if r.get('valid', False))
    invalid_count = len(validation_results) - valid_count
    
    logger.info(f"Validation summary: {valid_count} valid, {invalid_count} invalid references")
    
    # If requested, update the file with fixes for broken links
    if update_file and invalid_count > 0:
        updated_file_path = update_references_in_file(file_path, validation_results)
        logger.info(f"Updated file saved to {updated_file_path}")
    
    return validation_results, report_path

def update_references_in_file(file_path: str, validation_results: List[Dict[str, Any]]) -> str:
    """
    Update references in a markdown file based on validation results.
    
    Args:
        file_path: Path to the markdown file
        validation_results: List of validation results
    
    Returns:
        Path to the updated file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a backup of the original file
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup of original file at {backup_path}")
    
    # Only process invalid references
    invalid_refs = [r for r in validation_results if not r.get('valid', False)]
    
    if not invalid_refs:
        logger.info("No invalid references to update")
        return file_path
    
    updated_content = content
    updated_count = 0
    
    # Process each invalid reference
    for ref in invalid_refs:
        url = ref.get('URL', '')
        if not url:
            continue
        
        # Try to find an alternative URL
        alternative_url = find_alternative_url(url)
        if alternative_url:
            # Replace the URL in the content
            escaped_url = re.escape(url)
            pattern = f'({escaped_url})'
            updated_content = re.sub(pattern, alternative_url, updated_content)
            updated_count += 1
            logger.info(f"Replaced {url} with {alternative_url}")
    
    if updated_count > 0:
        # Save the updated content to a new file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = Path(file_path).stem
        file_ext = Path(file_path).suffix
        updated_file_path = f"{os.path.dirname(file_path)}/{file_name}_updated_{timestamp}{file_ext}"
        
        with open(updated_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        logger.info(f"Updated {updated_count} references and saved to {updated_file_path}")
        return updated_file_path
    else:
        logger.info("No references were updated")
        return file_path

def find_alternative_url(url: str) -> Optional[str]:
    """
    Find an alternative URL for a broken link.
    
    This function implements various strategies to find working alternatives:
    1. Check common URL variations
    2. Check web archives
    3. Search for the same resource on different domains
    
    Args:
        url: The original broken URL
        
    Returns:
        An alternative URL if found, None otherwise
    """
    # Strategy 1: Try HTTPS if HTTP
    if url.startswith('http://'):
        https_url = 'https://' + url[7:]
        try:
            import requests
            response = requests.head(https_url, timeout=5, allow_redirects=True)
            if response.status_code < 400:
                return https_url
        except Exception:
            pass
    
    # Strategy 2: Check for DOI alternatives
    if 'doi.org' in url:
        # Try dx.doi.org instead of doi.org
        if 'doi.org' in url and 'dx.doi.org' not in url:
            alt_url = url.replace('doi.org', 'dx.doi.org')
            try:
                import requests
                response = requests.head(alt_url, timeout=5, allow_redirects=True)
                if response.status_code < 400:
                    return alt_url
            except Exception:
                pass
    
    # Strategy 3: Check web archive
    wayback_url = f"https://web.archive.org/web/{url}"
    try:
        import requests
        response = requests.head(wayback_url, timeout=5, allow_redirects=True)
        if response.status_code < 400:
            return wayback_url
    except Exception:
        pass
    
    # No working alternative found
    return None

def generate_markdown_report(validation_results: List[Dict[str, Any]], output_path: str) -> str:
    """
    Generate a markdown report from validation results.
    
    Args:
        validation_results: List of validation results
        output_path: Path to save the report
        
    Returns:
        Path to the generated report
    """
    valid_refs = [r for r in validation_results if r.get('valid', False)]
    invalid_refs = [r for r in validation_results if not r.get('valid', False)]
    
    report_content = f"""# Reference Validation Report
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Total References: {len(validation_results)}
- Valid References: {len(valid_refs)} ({len(valid_refs)/len(validation_results)*100:.1f}%)
- Invalid References: {len(invalid_refs)} ({len(invalid_refs)/len(validation_results)*100:.1f}%)

## Invalid References
"""
    
    if invalid_refs:
        report_content += """
| URL | Title | Type | Status | Error |
|-----|-------|------|--------|-------|
"""
        for ref in invalid_refs:
            report_content += f"| {ref.get('URL', '')} | {ref.get('Title', '')} | {ref.get('Type', '')} | {ref.get('status', 'Unknown')} | {ref.get('error', '')} |\n"
    else:
        report_content += "\nNo invalid references found.\n"
    
    report_content += "\n## Valid References\n"
    
    if valid_refs:
        report_content += """
| URL | Title | Type | Response Time (s) |
|-----|-------|------|------------------|
"""
        for ref in valid_refs:
            report_content += f"| {ref.get('URL', '')} | {ref.get('Title', '')} | {ref.get('Type', '')} | {ref.get('response_time', 0)} |\n"
    else:
        report_content += "\nNo valid references found.\n"
    
    # Save the report
    md_output_path = output_path.replace('.csv', '.md')
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return md_output_path

def main():
    """Command-line interface for reference validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate references in markdown files')
    parser.add_argument('file_path', help='Path to the markdown file')
    parser.add_argument('--output-dir', help='Directory to save validation reports')
    parser.add_argument('--update', action='store_true', help='Update broken links in the file')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout in seconds')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent workers')
    parser.add_argument('--markdown-report', action='store_true', help='Generate a markdown report')
    
    args = parser.parse_args()
    
    validation_results, report_path = validate_and_update_references(
        args.file_path,
        args.output_dir,
        args.update,
        args.timeout,
        args.max_workers
    )
    
    if args.markdown_report and report_path:
        md_report_path = generate_markdown_report(validation_results, report_path)
        logger.info(f"Markdown report saved to {md_report_path}")

if __name__ == "__main__":
    main() 