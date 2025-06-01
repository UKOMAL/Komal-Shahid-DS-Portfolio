#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test DOI URL Extractor
Extracts and prints DOI URLs from the white paper to identify any incomplete DOIs.
"""

import os
import re
from pathlib import Path

def extract_doi_urls_with_context(file_path):
    """Extract DOI URLs from a markdown file with line numbers and context."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # DOI pattern
    doi_pattern = r'https?://doi\.org/10\.1007/s13278(?:-\d+-\d+-\d+)?'
    
    # Find all DOI URLs with context
    results = []
    for i, line in enumerate(lines, 1):
        matches = re.findall(doi_pattern, line)
        if matches:
            for match in matches:
                context = f"Line {i}: {line.strip()}"
                results.append((match, context))
    
    return results

def main():
    """Main function."""
    # Define paths
    project_root = Path(__file__).resolve().parent.parent
    white_paper_path = os.path.join(
        project_root,
        "docs",
        "white_paper.md"
    )
    
    # Extract DOI URLs with context
    doi_urls = extract_doi_urls_with_context(white_paper_path)
    
    # Print results
    print(f"Found {len(doi_urls)} DOI URLs matching pattern in {white_paper_path}:")
    for i, (url, context) in enumerate(doi_urls, 1):
        print(f"{i}. URL: {url}")
        print(f"   Context: {context}")
        print()

    # Also check the binary content for incomplete DOIs
    with open(white_paper_path, 'rb') as f:
        binary_content = f.read()
    
    search_patterns = [b"doi.org/10.1007/s13278"]
    for pattern in search_patterns:
        positions = [m.start() for m in re.finditer(pattern, binary_content)]
        if positions:
            print(f"Found binary matches for {pattern} at positions: {positions}")
            for pos in positions:
                context = binary_content[max(0, pos-20):min(len(binary_content), pos+len(pattern)+20)]
                print(f"Binary context: {context}")

if __name__ == "__main__":
    main() 