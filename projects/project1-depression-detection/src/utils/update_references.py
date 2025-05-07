#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update References Module
Identifies and updates invalid URL references in markdown files based on validation results.
"""

import re
import os
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_references')

def suggest_alternatives(url: str) -> List[str]:
    """
    Suggest alternative URLs for invalid references.
    
    Args:
        url: The invalid URL
        
    Returns:
        List of suggested alternative URLs
    """
    suggestions = []
    parsed = urlparse(url)
    
    # DOI alternatives
    if 'doi.org' in url:
        doi_id = url.split('doi.org/')[-1]
        suggestions.append(f"https://doi.org/{doi_id}")
        suggestions.append(f"https://dx.doi.org/{doi_id}")
        
        # Sci-Hub alternative (for educational purposes only)
        # suggestions.append(f"https://sci-hub.se/{doi_id}")
    
    # arXiv alternatives
    elif 'arxiv.org' in url:
        arxiv_id = re.search(r'(\d+\.\d+)', url)
        if arxiv_id:
            arxiv_id = arxiv_id.group(1)
            suggestions.append(f"https://arxiv.org/abs/{arxiv_id}")
            suggestions.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
    
    # Common domain alternatives and fixes
    else:
        # Try HTTPS if using HTTP
        if parsed.scheme == 'http':
            suggestions.append(url.replace('http://', 'https://'))
        
        # Fix common URL typos
        if 'www' not in url and parsed.netloc:
            suggestions.append(f"{parsed.scheme}://www.{parsed.netloc}{parsed.path}")
            
        # Remove trailing characters that might be causing issues
        clean_url = re.sub(r'[.,;:)]$', '', url)
        if clean_url != url:
            suggestions.append(clean_url)
    
    # Return unique suggestions
    return list(set(suggestions))

def update_references_in_file(file_path: str, validation_results: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    """
    Update references in a markdown file based on validation results.
    
    Args:
        file_path: Path to the markdown file
        validation_results: DataFrame with validation results
        
    Returns:
        Tuple containing the updated content and a dictionary of updated URLs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return content, {}
    
    # Track updated URLs
    updated_urls = {}
    
    # Filter invalid URLs
    invalid_urls = validation_results[~validation_results['Valid']]
    
    if invalid_urls.empty:
        logger.info("No invalid URLs to update")
        return content, {}
    
    # Process each invalid URL
    for _, row in invalid_urls.iterrows():
        invalid_url = row['URL']
        reference = row['Reference']
        
        # Generate suggestions
        suggestions = suggest_alternatives(invalid_url)
        
        if not suggestions:
            logger.warning(f"No suggestions found for {invalid_url}")
            continue
        
        # Use the first suggestion as the replacement
        new_url = suggestions[0]
        
        # Replace the URL in the content
        # Use negative lookbehind to avoid replacing the URL in code blocks
        content = re.sub(
            r'(?<!\`\`\`)' + re.escape(invalid_url),
            new_url,
            content
        )
        
        # Update tracking
        updated_urls[invalid_url] = new_url
        logger.info(f"Updated: {invalid_url} -> {new_url}")
    
    return content, updated_urls

def save_updated_file(file_path: str, content: str, create_backup: bool = True) -> bool:
    """
    Save the updated content back to the file.
    
    Args:
        file_path: Path to the markdown file
        content: Updated content to save
        create_backup: Whether to create a backup of the original file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create backup if requested
        if create_backup:
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
            
        logger.info(f"Successfully updated {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving updated file: {e}")
        return False

def update_references(file_path: str, validation_results: pd.DataFrame, create_backup: bool = True) -> Dict[str, any]:
    """
    Update invalid references in a markdown file based on validation results.
    
    Args:
        file_path: Path to the markdown file
        validation_results: DataFrame with validation results
        create_backup: Whether to create a backup of the original file
        
    Returns:
        Dictionary with update results
    """
    logger.info(f"Updating references in {file_path}")
    
    # Update references
    updated_content, updated_urls = update_references_in_file(file_path, validation_results)
    
    if not updated_urls:
        logger.info("No references were updated")
        return {
            "success": True,
            "file_path": file_path,
            "updates": 0,
            "updated_urls": {}
        }
    
    # Save the updated file
    success = save_updated_file(file_path, updated_content, create_backup)
    
    return {
        "success": success,
        "file_path": file_path,
        "updates": len(updated_urls),
        "updated_urls": updated_urls
    }

def main():
    """Command-line interface to update references in a markdown file."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update invalid references in markdown files")
    parser.add_argument("file_path", help="Path to the markdown file")
    parser.add_argument("validation_csv", help="Path to the validation results CSV file")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    args = parser.parse_args()
    
    try:
        # Load validation results
        validation_results = pd.read_csv(args.validation_csv)
        
        # Update references
        result = update_references(args.file_path, validation_results, not args.no_backup)
        
        # Print summary
        if result["success"]:
            print(f"Successfully updated {result['updates']} references in {args.file_path}")
            if result["updates"] > 0:
                print("\nUpdated URLs:")
                for old_url, new_url in result["updated_urls"].items():
                    print(f"  - {old_url} -> {new_url}")
        else:
            print(f"Failed to update references in {args.file_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main() 