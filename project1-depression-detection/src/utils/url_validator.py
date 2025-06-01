#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
URL Validator Module
Functions for extracting and validating URLs from markdown files.
"""

import re
import os
import time
import logging
import requests
from typing import Dict, List, Union, Optional, Any
from pathlib import Path
import concurrent.futures
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('url_validator')

def extract_urls_from_markdown(file_path: str) -> List[Dict[str, str]]:
    """
    Extract all URLs from a markdown file.
    
    Args:
        file_path: Path to the markdown file
    
    Returns:
        List of dictionaries with URL, title, and type information
    """
    # Ensure file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    references = []
    url_dict = {}
    
    # Extract inline links: [text](url)
    inline_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    for title, url in inline_links:
        if url not in url_dict:
            ref_type = "Inline"
            url_dict[url] = {
                "URL": url,
                "Title": title,
                "Type": ref_type
            }
    
    # Extract reference-style links: [text][ref] ... [ref]: url
    ref_links = re.findall(r'\[([^\]]+)\]\[([^\]]+)\]', content)
    ref_defs = re.findall(r'\[([^\]]+)\]:\s*([^\s]+)(?:\s+"([^"]+)")?', content)
    
    # Create a mapping of references
    ref_map = {}
    for ref_id, url, title in ref_defs:
        ref_map[ref_id] = (url, title or "")
    
    # Add reference-style links to the results
    for text, ref_id in ref_links:
        if ref_id in ref_map:
            url, title = ref_map[ref_id]
            if url not in url_dict:
                ref_type = "Reference"
                title = title or text
                url_dict[url] = {
                    "URL": url,
                    "Title": title,
                    "Type": ref_type
                }
    
    # Look for DOI patterns
    doi_patterns = [
        r'https?://doi\.org/[a-zA-Z0-9./]+',
        r'https?://dx\.doi\.org/[a-zA-Z0-9./]+'
    ]
    
    for pattern in doi_patterns:
        doi_matches = re.findall(pattern, content)
        for url in doi_matches:
            if url not in url_dict:
                ref_type = "DOI"
                url_dict[url] = {
                    "URL": url,
                    "Title": "DOI Reference",
                    "Type": ref_type
                }
    
    # Look for arXiv patterns
    arxiv_matches = re.findall(r'https?://arxiv\.org/abs/[a-zA-Z0-9./]+', content)
    for url in arxiv_matches:
        if url not in url_dict:
            ref_type = "arXiv"
            url_dict[url] = {
                "URL": url,
                "Title": "arXiv Paper",
                "Type": ref_type
            }
    
    # Convert dictionary to list
    references = list(url_dict.values())
    
    return references

def validate_url(url_data: Dict[str, str], timeout: int = 10) -> Dict[str, Any]:
    """
    Validate a single URL and extract its title.
    
    Args:
        url_data: Dictionary containing URL, title, and type
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with validation results
    """
    url = url_data["URL"]
    result = {
        "URL": url,
        "Title": url_data["Title"],
        "Type": url_data["Type"],
        "valid": False,
        "status": "Unknown",
        "response_time": 0,
        "error": ""
    }
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        result["status"] = response.status_code
        result["response_time"] = response.elapsed.total_seconds()
        
        # Check if response is successful (status code 200-399)
        if 200 <= response.status_code < 400:
            result["valid"] = True
            
            # Try to extract title from HTML
            if "text/html" in response.headers.get("Content-Type", ""):
                soup = BeautifulSoup(response.text, "html.parser")
                if soup.title and soup.title.string:
                    result["Title"] = soup.title.string.strip()
        else:
            result["error"] = f"HTTP Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
        result["status"] = "Timeout"
    except requests.exceptions.SSLError:
        result["error"] = "SSL Error"
        result["status"] = "SSL Error"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection Error"
        result["status"] = "Connection Error"
    except Exception as e:
        result["error"] = str(e)
        result["status"] = "Error"
    
    return result

def validate_urls(references: List[Dict[str, str]], timeout: int = 10, max_workers: int = 5) -> List[Dict[str, Any]]:
    """
    Validate a list of URLs concurrently.
    
    Args:
        references: List of dictionaries containing URL information
        timeout: Request timeout in seconds
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of dictionaries with validation results
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(validate_url, ref, timeout): ref 
            for ref in references
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            ref = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                status = "✅" if result["valid"] else "❌"
                logger.info(f"[{i+1}/{len(references)}] {status} {ref['URL']}")
            except Exception as e:
                logger.error(f"Error validating {ref['URL']}: {e}")
                results.append({
                    "URL": ref["URL"],
                    "Title": ref["Title"],
                    "Type": ref["Type"],
                    "valid": False,
                    "status": "Error",
                    "response_time": 0,
                    "error": str(e)
                })
    
    return results

def extract_title_from_doi(doi_url: str, timeout: int = 10) -> Optional[str]:
    """
    Extract title from a DOI URL.
    
    Args:
        doi_url: DOI URL
        timeout: Request timeout in seconds
        
    Returns:
        Title string or None if extraction failed
    """
    try:
        headers = {
            "Accept": "application/vnd.citationstyles.csl+json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        
        response = requests.get(doi_url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            return data.get("title")
    except Exception as e:
        logger.error(f"Error extracting title from DOI {doi_url}: {e}")
    
    return None

def extract_title_from_arxiv(arxiv_url: str, timeout: int = 10) -> Optional[str]:
    """
    Extract title from an arXiv URL.
    
    Args:
        arxiv_url: arXiv URL
        timeout: Request timeout in seconds
        
    Returns:
        Title string or None if extraction failed
    """
    arxiv_id_match = re.search(r'arxiv\.org/abs/([a-zA-Z0-9.]+)', arxiv_url)
    if not arxiv_id_match:
        return None
    
    arxiv_id = arxiv_id_match.group(1)
    api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    
    try:
        response = requests.get(api_url, timeout=timeout)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "xml")
            title_tag = soup.find("title")
            if title_tag and title_tag.string:
                return title_tag.string.strip()
    except Exception as e:
        logger.error(f"Error extracting title from arXiv {arxiv_url}: {e}")
    
    return None

def main():
    """Command-line interface for URL validation."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Validate URLs in markdown files")
    parser.add_argument("file_path", help="Path to the markdown file")
    parser.add_argument("--output", help="Output file path for validation results (JSON format)")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds")
    parser.add_argument("--workers", type=int, default=5, help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    try:
        # Extract URLs
        references = extract_urls_from_markdown(args.file_path)
        print(f"Found {len(references)} unique URLs in {args.file_path}")
        
        if not references:
            print("No URLs found to validate.")
            return
        
        # Validate URLs
        print(f"Validating URLs with timeout {args.timeout}s and {args.workers} workers...")
        results = validate_urls(references, args.timeout, args.workers)
        
        # Count valid/invalid URLs
        valid_count = sum(1 for r in results if r.get("valid", False))
        invalid_count = len(results) - valid_count
        
        print(f"\nValidation complete:")
        print(f"  - Total URLs: {len(results)}")
        print(f"  - Valid URLs: {valid_count} ({valid_count/len(results)*100:.1f}%)")
        print(f"  - Invalid URLs: {invalid_count} ({invalid_count/len(results)*100:.1f}%)")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
        # Print invalid URLs
        if invalid_count > 0:
            print("\nInvalid URLs:")
            for i, result in enumerate(results):
                if not result.get("valid", False):
                    print(f"  {i+1}. {result['URL']}")
                    print(f"     Status: {result['status']}")
                    print(f"     Error: {result['error']}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main() 