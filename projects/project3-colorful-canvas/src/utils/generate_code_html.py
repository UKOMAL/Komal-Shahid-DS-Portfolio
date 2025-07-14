#!/usr/bin/env python3
"""
Generate HTML file with syntax highlighting for the anamorphic billboard code
"""
import os
import sys
import re
from pathlib import Path
import pygments
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def create_html_with_syntax_highlighting(source_file, output_file):
    """Create an HTML file with syntax highlighting from Python source"""
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"Error: Source file {source_file} not found.")
        return False
    
    # Read source code
    with open(source_file, 'r') as f:
        code = f.read()
    
    # Get title from filename
    title = os.path.basename(source_file)
    title = title.replace('.py', '').replace('_', ' ').title()
    
    # Generate CSS
    css = HtmlFormatter().get_style_defs('.highlight')
    
    # Highlight code
    highlighted_code = highlight(code, PythonLexer(), HtmlFormatter())
    
    # Create HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', 'Arial', sans-serif;
            line-height: 1.5;
            margin: 40px;
            background: #f8f8f8;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .highlight {{
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 16px;
            overflow: auto;
            font-size: 14px;
            font-family: 'Consolas', 'Courier New', monospace;
        }}
        .project-info {{
            margin-bottom: 20px;
        }}
        {css}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="project-info">
        <p><strong>Project:</strong> ColorfulCanvas - Advanced 3D Rendering Techniques</p>
        <p><strong>Description:</strong> Implementation of an anamorphic billboard generator 
        that creates 3D objects that appear to pop out only when viewed from the correct angle.</p>
    </div>
    {highlighted_code}
    <div class="footer">
        <p><em>This file is part of the ColorfulCanvas project documentation.</em></p>
    </div>
</body>
</html>"""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write HTML to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Successfully created HTML file: {output_file}")
    return True

if __name__ == "__main__":
    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    
    # Source file can be provided as command line argument
    if len(sys.argv) > 1:
        source_file = sys.argv[1]
    else:
        # Default to the working_anamorphic_billboard-4.py if no argument provided
        source_file = os.path.join(os.path.expanduser("~"), "Downloads", "working_anamorphic_billboard-4.py")
    
    # Output file
    output_dir = os.path.join(project_root, "docs", "final")
    output_file = os.path.join(output_dir, "anamorphic_billboard_full.html")
    
    create_html_with_syntax_highlighting(source_file, output_file)
    print(f"Now you can open '{output_file}' in a browser and print to PDF") 