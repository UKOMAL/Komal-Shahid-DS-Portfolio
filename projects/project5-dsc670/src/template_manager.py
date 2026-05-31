#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Template Manager for Notely

Manages template storage, retrieval, and organization.
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import datetime

class TemplateManager:
    """
    Manages templates for the Notely application.
    
    Features:
    - Template storage and organization
    - Template metadata tracking
    - Template search and retrieval
    - Template collections
    """
    
    def __init__(self, template_dir="templates"):
        """
        Initialize the template manager with a template directory.
        
        Args:
            template_dir: Directory for template storage
        """
        self.template_dir = template_dir
        os.makedirs(template_dir, exist_ok=True)
        
        # Create a metadata file if it doesn't exist
        self.metadata_file = os.path.join(template_dir, "template_metadata.json")
        if not os.path.exists(self.metadata_file):
            self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize an empty metadata file"""
        metadata = {
            "templates": {},
            "collections": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load metadata from file"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._initialize_metadata()
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
    
    def _save_metadata(self, metadata):
        """Save metadata to file"""
        metadata["last_updated"] = datetime.datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_template(self, template_path, name=None, description=None, tags=None):
        """
        Add a template to the manager.
        
        Args:
            template_path: Path to the template file
            name: Name for the template (default: filename without extension)
            description: Description of the template
            tags: List of tags for the template
            
        Returns:
            Path to the stored template
        """
        # Get template filename
        template_filename = os.path.basename(template_path)
        
        # Use filename as name if not provided
        if name is None:
            name = os.path.splitext(template_filename)[0]
        
        # Default values
        if description is None:
            description = f"Template {name}"
        if tags is None:
            tags = []
        
        # Copy template to template directory if it's not already there
        if os.path.dirname(os.path.abspath(template_path)) != os.path.abspath(self.template_dir):
            destination = os.path.join(self.template_dir, template_filename)
            shutil.copy2(template_path, destination)
            template_path = destination
        
        # Update metadata
        metadata = self._load_metadata()
        metadata["templates"][template_filename] = {
            "name": name,
            "description": description,
            "tags": tags,
            "created": datetime.datetime.now().isoformat()
        }
        self._save_metadata(metadata)
        
        return template_path
    
    def get_template(self, template_id):
        """
        Get a template by ID (filename or name).
        
        Args:
            template_id: Template filename or name
            
        Returns:
            Path to the template, or None if not found
        """
        metadata = self._load_metadata()
        
        # Check if ID is a filename
        if template_id in metadata["templates"]:
            return os.path.join(self.template_dir, template_id)
        
        # Check if ID is a template name
        for filename, info in metadata["templates"].items():
            if info["name"] == template_id:
                return os.path.join(self.template_dir, filename)
        
        return None
    
    def list_templates(self, tag=None):
        """
        List all templates, optionally filtered by tag.
        
        Args:
            tag: Optional tag to filter by
            
        Returns:
            List of template information dictionaries
        """
        metadata = self._load_metadata()
        templates = []
        
        for filename, info in metadata["templates"].items():
            if tag is None or tag in info["tags"]:
                template_info = info.copy()
                template_info["filename"] = filename
                template_info["path"] = os.path.join(self.template_dir, filename)
                templates.append(template_info)
        
        return templates
    
    def create_collection(self, collection_name, template_ids=None):
        """
        Create a collection of templates.
        
        Args:
            collection_name: Name for the collection
            template_ids: List of template filenames or names
            
        Returns:
            Collection information dictionary
        """
        if template_ids is None:
            template_ids = []
        
        # Create collection directory
        collection_dir = os.path.join(self.template_dir, "collections", collection_name)
        os.makedirs(collection_dir, exist_ok=True)
        
        # Add templates to collection
        template_paths = []
        for template_id in template_ids:
            template_path = self.get_template(template_id)
            if template_path:
                template_paths.append(template_path)
        
        # Update metadata
        metadata = self._load_metadata()
        if "collections" not in metadata:
            metadata["collections"] = {}
        
        metadata["collections"][collection_name] = {
            "templates": template_ids,
            "created": datetime.datetime.now().isoformat()
        }
        self._save_metadata(metadata)
        
        return metadata["collections"][collection_name]
    
    def get_collection(self, collection_name):
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection information dictionary
        """
        metadata = self._load_metadata()
        if "collections" not in metadata or collection_name not in metadata["collections"]:
            return None
        
        collection_info = metadata["collections"][collection_name]
        template_paths = []
        
        for template_id in collection_info["templates"]:
            template_path = self.get_template(template_id)
            if template_path:
                template_paths.append(template_path)
        
        collection_info["template_paths"] = template_paths
        return collection_info
    
    def remove_template(self, template_id):
        """
        Remove a template from the manager.
        
        Args:
            template_id: Template filename or name
            
        Returns:
            True if removed, False if not found
        """
        template_path = self.get_template(template_id)
        if not template_path:
            return False
        
        # Get filename
        filename = os.path.basename(template_path)
        
        # Update metadata
        metadata = self._load_metadata()
        if filename in metadata["templates"]:
            del metadata["templates"][filename]
            self._save_metadata(metadata)
        
        # Remove file
        if os.path.exists(template_path):
            os.remove(template_path)
        
        return True
    
    def search_templates(self, query):
        """
        Search for templates by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching template information dictionaries
        """
        metadata = self._load_metadata()
        results = []
        query = query.lower()
        
        for filename, info in metadata["templates"].items():
            # Check name, description, and tags
            if (query in info["name"].lower() or 
                query in info["description"].lower() or 
                any(query in tag.lower() for tag in info["tags"])):
                
                template_info = info.copy()
                template_info["filename"] = filename
                template_info["path"] = os.path.join(self.template_dir, filename)
                results.append(template_info)
        
        return results
    
    def export_template(self, template_id, destination):
        """
        Export a template to a specified location.
        
        Args:
            template_id: Template filename or name
            destination: Destination path
            
        Returns:
            Path to the exported template, or None if not found
        """
        template_path = self.get_template(template_id)
        if not template_path:
            return None
        
        # Copy template to destination
        shutil.copy2(template_path, destination)
        return destination

if __name__ == "__main__":
    # Example usage
    manager = TemplateManager()
    
    # Add a template
    template_path = "example_template.png"
    if os.path.exists(template_path):
        manager.add_template(template_path, "Example Template", "An example template", ["basic", "example"])
    
    # List all templates
    templates = manager.list_templates()
    for template in templates:
        print(f"Template: {template['name']} ({template['filename']})")
        print(f"  Description: {template['description']}")
        print(f"  Tags: {', '.join(template['tags'])}")
        print() 