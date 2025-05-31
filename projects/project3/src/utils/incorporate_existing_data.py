import os
import shutil
import json
import hashlib
from pathlib import Path

class ExistingDataIncorporator:
    """Utility class to incorporate existing data outputs into the training dataset."""
    
    def __init__(self, 
                 dataset_dir="dataset/anamorphic",
                 output_dirs=None):
        """Initialize with dataset directory and output directories to incorporate."""
        self.dataset_dir = dataset_dir
        self.raw_dir = os.path.join(dataset_dir, "raw")
        self.processed_dir = os.path.join(dataset_dir, "processed")
        self.metadata_dir = os.path.join(dataset_dir, "metadata")
        self.categories_dir = os.path.join(dataset_dir, "categories")
        
        # Default output directories to incorporate
        if output_dirs is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            project3_root = Path(__file__).resolve().parent.parent.parent
            self.output_dirs = [
                os.path.join(project_root, "anamorphic_output"),
                os.path.join(project_root, "holographic_output"),
                os.path.join(project_root, "parallax_3d_output"),
                os.path.join(project3_root, "billboard_3d_output"),
                os.path.join(project3_root, "true_anamorphic")
            ]
        else:
            self.output_dirs = output_dirs
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.categories_dir, exist_ok=True)
        
        # Initialize metadata
        self.metadata_path = os.path.join(self.metadata_dir, "metadata.json")
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "dataset_name": "3D Anamorphic Illusions Dataset",
                "version": "1.0",
                "sources": [],
                "images": []
            }
    
    def _is_valid_image(self, filename):
        """Check if the file is a valid image or video file."""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.webm', '.mov']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)
    
    def _get_category_from_path(self, path):
        """Determine category from file path."""
        path_lower = path.lower()
        if "anamorphic" in path_lower:
            return "anamorphic_illusion"
        elif "billboard" in path_lower:
            return "billboard_3d"
        elif "holographic" in path_lower:
            if "wave" in path_lower:
                return "wave_holographic"
            return "holographic"
        elif "parallax" in path_lower:
            return "parallax_3d"
        else:
            return "misc_3d_effect"
    
    def _copy_file(self, source_path, filename, category):
        """Copy a file to the raw directory and add to metadata."""
        # Generate a hash for the file
        with open(source_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Determine file extension
        _, extension = os.path.splitext(source_path)
        
        # Create new filename
        new_filename = f"existing_{category}_{file_hash}{extension}"
        dest_path = os.path.join(self.raw_dir, new_filename)
        
        # Copy the file
        shutil.copy2(source_path, dest_path)
        
        # Add to metadata
        img_metadata = {
            "id": file_hash,
            "filename": new_filename,
            "original_path": source_path,
            "description": f"Existing {category.replace('_', ' ')} example",
            "category": category,
            "source": "existing_outputs"
        }
        
        self.metadata["images"].append(img_metadata)
        
        # Create category directory if it doesn't exist
        category_dir = os.path.join(self.categories_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Create symlink in categories directory
        category_path = os.path.join(category_dir, new_filename)
        if os.path.exists(category_path):
            os.remove(category_path)
        
        # Create relative path to the raw file
        rel_path = os.path.relpath(dest_path, category_dir)
        try:
            os.symlink(rel_path, category_path)
        except OSError as e:
            print(f"Warning: Could not create symlink ({e}). Continuing anyway.")
        
        return True
    
    def incorporate_data(self):
        """Incorporate existing data into the dataset."""
        print("=== Incorporating Existing Data into Dataset ===")
        
        total_incorporated = 0
        source_added = False
        
        for output_dir in self.output_dirs:
            if not os.path.exists(output_dir):
                print(f"Directory not found: {output_dir}")
                continue
            
            print(f"Processing directory: {output_dir}")
            incorporated_count = 0
            
            # Walk through the directory
            for root, _, files in os.walk(output_dir):
                for filename in files:
                    if not self._is_valid_image(filename):
                        continue
                    
                    source_path = os.path.join(root, filename)
                    category = self._get_category_from_path(source_path)
                    
                    try:
                        if self._copy_file(source_path, filename, category):
                            incorporated_count += 1
                            total_incorporated += 1
                    except Exception as e:
                        print(f"Error processing file {source_path}: {e}")
            
            if incorporated_count > 0 and not source_added:
                # Add to sources in metadata
                self.metadata["sources"].append({
                    "name": "existing_outputs",
                    "description": "Existing output files from previous runs",
                    "count": total_incorporated
                })
                source_added = True
            
            print(f"Incorporated {incorporated_count} files from {output_dir}")
        
        # Save updated metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Total incorporated: {total_incorporated} files")
        print(f"Metadata saved to: {self.metadata_path}")
        
        return total_incorporated

if __name__ == "__main__":
    incorporator = ExistingDataIncorporator()
    incorporator.incorporate_data() 