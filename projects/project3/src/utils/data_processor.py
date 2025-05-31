import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import cv2
import glob
from tqdm import tqdm

class AnamorphicDataProcessor:
    def __init__(self, dataset_dir="dataset/anamorphic"):
        """Initialize the data processor with the dataset directory."""
        self.dataset_dir = dataset_dir
        self.raw_dir = os.path.join(dataset_dir, "raw")
        self.processed_dir = os.path.join(dataset_dir, "processed")
        self.metadata_dir = os.path.join(dataset_dir, "metadata")
        self.categories_dir = os.path.join(dataset_dir, "categories")
        
        # Load metadata
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
    
    def analyze_dataset(self):
        """Analyze the dataset and print statistics."""
        if not os.path.exists(self.metadata_path):
            print("Metadata file not found. Run data collection first.")
            return
        
        # Count images by category
        category_counts = {}
        source_counts = {}
        
        for img in self.metadata["images"]:
            # Count by category
            category = img.get("category", "unknown")
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
            
            # Count by source
            source = img.get("source", "unknown")
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        # Print statistics
        print("Dataset Statistics:")
        print(f"Total images: {len(self.metadata['images'])}")
        
        print("Images by category:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count}")
        
        print("Images by source:")
        for source, count in source_counts.items():
            print(f"  - {source}: {count}")
    
    def _is_valid_image(self, img_path):
        """Check if an image is valid and can be processed."""
        try:
            # Check if it's an image file
            if not os.path.isfile(img_path):
                return False
                
            # Skip video files
            if any(img_path.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.webm']):
                return False
                
            # Try to open the image
            img = Image.open(img_path)
            img.verify()  # Verify that it's an image
            
            # Check image dimensions
            img = Image.open(img_path)
            if img.width < 64 or img.height < 64:
                return False
                
            return True
        except Exception as e:
            print(f"Invalid image {img_path}: {e}")
            return False

    def _preprocess_image(self, img_path, target_size):
        """Preprocess an image for the neural network."""
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Normalize to [-1, 1]
            img = img.astype(np.float32) / 127.5 - 1.0
            
            return img
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            return None
    
    def process_dataset(self, target_size=(512, 512)):
        """Process the raw dataset and save normalized images."""
        processed_dir = os.path.join(self.processed_dir, f"{target_size[0]}x{target_size[1]}")
        os.makedirs(processed_dir, exist_ok=True)
        
        print(f"Processing images to {target_size[0]}x{target_size[1]}...")
        
        # Create a list of raw image paths
        raw_images = []
        for img_meta in self.metadata["images"]:
            img_path = os.path.join(self.raw_dir, img_meta["filename"])
            if self._is_valid_image(img_path):
                raw_images.append((img_path, img_meta))
        
        # Process each image
        valid_count = 0
        for img_path, img_meta in tqdm(raw_images):
            # Generate output path
            filename = os.path.basename(img_path)
            output_path = os.path.join(processed_dir, filename)
            
            # Skip if already processed
            if os.path.exists(output_path):
                valid_count += 1
                continue
            
            # Preprocess image
            processed_img = self._preprocess_image(img_path, target_size)
            if processed_img is not None:
                # Save as numpy array
                np.save(output_path.replace('.jpg', '.npy').replace('.jpeg', '.npy').replace('.png', '.npy'), 
                        processed_img)
                valid_count += 1
        
        print(f"Processed {valid_count} valid images out of {len(raw_images)} total images")
        print(f"Processed images saved to {processed_dir}")
        
        return valid_count
    
    def create_tf_dataset(self, target_size=(512, 512), batch_size=4, validation_split=0.2, shuffle_buffer=1000):
        """Create TensorFlow datasets for training and validation."""
        # Process the dataset if not already done
        processed_dir = os.path.join(self.processed_dir, f"{target_size[0]}x{target_size[1]}")
        if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
            print("Processing dataset...")
            self.process_dataset(target_size)
        
        # Get processed image paths
        image_paths = glob.glob(os.path.join(processed_dir, "*.npy"))
        if not image_paths:
            # Try to find jpg/png files and convert them to npy
            image_paths = glob.glob(os.path.join(self.raw_dir, "*.jpg")) + \
                         glob.glob(os.path.join(self.raw_dir, "*.jpeg")) + \
                         glob.glob(os.path.join(self.raw_dir, "*.png"))
            
            if not image_paths:
                print("No processed images found. Processing the dataset...")
                self.process_dataset(target_size)
                image_paths = glob.glob(os.path.join(processed_dir, "*.npy"))
        
        if not image_paths:
            print("ERROR: No images found after processing. Check dataset directory.")
            return None, None
        
        print(f"Found {len(image_paths)} processed images")
        
        # Shuffle paths
        np.random.shuffle(image_paths)
        
        # Split into training and validation
        split_idx = int(len(image_paths) * (1 - validation_split))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]
        
        # Create dataset loading function
        def load_and_preprocess_image(path):
            try:
                # Convert the TensorFlow tensor path to string
                path_str = path.numpy().decode('utf-8') if hasattr(path, 'numpy') else path
                
                if path_str.endswith('.npy'):
                    # Load numpy array directly
                    img = np.load(path_str)
                else:
                    # Load and process image file
                    img = self._preprocess_image(path_str, target_size)
                    if img is None:
                        # Return a zero tensor as a fallback
                        img = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
                
                # No labels for now, return the same image as both input and target
                return img, img
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Return a zero tensor as a fallback
                zero_img = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
                return zero_img, zero_img
        
        # Create TensorFlow datasets
        def create_dataset(paths):
            # Convert paths to dataset
            ds = tf.data.Dataset.from_tensor_slices(paths)
            # Map paths to images
            ds = ds.map(
                lambda path: tf.py_function(
                    load_and_preprocess_image,
                    [path], 
                    [tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            # Set output shapes explicitly
            ds = ds.map(
                lambda x, y: (
                    tf.ensure_shape(x, [target_size[0], target_size[1], 3]),
                    tf.ensure_shape(y, [target_size[0], target_size[1], 3])
                )
            )
            # Filter out invalid images (all zeros)
            ds = ds.filter(lambda x, y: tf.reduce_sum(tf.abs(x)) > 0)
            # Cache for better performance
            ds = ds.cache()
            # Shuffle, batch, and prefetch
            ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(paths)))
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds
        
        # Create training and validation datasets
        train_ds = create_dataset(train_paths)
        val_ds = create_dataset(val_paths)
        
        # Try to get a sample batch to verify the dataset works
        try:
            for images, _ in train_ds.take(1):
                print(f"Dataset created successfully. Batch shape: {images.shape}")
                break
        except Exception as e:
            print(f"Error creating dataset: {e}")
        
        return train_ds, val_ds

if __name__ == "__main__":
    processor = AnamorphicDataProcessor()
    processor.analyze_dataset()
    train_ds, val_ds = processor.create_tf_dataset(target_size=(256, 256), batch_size=8)
    
    # Test the dataset
    for images, targets in train_ds.take(1):
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Images min/max: {tf.reduce_min(images):.4f}/{tf.reduce_max(images):.4f}") 