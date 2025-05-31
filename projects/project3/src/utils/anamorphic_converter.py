import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from anamorphic.generator import AnamorphicGenerator

class AnamorphicConverter:
    def __init__(self, model_path=None):
        """Initialize the anamorphic converter."""
        self.img_size = (512, 512, 3)
        
        if model_path and os.path.exists(model_path):
            # Load the pre-trained model
            print(f"Loading model from {model_path}")
            self.generator = tf.keras.models.load_model(model_path)
        else:
            # Create a new generator
            print("Creating a new generator model")
            generator_model = AnamorphicGenerator(img_size=self.img_size)
            self.generator = generator_model.generator
    
    def convert_image(self, input_path, output_path=None, viewing_angle='front'):
        """Convert a 2D image to a 3D anamorphic illusion."""
        # Load the image
        img = self._load_image(input_path)
        
        # Apply the transformation
        transformed_img = self._apply_transformation(img, viewing_angle)
        
        # Save the result if output path is provided
        if output_path:
            self._save_image(transformed_img, output_path)
        
        return transformed_img
    
    def _load_image(self, input_path):
        """Load and preprocess an image."""
        # Load the image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Failed to load image from {input_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to the model's input size
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        return img
    
    def _apply_transformation(self, img, viewing_angle):
        """Apply the appropriate transformation based on viewing angle."""
        # For model-based transformation
        if hasattr(self, 'generator'):
            # Reshape for model input
            img_batch = np.expand_dims(img, axis=0)
            
            # Generate the anamorphic version
            transformed_img = self.generator.predict(img_batch)[0]
            
            # Denormalize
            transformed_img = (transformed_img + 1.0) * 127.5
            transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
            
            return transformed_img
        
        # For manual transformation (if no model is available)
        elif viewing_angle == 'side':
            # Apply side-view transformation (stretch horizontally)
            return self._apply_side_view_transform(img)
        else:
            # Apply front-view transformation (default)
            return self._apply_front_view_transform(img)
    
    def _apply_front_view_transform(self, img):
        """Apply front-view transformation."""
        # Denormalize
        img_denorm = (img + 1.0) * 127.5
        img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
        
        height, width = img_denorm.shape[:2]
        
        # Define source and destination points for perspective transform
        src_points = np.float32([
            [0, 0],               # Top-left
            [width, 0],           # Top-right
            [width, height],      # Bottom-right
            [0, height]           # Bottom-left
        ])
        
        dst_points = np.float32([
            [width * 0.1, height * 0.1],    # Top-left (moved in)
            [width * 0.9, height * 0.1],    # Top-right (moved in)
            [width, height],                # Bottom-right (unchanged)
            [0, height]                     # Bottom-left (unchanged)
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        transformed_img = cv2.warpPerspective(img_denorm, matrix, (width, height))
        
        return transformed_img
    
    def _apply_side_view_transform(self, img):
        """Apply side-view transformation."""
        # Denormalize
        img_denorm = (img + 1.0) * 127.5
        img_denorm = np.clip(img_denorm, 0, 255).astype(np.uint8)
        
        height, width = img_denorm.shape[:2]
        
        # Define source and destination points for perspective transform
        src_points = np.float32([
            [0, 0],               # Top-left
            [width, 0],           # Top-right
            [width, height],      # Bottom-right
            [0, height]           # Bottom-left
        ])
        
        dst_points = np.float32([
            [width * 0.3, 0],             # Top-left (moved right)
            [width, 0],                   # Top-right (unchanged)
            [width, height],              # Bottom-right (unchanged)
            [width * 0.3, height]         # Bottom-left (moved right)
        ])
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        transformed_img = cv2.warpPerspective(img_denorm, matrix, (width, height))
        
        return transformed_img
    
    def _save_image(self, img, output_path):
        """Save the image to the specified output path."""
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Save the image
        cv2.imwrite(output_path, img)
        
        print(f"Image saved to {output_path}")
    
    def visualize_transformation(self, input_path, output_dir=None, angles=None):
        """Visualize the transformation from different viewing angles."""
        if angles is None:
            angles = ['front', 'side']
        
        # Load the original image
        original_img = self._load_image(input_path)
        original_img_display = (original_img + 1.0) * 127.5
        original_img_display = np.clip(original_img_display, 0, 255).astype(np.uint8)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, len(angles) + 1, figsize=(5 * (len(angles) + 1), 5))
        
        # Display the original image
        axes[0].imshow(original_img_display)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Display the transformed images
        for i, angle in enumerate(angles):
            transformed_img = self._apply_transformation(original_img, angle)
            axes[i + 1].imshow(transformed_img)
            axes[i + 1].set_title(f"{angle.capitalize()} View")
            axes[i + 1].axis('off')
        
        plt.tight_layout()
        
        # Save the visualization if output_dir is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"{base_filename}_visualization.png")
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        
        plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert 2D images to 3D anamorphic illusions")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--model", type=str, default=None, help="Path to pre-trained model")
    parser.add_argument("--angle", type=str, default="front", choices=["front", "side"], help="Viewing angle")
    parser.add_argument("--visualize", action="store_true", help="Visualize the transformation")
    
    args = parser.parse_args()
    
    converter = AnamorphicConverter(model_path=args.model)
    
    if args.visualize:
        converter.visualize_transformation(args.input, os.path.dirname(args.output) if args.output else None)
    else:
        converter.convert_image(args.input, args.output, args.angle) 