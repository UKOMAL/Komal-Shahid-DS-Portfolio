#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ColorDepthMiDaS Model Script
This script trains the ColorDepthMiDaS model on available depth data
and saves the trained model for use in the milestone 3 pipeline.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Project: Colorful Canvas AI Art Studio
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Add src directory to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import our ColorDepthMiDaS class
from src.color_depth_midas import ColorDepthMiDaS, train_color_depth_model

class DepthDataset(Dataset):
    """Dataset for training the color depth model"""
    
    def __init__(self, benchmark_dir, transform=None):
        """Initialize dataset with benchmark images"""
        self.benchmark_dir = Path(benchmark_dir)
        self.image_files = sorted([f for f in self.benchmark_dir.glob("*.png") if "depth" not in f.name])
        print(f"Found {len(self.image_files)} images in {benchmark_dir}")
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get RGB image and create corresponding depth color representation"""
        # Load RGB image
        img_path = self.image_files[idx]
        rgb_image = Image.open(img_path).convert("RGB")
        
        # Generate basic grayscale depth map using existing MiDaS model
        temp_midas = ColorDepthMiDaS(device="cpu")
        depth_gray = temp_midas.generate_depth_map(rgb_image)
        
        # Convert grayscale to RGB color representation
        # This creates a color mapping that the model will learn to reproduce
        depth_color = self._create_color_depth_representation(depth_gray)
        
        # Convert to tensors
        rgb_tensor = self._image_to_tensor(rgb_image)
        depth_color_tensor = self._image_to_tensor(depth_color)
        
        return rgb_tensor, depth_color_tensor
    
    def _create_color_depth_representation(self, gray_depth):
        """Create color representation from grayscale depth"""
        depth_np = np.array(gray_depth)
        
        # Create RGB channels based on depth values
        r_channel = depth_np.copy()
        g_channel = 255 - depth_np  # Invert for green channel
        b_channel = np.abs(depth_np - 128)  # Middle values for blue
        
        # Combine channels
        color_depth = np.stack([r_channel, g_channel, b_channel], axis=2)
        return Image.fromarray(color_depth.astype(np.uint8))
    
    def _image_to_tensor(self, image):
        """Convert PIL image to normalized tensor"""
        img_np = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC to CHW
        return img_tensor


def main():
    """Main training function"""
    print("🚀 Starting ColorDepthMiDaS Training Pipeline")
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = project_root / "models" / "midas"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "midas_color.pth"
    
    # Load training data from benchmarks directory
    benchmark_dir = project_root / "data" / "benchmarks"
    if not benchmark_dir.exists():
        print(f"❌ Benchmark directory not found: {benchmark_dir}")
        print("Using sample data from input directory instead")
        benchmark_dir = project_root / "data" / "input"
    
    # Create dataset and dataloader
    dataset = DepthDataset(benchmark_dir)
    
    # Set training parameters
    batch_size = 2
    iterations = 1000 if len(dataset) > 5 else 200  # Less iterations for small datasets
    learning_rate = 1e-4
    
    print(f"Training ColorDepthMiDaS model with {len(dataset)} images")
    print(f"Batch size: {batch_size}, Iterations: {iterations}, Learning rate: {learning_rate}")
    
    # Train the model
    try:
        model_path = train_color_depth_model(
            dataset=dataset,
            out_ckpt=str(model_path),
            iters=iterations,
            batch_size=batch_size,
            lr=learning_rate,
            device=device
        )
        print(f"✅ Training complete! Model saved to {model_path}")
        print("Now you can run the milestone 3 script to generate anamorphic billboards with color depth.")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 