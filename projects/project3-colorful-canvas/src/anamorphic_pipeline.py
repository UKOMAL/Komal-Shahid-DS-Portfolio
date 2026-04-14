"""
Anamorphic 3D Billboard Pipeline
Core geometric warping engine for creating perspective-based 3D LED billboard illusions.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from typing import Tuple, Optional
import warnings


class AnamorphicBillboardPipeline:
    """
    Pipeline for creating anamorphic 3D billboard effects.
    Uses depth-based perspective warping to create illusion of 3D content
    bursting out of a 2D LED display.
    """

    def __init__(self, strength: float = 4.0, viewer_distance: float = 3.0):
        """
        Initialize the pipeline.

        Args:
            strength: Warp intensity multiplier (typically 2.0-6.0)
            viewer_distance: Relative viewer distance for perspective (typically 2.0-5.0)
        """
        self.strength = strength
        self.viewer_distance = viewer_distance

    @staticmethod
    def generate_synthetic_depth(image_array: np.ndarray) -> np.ndarray:
        """
        Generate a synthetic depth map from an image using edge detection and blur.

        Algorithm:
        1. Compute Laplacian edge detection
        2. Normalize edges
        3. Apply Gaussian blur for smooth depth falloff
        4. Normalize to [0, 1]

        Args:
            image_array: Input image as numpy array (H, W, 3) or (H, W)

        Returns:
            Depth map as numpy array (H, W) with values in [0, 1]
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.astype(np.uint8)

        # Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(laplacian)

        # Normalize edges
        if edges.max() > 0:
            edges = edges / edges.max()

        # Apply Gaussian blur for smooth depth gradient
        depth = cv2.GaussianBlur(edges, (21, 21), 0)

        # Normalize to [0, 1]
        if depth.max() > 0:
            depth = depth / depth.max()

        return depth.astype(np.float32)

    @staticmethod
    def apply_anamorphic_warp(
        image: np.ndarray,
        depth_map: np.ndarray,
        strength: float = 4.0,
        viewer_distance: float = 3.0
    ) -> np.ndarray:
        """
        Apply perspective-based anamorphic warp to create 3D illusion.

        For each pixel, displacement is proportional to:
        - depth_value: determines how far to push outward
        - distance_from_center: radial position determines direction
        - strength: multiplier for warp intensity

        Args:
            image: Input image (H, W, 3)
            depth_map: Depth map (H, W) with values in [0, 1]
            strength: Warp intensity (default 4.0)
            viewer_distance: Relative viewer distance (default 3.0)

        Returns:
            Warped image with same shape as input
        """
        h, w = depth_map.shape

        # Create normalized coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Center coordinates
        center_y, center_x = h / 2.0, w / 2.0

        # Distance from center (normalized)
        dy = (y_coords - center_y) / max(h, w)
        dx = (x_coords - center_x) / max(h, w)
        radial_dist = np.sqrt(dy**2 + dx**2)

        # Prevent division by zero
        radial_dist = np.maximum(radial_dist, 1e-6)

        # Direction unit vectors (normalized)
        dir_y = dy / radial_dist
        dir_x = dx / radial_dist

        # Displacement magnitude: depth * strength * radial_distance
        displacement = depth_map * strength * radial_dist

        # Apply displacement in radial direction
        map_x = x_coords + dir_x * displacement * (w / max(h, w))
        map_y = y_coords + dir_y * displacement * (h / max(h, w))

        # Convert to float32 for cv2.remap
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # Remap the image
        image_uint8 = image.astype(np.uint8)
        warped = cv2.remap(
            image_uint8,
            map_x, map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return warped

    @staticmethod
    def enhance_colors(
        image: np.ndarray,
        saturation: float = 1.8,
        brightness: float = 1.3
    ) -> np.ndarray:
        """
        Enhance colors for LED display optimization.
        Increases saturation and brightness to match LED aesthetic.

        Args:
            image: Input image (H, W, 3)
            saturation: Saturation multiplier (default 1.8)
            brightness: Brightness multiplier (default 1.3)

        Returns:
            Enhanced image as numpy array
        """
        # Convert to PIL Image
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image.astype(np.uint8))
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))

        # Enhance saturation
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)

        # Enhance brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness)

        # Convert back to numpy
        return np.array(pil_image)

    def full_pipeline(
        self,
        input_path: str,
        output_path: str,
        strength: Optional[float] = None,
        enhance: bool = True
    ) -> dict:
        """
        Full end-to-end pipeline: load image → depth map → warp → enhance → save.

        Args:
            input_path: Path to input image
            output_path: Path to save output image
            strength: Override default strength (if None, use self.strength)
            enhance: Whether to apply color enhancement

        Returns:
            Dictionary with processing metadata
        """
        strength = strength or self.strength

        # Load image
        image = Image.open(input_path).convert('RGB')
        image_array = np.array(image)

        # Generate depth map
        depth_map = self.generate_synthetic_depth(image_array)

        # Apply warp
        warped = self.apply_anamorphic_warp(
            image_array,
            depth_map,
            strength=strength,
            viewer_distance=self.viewer_distance
        )

        # Enhance colors if requested
        if enhance:
            warped = self.enhance_colors(warped)

        # Save output
        output_image = Image.fromarray(warped.astype(np.uint8))
        output_image.save(output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'strength': strength,
            'image_size': image_array.shape,
            'depth_map_range': (depth_map.min(), depth_map.max()),
            'enhanced': enhance
        }

    @staticmethod
    def generate_demo_image(width: int = 512, height: int = 512) -> np.ndarray:
        """
        Generate a synthetic geometric test image with concentric shapes
        at different "depth" levels. No external data required.

        Creates:
        - Concentric circles with alternating colors
        - Central square
        - Radiating lines

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            RGB image as numpy array (height, width, 3)
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 20  # Dark background

        center_y, center_x = height // 2, width // 2
        max_radius = min(height, width) // 2 - 10

        # Define color palette (vibrant)
        colors = [
            (255, 50, 50),    # Red
            (50, 255, 50),    # Green
            (50, 50, 255),    # Blue
            (255, 255, 50),   # Yellow
            (255, 50, 255),   # Magenta
            (50, 255, 255),   # Cyan
        ]

        # Draw concentric circles
        num_circles = 6
        for i in range(num_circles):
            radius = int(max_radius * (1 - i / num_circles))
            if radius > 5:
                color = colors[i % len(colors)]
                cv2.circle(image, (center_x, center_y), radius, color, 3)

        # Draw central square
        sq_size = 40
        cv2.rectangle(
            image,
            (center_x - sq_size, center_y - sq_size),
            (center_x + sq_size, center_y + sq_size),
            (200, 200, 200),
            -1
        )

        # Draw radiating lines
        num_lines = 12
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            x_end = int(center_x + max_radius * np.cos(angle))
            y_end = int(center_y + max_radius * np.sin(angle))
            color = colors[i % len(colors)]
            cv2.line(image, (center_x, center_y), (x_end, y_end), color, 2)

        return image


if __name__ == '__main__':
    # Demo usage
    pipeline = AnamorphicBillboardPipeline(strength=4.0)

    # Generate demo image
    demo_img = pipeline.generate_demo_image(width=512, height=512)
    demo_pil = Image.fromarray(demo_img)
    demo_pil.save('/tmp/demo_input.png')

    # Run pipeline
    result = pipeline.full_pipeline(
        '/tmp/demo_input.png',
        '/tmp/demo_output.png',
        strength=4.0
    )
    print("Pipeline complete:", result)
