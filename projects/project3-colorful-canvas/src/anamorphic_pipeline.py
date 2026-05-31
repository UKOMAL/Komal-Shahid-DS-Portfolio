"""
Anamorphic 3D Billboard Pipeline
Core geometric warping engine for creating perspective-based 3D LED billboard illusions.
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from typing import Optional, Literal
import warnings

from src.depth_estimator import SyntheticDepthEstimator, MiDaSDepthEstimator


class AnamorphicBillboardPipeline:
    """
    Pipeline for creating anamorphic 3D billboard effects.
    Uses depth-based perspective warping to create the illusion of 3D content
    bursting out of a 2D LED display.

    The perspective math models a viewer standing at a known distance from the
    billboard surface. Objects with high depth values are displaced outward
    (appear to pop out) while the background stays flat.
    """

    def __init__(
        self,
        strength: float = 4.0,
        viewer_distance: float = 3.0,
        viewer_angle: float = 0.0,
        max_depth_displacement: float = 0.8,
    ):
        """
        Initialize the pipeline.

        Args:
            strength: Warp intensity multiplier (typically 2.0-6.0).
            viewer_distance: Relative viewer distance for perspective.
                             Higher values = less distortion (viewer far away).
                             Typically 2.0-5.0.
            viewer_angle: Horizontal off-axis viewer angle in degrees.
                          0 = dead center, positive = viewer right of center.
            max_depth_displacement: Maximum depth-based displacement as a
                                    fraction of viewer_distance (0-1). Clamps
                                    the perspective scale to avoid singularities.
        """
        self.strength = strength
        self.viewer_distance = viewer_distance
        self.viewer_angle = viewer_angle
        self.max_depth_displacement = np.clip(max_depth_displacement, 0.1, 0.95)

    @staticmethod
    def generate_synthetic_depth(image_array: np.ndarray) -> np.ndarray:
        """
        Generate a synthetic depth map from an image using edge detection and blur.

        Kept for backward compatibility. For higher quality, use
        ``SyntheticDepthEstimator`` from ``depth_estimator.py``.

        Args:
            image_array: Input image as numpy array (H, W, 3) or (H, W).

        Returns:
            Depth map as numpy array (H, W) with values in [0, 1].
        """
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.astype(np.uint8)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(laplacian)

        if edges.max() > 0:
            edges = edges / edges.max()

        depth = cv2.GaussianBlur(edges, (21, 21), 0)

        if depth.max() > 0:
            depth = depth / depth.max()

        return depth.astype(np.float32)

    @staticmethod
    def apply_anamorphic_warp(
        image: np.ndarray,
        depth_map: np.ndarray,
        strength: float = 4.0,
        viewer_distance: float = 3.0,
        viewer_angle: float = 0.0,
        max_depth_displacement: float = 0.8,
    ) -> np.ndarray:
        """
        Apply perspective-correct anamorphic warp to create a 3D pop-out illusion.

        The warp models a virtual viewer at ``viewer_distance`` units from the
        billboard surface. Each pixel is displaced radially outward from the
        viewer's projected center by an amount that depends on:

        * **depth**: how far the object "pops out" of the surface (0 = flat, 1 = max)
        * **perspScale**: ``viewer_distance / (viewer_distance - depth * max_disp)``
          — objects closer to the viewer are magnified more, matching real
          perspective foreshortening.
        * **strength**: artist-controlled intensity multiplier.
        * **viewer_angle**: off-axis horizontal shift (degrees). A non-zero angle
          biases the displacement direction so the illusion is optimized for a
          viewer who is not standing dead-center.

        When ``viewer_distance`` is very large, ``perspScale → 1`` (orthographic,
        no distortion). When ``depth = 0``, displacement is zero (background
        stays flat). When ``depth = 1`` and ``strength`` is high, maximum pop-out.

        Args:
            image: Input image (H, W, 3), uint8.
            depth_map: Depth map (H, W) in [0, 1].
            strength: Warp intensity multiplier.
            viewer_distance: Relative viewer distance (unitless, >0).
            viewer_angle: Off-axis angle in degrees (0 = centered).
            max_depth_displacement: Caps depth displacement to avoid division
                                    by zero in perspScale (0.1–0.95).

        Returns:
            Warped image (H, W, 3), uint8.
        """
        h, w = depth_map.shape
        max_depth_displacement = np.clip(max_depth_displacement, 0.1, 0.95)

        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij',
        )

        angle_rad = np.deg2rad(viewer_angle)
        center_x = w / 2.0 + np.tan(angle_rad) * viewer_distance * (w / 4.0)
        center_y = h / 2.0

        norm = float(max(h, w))
        dy = (y_coords - center_y) / norm
        dx = (x_coords - center_x) / norm
        radial_dist = np.sqrt(dy ** 2 + dx ** 2)
        radial_dist_safe = np.maximum(radial_dist, 1e-6)

        dir_y = dy / radial_dist_safe
        dir_x = dx / radial_dist_safe

        effective_disp = depth_map * max_depth_displacement
        persp_scale = viewer_distance / (viewer_distance - effective_disp)
        persp_scale = np.clip(persp_scale, 1.0, 10.0)

        foreshorten = 1.0 + depth_map * 0.15

        displacement = depth_map * strength * radial_dist * persp_scale * foreshorten

        map_x = x_coords + dir_x * displacement * (w / norm)
        map_y = y_coords + dir_y * displacement * (h / norm)

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        image_uint8 = image.astype(np.uint8)
        warped = cv2.remap(
            image_uint8,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return warped

    @staticmethod
    def enhance_colors(
        image: np.ndarray,
        saturation: float = 1.8,
        brightness: float = 1.3,
        contrast: float = 1.0,
        mode: Literal['standard', 'led', 'bloom'] = 'standard',
        bloom_radius: int = 25,
        bloom_intensity: float = 0.35,
    ) -> np.ndarray:
        """
        Enhance colors for LED display optimization.

        Modes:
            ``standard``  — Saturation + brightness adjustment (original behavior).
            ``led``       — High saturation + high contrast tuned for LED panels.
            ``bloom``     — Adds a glow/bloom halo around bright areas, simulating
                            LED light bleed.

        Args:
            image: Input image (H, W, 3), uint8 or float [0,1].
            saturation: Saturation multiplier.
            brightness: Brightness multiplier.
            contrast: Contrast multiplier (used in ``led`` and ``bloom`` modes).
            mode: Enhancement mode.
            bloom_radius: Gaussian blur radius for bloom halo (pixels, must be odd).
            bloom_intensity: Blend factor for the bloom layer (0-1).

        Returns:
            Enhanced image (H, W, 3), uint8.
        """
        if image.dtype != np.uint8:
            pil_image = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
        else:
            pil_image = Image.fromarray(image)

        pil_image = ImageEnhance.Color(pil_image).enhance(saturation)
        pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness)

        if mode == 'led':
            pil_image = ImageEnhance.Contrast(pil_image).enhance(max(contrast, 1.4))
            pil_image = ImageEnhance.Color(pil_image).enhance(1.3)

        result = np.array(pil_image)

        if mode == 'bloom':
            result = _apply_bloom(result, bloom_radius, bloom_intensity, contrast)

        return result

    def full_pipeline(
        self,
        input_path: str,
        output_path: str,
        strength: Optional[float] = None,
        enhance: bool = True,
        color_mode: Literal['standard', 'led', 'bloom'] = 'standard',
        depth_method: Literal['legacy', 'synthetic', 'midas'] = 'synthetic',
        midas_model_size: str = 'small',
    ) -> dict:
        """
        Full end-to-end pipeline: load → depth → warp → enhance → save.

        Args:
            input_path: Path to input image.
            output_path: Path to save output image.
            strength: Override default strength (None = use self.strength).
            enhance: Whether to apply color enhancement.
            color_mode: Enhancement mode ('standard', 'led', 'bloom').
            depth_method:
                'legacy'    — uses the simple Laplacian-based method on this class.
                'synthetic' — uses ``SyntheticDepthEstimator`` (multi-scale Sobel).
                'midas'     — uses ``MiDaSDepthEstimator`` (requires torch).
            midas_model_size: MiDaS model variant ('small', 'base', 'large').

        Returns:
            Dictionary with processing metadata.
        """
        strength = strength if strength is not None else self.strength

        image = Image.open(input_path).convert('RGB')
        image_array = np.array(image)

        depth_map = self._estimate_depth(image_array, depth_method, midas_model_size)

        warped = self.apply_anamorphic_warp(
            image_array,
            depth_map,
            strength=strength,
            viewer_distance=self.viewer_distance,
            viewer_angle=self.viewer_angle,
            max_depth_displacement=self.max_depth_displacement,
        )

        if enhance:
            warped = self.enhance_colors(warped, mode=color_mode)

        output_image = Image.fromarray(warped.astype(np.uint8))
        output_image.save(output_path)

        return {
            'input_path': input_path,
            'output_path': output_path,
            'strength': strength,
            'viewer_distance': self.viewer_distance,
            'viewer_angle': self.viewer_angle,
            'depth_method': depth_method,
            'image_size': image_array.shape,
            'depth_map_range': (float(depth_map.min()), float(depth_map.max())),
            'enhanced': enhance,
            'color_mode': color_mode,
        }

    def _estimate_depth(
        self,
        image_array: np.ndarray,
        method: str,
        midas_model_size: str,
    ) -> np.ndarray:
        """Dispatch to the requested depth estimation backend."""
        if method == 'legacy':
            return self.generate_synthetic_depth(image_array)

        if method == 'synthetic':
            estimator = SyntheticDepthEstimator()
            return estimator.estimate_depth(image_array)

        if method == 'midas':
            try:
                estimator = MiDaSDepthEstimator(model_size=midas_model_size)
            except ImportError as exc:
                warnings.warn(
                    f"MiDaS unavailable ({exc}). Falling back to synthetic depth.",
                    stacklevel=2,
                )
                return SyntheticDepthEstimator().estimate_depth(image_array)
            return estimator.estimate_depth(image_array)

        raise ValueError(
            f"Unknown depth_method '{method}'. "
            "Choose from 'legacy', 'synthetic', or 'midas'."
        )

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
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            RGB image as numpy array (height, width, 3).
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 20

        center_y, center_x = height // 2, width // 2
        max_radius = min(height, width) // 2 - 10

        colors = [
            (255, 50, 50),
            (50, 255, 50),
            (50, 50, 255),
            (255, 255, 50),
            (255, 50, 255),
            (50, 255, 255),
        ]

        num_circles = 6
        for i in range(num_circles):
            radius = int(max_radius * (1 - i / num_circles))
            if radius > 5:
                color = colors[i % len(colors)]
                cv2.circle(image, (center_x, center_y), radius, color, 3)

        sq_size = 40
        cv2.rectangle(
            image,
            (center_x - sq_size, center_y - sq_size),
            (center_x + sq_size, center_y + sq_size),
            (200, 200, 200),
            -1,
        )

        num_lines = 12
        for i in range(num_lines):
            angle = 2 * np.pi * i / num_lines
            x_end = int(center_x + max_radius * np.cos(angle))
            y_end = int(center_y + max_radius * np.sin(angle))
            color = colors[i % len(colors)]
            cv2.line(image, (center_x, center_y), (x_end, y_end), color, 2)

        return image


def _apply_bloom(
    image: np.ndarray,
    radius: int = 25,
    intensity: float = 0.35,
    contrast: float = 1.0,
) -> np.ndarray:
    """
    Simulate LED bloom/glow by extracting bright regions, blurring them,
    and additively blending back onto the image.

    Args:
        image: Input image (H, W, 3), uint8.
        radius: Gaussian blur kernel size (must be odd).
        intensity: Blend weight of the bloom layer (0-1).
        contrast: Optional contrast boost applied before bloom extraction.

    Returns:
        Image with bloom effect applied (H, W, 3), uint8.
    """
    pil_img = Image.fromarray(image)
    if contrast != 1.0:
        pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    img_f = np.array(pil_img).astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright_mask = (gray > 180).astype(np.float32)
    bright_mask = cv2.GaussianBlur(bright_mask, (5, 5), 0)

    bloom_layer = img_f * bright_mask[:, :, np.newaxis]

    ksize = radius if radius % 2 == 1 else radius + 1
    bloom_layer = cv2.GaussianBlur(bloom_layer, (ksize, ksize), 0)

    result = img_f + bloom_layer * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("Anamorphic 3D Billboard Pipeline — Demo")
    print("=" * 60)

    pipeline = AnamorphicBillboardPipeline(
        strength=4.0,
        viewer_distance=3.0,
        viewer_angle=0.0,
    )

    print("\n[1/3] Generating synthetic test image...")
    demo_img = pipeline.generate_demo_image(width=512, height=512)
    demo_pil = Image.fromarray(demo_img)
    demo_pil.save('/tmp/demo_input.png')
    print(f"  Saved: /tmp/demo_input.png ({demo_img.shape})")

    print("\n[2/3] Running full pipeline (synthetic depth → warp → enhance)...")
    result = pipeline.full_pipeline(
        input_path='/tmp/demo_input.png',
        output_path='/tmp/demo_output.png',
        strength=4.0,
        enhance=True,
        color_mode='led',
        depth_method='synthetic',
    )
    print(f"  Saved: {result['output_path']}")
    print(f"  Depth range: {result['depth_map_range']}")
    print(f"  Color mode: {result['color_mode']}")

    print("\n[3/3] Running bloom mode variant...")
    result_bloom = pipeline.full_pipeline(
        input_path='/tmp/demo_input.png',
        output_path='/tmp/demo_output_bloom.png',
        strength=4.0,
        enhance=True,
        color_mode='bloom',
        depth_method='synthetic',
    )
    print(f"  Saved: {result_bloom['output_path']}")

    print("\n" + "=" * 60)
    print("Done. Outputs saved to /tmp/demo_output*.png")
    print("=" * 60)
    sys.exit(0)
