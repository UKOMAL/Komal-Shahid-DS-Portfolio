"""
Depth Estimation Module
Pure numpy/scipy-based depth estimation for anamorphic effects.
Production deployment note: Use MiDaS for state-of-the-art monocular depth estimation.
"""

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from typing import Optional, List
import warnings


class SyntheticDepthEstimator:
    """
    Pure numpy/scipy depth estimator using multi-scale edge density,
    distance transforms, and radial gradients. Fast, requires no GPU
    or ML models. Good for synthetic/stylized content.

    Production alternative: MiDaSDepthEstimator (see below)
    """

    def __init__(self, center_weight: float = 0.3, edge_weight: float = 0.4,
                 distance_weight: float = 0.3):
        """
        Initialize the depth estimator.

        Args:
            center_weight: Weight for center-to-edge radial gradient [0, 1].
            edge_weight: Weight for multi-scale edge response [0, 1].
            distance_weight: Weight for distance-transform component [0, 1].
        """
        total = center_weight + edge_weight + distance_weight
        self.center_weight = center_weight / total
        self.edge_weight = edge_weight / total
        self.distance_weight = distance_weight / total

    def estimate_depth(self, image_array: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using multi-scale Sobel edges, distance transform,
        and radial center-to-edge gradient.

        Algorithm:
        1. Compute multi-scale Sobel gradients (ksize 3, 5, 7) and combine
        2. Apply distance transform from thresholded edges
        3. Generate smooth radial gradient (center=1, edges=0)
        4. Blend all three channels with configurable weights
        5. Normalize to [0, 1]

        Args:
            image_array: Input image (H, W, 3) or (H, W), uint8 or float.

        Returns:
            Depth map (H, W) with values in [0, 1], float32.
        """
        gray = self._to_grayscale(image_array)

        edge_response = self._multi_scale_sobel(gray)
        distance_channel = self._edge_distance_transform(edge_response)
        radial_channel = self._radial_gradient(gray.shape)

        depth = (self.edge_weight * edge_response
                 + self.distance_weight * distance_channel
                 + self.center_weight * radial_channel)

        depth = cv2.GaussianBlur(depth, (15, 15), 0)

        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 0:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    @staticmethod
    def _to_grayscale(image_array: np.ndarray) -> np.ndarray:
        """Convert image to uint8 grayscale."""
        if len(image_array.shape) == 3:
            return cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return image_array.astype(np.uint8)

    @staticmethod
    def _multi_scale_sobel(gray: np.ndarray, scales: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute multi-scale Sobel gradient magnitude and combine.

        Args:
            gray: Grayscale image (H, W), uint8.
            scales: List of Sobel kernel sizes (must be odd). Defaults to [3, 5, 7].

        Returns:
            Normalized gradient magnitude (H, W) in [0, 1], float32.
        """
        if scales is None:
            scales = [3, 5, 7]

        combined = np.zeros(gray.shape, dtype=np.float64)

        for ksize in scales:
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            mag = np.sqrt(sx ** 2 + sy ** 2)
            mag_max = mag.max()
            if mag_max > 0:
                mag /= mag_max
            combined += mag

        cmax = combined.max()
        if cmax > 0:
            combined /= cmax

        return combined.astype(np.float32)

    @staticmethod
    def _edge_distance_transform(edge_response: np.ndarray,
                                 threshold: float = 0.15) -> np.ndarray:
        """
        Distance transform from edge regions: farther from edges = higher value.

        Args:
            edge_response: Normalized edge map (H, W) in [0, 1].
            threshold: Edge binarization threshold.

        Returns:
            Normalized distance channel (H, W) in [0, 1], float32.
        """
        edge_mask = (edge_response > threshold).astype(np.uint8)

        non_edge_mask = 1 - edge_mask
        distance = ndimage.distance_transform_edt(non_edge_mask)

        dmax = distance.max()
        if dmax > 0:
            distance /= dmax

        return distance.astype(np.float32)

    @staticmethod
    def _radial_gradient(shape: tuple) -> np.ndarray:
        """
        Generate smooth center-to-edge radial gradient. Center = 1 (close to
        viewer / foreground), edges = 0 (far / background).

        Args:
            shape: (H, W) of the output gradient.

        Returns:
            Radial gradient (H, W) in [0, 1], float32.
        """
        h, w = shape[:2]
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        max_r = np.sqrt(cy ** 2 + cx ** 2)
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        gradient = 1.0 - (r / max_r)
        return np.clip(gradient, 0, 1).astype(np.float32)

    @staticmethod
    def visualize_depth(
        depth_map: np.ndarray,
        save_path: str,
        title: str = "Depth Map"
    ) -> None:
        """
        Visualize depth map as a heatmap and save.

        Args:
            depth_map: Depth map (H, W) with values in [0, 1].
            save_path: Path to save visualization PNG.
            title: Plot title.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(depth_map, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Depth Value', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    @staticmethod
    def compare_depths(
        original_image: np.ndarray,
        depth_map: np.ndarray,
        save_path: str
    ) -> None:
        """
        Create side-by-side comparison of original image and depth map.

        Args:
            original_image: Original image (H, W, 3).
            depth_map: Depth map (H, W).
            save_path: Path to save comparison.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(depth_map, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Estimated Depth Map', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        plt.colorbar(im, ax=axes[1], label='Depth')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()


class MiDaSDepthEstimator:
    """
    Production-grade monocular depth estimation using Intel's MiDaS v2.1
    (via PyTorch Hub). Falls back gracefully when torch is not installed.

    Model variants:
        'small'  — MiDaS v2.1 Small  (~70 MB, fastest, ~15 FPS on CPU)
        'base'   — DPT-Hybrid        (~300 MB, good balance)
        'large'  — DPT-Large          (~1.2 GB, best quality)

    Installation (optional):
        pip install torch torchvision timm
    """

    _HUB_NAMES = {
        'small': 'MiDaS_small',
        'base': 'DPT_Hybrid',
        'large': 'DPT_Large',
    }

    def __init__(self, model_size: str = 'small', device: Optional[str] = None):
        """
        Initialize MiDaS estimator.

        Args:
            model_size: 'small', 'base', or 'large'.
            device: 'cpu', 'cuda', or None (auto-detect).

        Raises:
            ImportError: If torch is not installed.
            ValueError: If model_size is not recognized.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "MiDaSDepthEstimator requires PyTorch. Install with:\n"
                "  pip install torch torchvision timm\n"
                "Use SyntheticDepthEstimator for a CPU-only alternative."
            )

        if model_size not in self._HUB_NAMES:
            raise ValueError(
                f"model_size must be one of {list(self._HUB_NAMES.keys())}, "
                f"got '{model_size}'"
            )

        self.model_size = model_size
        self._torch = torch

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        hub_name = self._HUB_NAMES[model_size]
        self.model = torch.hub.load('intel-isl/MiDaS', hub_name, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        if model_size == 'large':
            self.transform = midas_transforms.dpt_transform
        elif model_size == 'base':
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def estimate_depth(self, image_array: np.ndarray) -> np.ndarray:
        """
        Estimate depth using MiDaS.

        MiDaS outputs inverse-relative depth (higher = closer). This method
        normalizes the output to [0, 1] where 1 = closest to viewer.

        Args:
            image_array: Input image (H, W, 3), uint8 RGB.

        Returns:
            Depth map (H, W) in [0, 1], float32. 1 = near, 0 = far.
        """
        torch = self._torch

        if image_array.dtype != np.uint8:
            image_array = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)

        input_batch = self.transform(image_array).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_array.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()

        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 0:
            depth = (depth - dmin) / (dmax - dmin)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)


if __name__ == '__main__':
    estimator = SyntheticDepthEstimator()

    test_image = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                test_image[i:i+32, j:j+32] = 255

    depth = estimator.estimate_depth(test_image)
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.3f}, {depth.max():.3f}]")

    estimator.visualize_depth(depth, '/tmp/depth_heatmap.png')
    estimator.compare_depths(test_image, depth, '/tmp/depth_comparison.png')
    print("Visualizations saved to /tmp/")

    try:
        midas = MiDaSDepthEstimator(model_size='small')
        midas_depth = midas.estimate_depth(test_image)
        print(f"MiDaS depth shape: {midas_depth.shape}")
        print(f"MiDaS depth range: [{midas_depth.min():.3f}, {midas_depth.max():.3f}]")
    except ImportError as exc:
        print(f"MiDaS unavailable: {exc}")
