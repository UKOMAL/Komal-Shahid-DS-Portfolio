# Colorful Canvas — Anamorphic 3D Billboard

**Course:** DSC680 Applied Data Science · Project 3  
**Institution:** Bellevue University  
**Author:** Komal Shahid  
**Date:** 2026

---

## Overview

A real-time anamorphic 3D billboard effect that creates the illusion of content breaking out of an L-shaped corner LED screen. Built with Three.js WebGL for the browser and OpenCV for the Python pipeline.

Anamorphic billboards are the technology behind viral installations like Seoul's Coex Wave, Tokyo's Shinjuku 3D Cat, and Nike's Air Max Day billboard. This project implements the full rendering pipeline — from off-axis perspective projection to depth-based displacement warping.

---

## Live Demo

Open `index.html` in a modern browser (Chrome, Firefox, Edge). No build step required.

---

## Features

- **Anamorphic 3D Effect**: Content appears to physically pop out of the billboard from the designed "sweet spot" viewing angle
- **L-Shaped Corner Billboard**: Two-face screen geometry matching real commercial installations
- **Portal Rendering**: Off-axis perspective projection renders content onto billboard faces
- **Interactive Controls**:
  - Orbit camera to explore the 3D illusion
  - Angle slider for smooth viewing angle control
  - Sweet Spot snap button
  - Before/After toggle (3D vs Flat comparison)
  - Screenshot capture
- **Product Scene**: Juice bottle with fruit burst — glass materials, realistic fruit, splash effects
- **Post-Processing**: Bloom, vignette, floating particles, floor reflections
- **Python Pipeline**: OpenCV-based anamorphic warp with perspective projection, depth estimation (synthetic + MiDaS)

---

## Technical Architecture

### Browser (index.html)

- Three.js r168 via CDN (ES module imports)
- WebGLRenderTarget for portal/render-to-texture
- Off-axis perspective projection (CameraUtils-style)
- Clipping planes for depth layering
- UnrealBloomPass + custom vignette ShaderPass
- MeshPhysicalMaterial (transmission, IOR, clearcoat, iridescence)

### Python Pipeline (src/)

- `anamorphic_pipeline.py`: Perspective-correct warp with viewer_distance, viewer_angle, perspScale
- `depth_estimator.py`: SyntheticDepthEstimator (multi-scale Sobel) + MiDaSDepthEstimator (PyTorch)
- Three color enhancement modes: standard, LED, bloom

---

## How Anamorphic Billboards Work

Anamorphic billboards exploit forced perspective: content is pre-distorted so that from one specific "sweet spot" viewing angle, the flat 2D display creates a convincing 3D illusion. The trick relies on an L-shaped (corner-wrapped) LED screen — the two perpendicular faces provide binocular depth cues that the brain interprets as volumetric space. Objects rendered with off-axis projection appear to break through the screen boundary, while background elements remain flush with the display surface.

---

## Project Structure

```
project3-colorful-canvas/
├── index.html                  # Three.js WebGL anamorphic billboard (self-contained)
├── README.md                   # Project documentation
├── DESIGN_NOTES.md             # Artistic vision and technical design rationale
├── requirements.txt            # Python dependencies
└── src/
    ├── anamorphic_pipeline.py  # Core warp engine + color enhancement + full pipeline
    └── depth_estimator.py      # Synthetic depth (Sobel) + MiDaS depth (PyTorch)
```

---

## Requirements

### Browser Demo
- Any modern browser with WebGL2 support (Chrome 80+, Firefox 75+, Edge 80+, Safari 15+)
- No installation or build step required

### Python Pipeline
- Python 3.8+
- See `requirements.txt` for dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Run the anamorphic pipeline demo
python -m src.anamorphic_pipeline
```

---

## Usage

### Python Pipeline

```python
from src.anamorphic_pipeline import AnamorphicBillboardPipeline

pipeline = AnamorphicBillboardPipeline(
    strength=4.0,
    viewer_distance=3.0,
    viewer_angle=0.0,
)

result = pipeline.full_pipeline(
    input_path="input.png",
    output_path="output.png",
    strength=4.0,
    enhance=True,
    color_mode="led",
    depth_method="synthetic",
)

print(result)
```

### Depth Estimation

```python
from src.depth_estimator import SyntheticDepthEstimator

estimator = SyntheticDepthEstimator()
depth_map = estimator.estimate_depth(image_array)
estimator.visualize_depth(depth_map, "depth_heatmap.png")
```

---

## References

- **Seoul Wave** — d'strict, Coex K-Pop Square (2020)
- **Shinjuku 3D Cat** — Cross Shinjuku Vision, MicroAd Digital Signage (2021)
- **Nike Air Max Day** — Cross Shinjuku Vision, Tokyo (2022)
- Ranftl et al., "Towards Robust Monocular Depth Estimation" (2022) — MiDaS
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"

---

## Course

DSC680 — Applied Data Science / Bellevue University

---

**Last Updated:** 2026-05-31
