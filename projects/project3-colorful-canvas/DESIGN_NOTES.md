# Design Notes — Colorful Canvas Anamorphic 3D Billboard

## Artistic Vision

This project recreates the viral "impossible 3D" effect seen on corner-mounted LED billboards in Seoul, Tokyo, and other global cities. The goal is a product advertisement — a juice bottle surrounded by bursting fruit — that appears to physically break out of a flat screen when viewed from the correct angle.

The visual identity is premium and cinematic: dark environments, vivid LED-saturated colors, volumetric lighting, and glass/liquid materials that catch light realistically.

---

## Real-World References

| Installation | Location | Key Technique |
|---|---|---|
| Seoul Wave | Coex K-Pop Square, Seoul | Virtual ocean crashing out of corner LED |
| 3D Cat | Cross Shinjuku Vision, Tokyo | Photorealistic cat peering from curved display |
| Nike Air Max Day | Shinjuku, Tokyo | Sneakers with decorative elements at multiple depths |
| Juice Billboard | Various | Beverage bottle with fruit burst, strong depth layers |

All of these share a common principle: an L-shaped (corner-wrapped) LED screen, pre-distorted content rendered for a specific "sweet spot" viewing angle, and aggressive depth layering to sell the 3D illusion.

---

## Technical Approach

### Browser Rendering (Three.js)

The browser demo uses a full 3D scene rendered with Three.js to simulate what a production LED billboard would display:

1. **L-Shaped Geometry**: Two perpendicular planes (front face + side face) representing the physical corner billboard structure.
2. **Off-Axis Projection**: A virtual camera positioned at the "sweet spot" renders the 3D product scene onto the billboard faces using render-to-texture (WebGLRenderTarget). The projection is intentionally off-axis to match how a pedestrian would view the actual billboard.
3. **Product Scene**: A juice bottle (MeshPhysicalMaterial with transmission, IOR, clearcoat) surrounded by fruit geometry with particle effects, bloom post-processing, and floor reflections.
4. **Interactive Exploration**: Users can orbit the camera to see how the illusion breaks from non-ideal angles, with a slider to smoothly animate viewing angle.

### Python Pipeline (OpenCV)

The Python pipeline implements the mathematical core of anamorphic warping for static images:

1. **Depth Estimation**: Multi-scale Sobel edge detection combined with distance transforms and radial gradients produces a depth map where foreground objects have high values (close to viewer) and background has low values (flush with screen).
2. **Perspective Warp**: Each pixel is displaced radially outward from the viewer's projected center. The displacement magnitude is: `depth × strength × radial_distance × perspScale × foreshorten`. The `perspScale` term (`viewer_distance / (viewer_distance - depth_displacement)`) ensures objects closer to the viewer are magnified — matching real perspective foreshortening.
3. **Color Enhancement**: LED panels have extreme saturation and brightness. Three modes (standard, LED, bloom) simulate the characteristic vivid appearance of high-nit displays.

---

## Key Design Decisions

### Why Two Implementations?

The browser version and Python version serve different purposes:

- **Browser (Three.js)**: Real-time interactive demonstration. Shows the 3D illusion in its natural context (a corner billboard). Allows exploration of viewing angles. Portfolio-ready.
- **Python (OpenCV)**: Batch processing pipeline. Takes any 2D image and produces an anamorphic-warped output. Scriptable, testable, extendable to video. Demonstrates the underlying math without GPU rendering.

### Why Not Generative AI?

Anamorphic effects are fundamentally a **geometric projection problem**, not a content generation problem. The illusion comes from pre-distorting known geometry for a known viewing angle — classical computer vision, not neural networks. MiDaS is offered as an optional upgrade for depth estimation on photographs, but the core warp is pure linear algebra.

### L-Shape vs Flat Screen

A flat screen cannot produce convincing anamorphic effects because there's no binocular depth cue — the viewer's eyes both see the same flat plane. The L-shaped corner provides the critical "wrap-around" that the brain interprets as volumetric space. This is why all viral anamorphic billboards use corner-mounted displays.

---

## Color & Material Language

| Element | Treatment |
|---|---|
| Background | Near-black (#080b14) with subtle blue tint |
| Billboard frame | Dark metallic with LED light bleed glow |
| Product (bottle) | Glass material — transmission, IOR 1.5, clearcoat |
| Fruit elements | Subsurface scattering approximation, vivid saturation |
| Particles | Additive blending, bloom-enhanced |
| Floor | Reflective plane with distance fade |

---

## Performance Considerations

- **Browser**: Targets 60 FPS on integrated GPUs. Bloom pass is the most expensive operation. Scene complexity is carefully bounded (low poly fruit, instanced particles).
- **Python**: ~0.5s per 512×512 image on CPU. The `cv2.remap` call is the bottleneck. Scales linearly with pixel count. Video processing is frame-by-frame (no temporal coherence).

---

## Future Directions

- Temporal coherence for video (optical flow-guided warp)
- Multi-angle rendering (adaptive content for tracked viewer position)
- Real LED hardware integration (Art-Net / DMX512 output)
- Custom MiDaS fine-tuning on billboard-specific depth distributions
