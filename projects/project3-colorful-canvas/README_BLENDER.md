# Anamorphic Billboard Generator with AI Integration

This script provides a complete implementation for creating professional 3D anamorphic billboard effects in Blender with optional AI-enhanced image processing.

## Features

- Seoul-style LED corner projection
- Screen pop-out effects
- Shadow box anamorphic distortion
- AI-enhanced depth map generation
- Dynamic 3D geometry creation
- Professional lighting and camera setup
- Particle effects and floating elements
- Automated parameter optimization based on image analysis

## Requirements

- Blender 2.93 or newer
- Python 3.7+
- Optional AI dependencies: 
  - NumPy
  - PIL (Pillow)
  - PyTorch (for advanced depth map generation)

## Usage

### Running in Blender

1. Open Blender
2. Go to the Scripting tab
3. Open the script `anamorphic_billboard_consolidated.py`
4. Update the `IMAGE_PATH` variable with your image path
5. Run the script

### Available Effects

The script provides three main effect types:

1. **Seoul-Style Corner Projection**
   ```python
   create_seoul_style_billboard(image_path)
   ```

2. **Screen Pop-Out Effect**
   ```python
   create_screen_popup_billboard(image_path)
   ```

3. **Shadow Box Effect**
   ```python
   create_shadow_box_billboard(image_path)
   ```

### Customizing Parameters

For full control over the effect parameters:

```python
main(
    image_path,              # Path to input image
    output_path,             # Path for rendered output
    effect_type="shadow_box", # "seoul_corner", "screen_pop", or "shadow_box"
    ai_strength=1.5          # Intensity of the effect (0.5-3.0)
)
```

## Output

The script generates:
- 3D anamorphic scene in Blender
- Processed image with applied effect
- AI-generated depth map (if available)
- Final render of the scene (if output_path is provided)

## Advanced Usage

### Creating Custom Shapes

```python
curve_points = [
    (0, 0, 0),
    (1, 1, 0),
    (2, 0, 0),
    (3, -1, 0)
]
custom_shape = create_custom_shape_from_curve(curve_points)
```

### Adding Animation

```python
# Animate the floating elements
animate_floating_elements()
```

## Integration with ColorfulCanvasAI

The script automatically detects and uses the ColorfulCanvasAI module if available, providing enhanced image processing capabilities. If not available, it falls back to basic image processing.

## Troubleshooting

- If you encounter import errors, ensure the ColorfulCanvasAI module is in the correct path
- For memory errors, reduce the subdivision levels or particle count
- For render issues, adjust the lighting setup or camera position 