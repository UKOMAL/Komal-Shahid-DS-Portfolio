# Colorful Canvas: AI Art Studio

A toolkit for creating 3D visual illusions and effects from 2D images.

## Features

- **Shadow Box Effect**: Create a realistic shadow box/display case illusion from any image
- **Screen Pop Effect**: Make objects appear to pop out of the screen with depth-based displacement and chromatic aberration

## Requirements

```
numpy
opencv-python
torch
transformers
pillow
matplotlib
```

## Installation

1. Install the required dependencies:

```bash
pip install numpy opencv-python torch transformers pillow matplotlib
```

2. Clone this repository:

```bash
git clone https://github.com/yourusername/colorful-canvas.git
cd colorful-canvas
```

## Usage

The consolidated script `colorful_canvas.py` provides a simple interface for generating various effects:

### Shadow Box Effect

```bash
python src/colorful_canvas.py --effect shadow_box --input_image path/to/your/image.jpg
```

This creates a realistic shadow box display case illusion with:
- Depth-based 3D enhancement
- White frame with inner border
- Drop shadow for 3D effect
- Optional glass reflection

### Screen Pop Effect

```bash
python src/colorful_canvas.py --effect screen_pop --input_image path/to/your/image.jpg --depth_factor 2.0
```

This creates an effect that makes objects appear to pop out of the screen with:
- Depth-based displacement mapping
- Chromatic aberration for enhanced 3D effect
- Increased contrast based on depth

## Command Line Arguments

- `--effect`: Type of effect to generate (`shadow_box` or `screen_pop`)
- `--input_image`: Path to the input image
- `--output_dir`: Output directory (defaults to effect-specific directory)
- `--depth_factor`: Depth factor for 3D effects (1.0-3.0, default: 2.0)

## Output

For each effect, the script creates:
- A directory containing all output files
- The final result image
- Intermediate images (depth map, enhanced 3D)
- A visualization of the entire process

## Example

Input image:
![Input image](examples/original.jpg)

Shadow box effect:
![Shadow box effect](examples/shadow_box.jpg)

Screen pop effect:
![Screen pop effect](examples/screen_pop.jpg)

## How It Works

1. **Depth Estimation**: The script uses a pre-trained depth estimation model to extract depth information from the 2D image
2. **3D Enhancement**: Applies transformations based on depth to create a 3D illusion
3. **Post-processing**: Adds additional effects (frames, shadows, chromatic aberration) to enhance the illusion

## Notes

- Processing time depends on image size and complexity
- Best results are achieved with images that have clear foreground objects and depth variation
- For optimal quality, use high-resolution images with good lighting 