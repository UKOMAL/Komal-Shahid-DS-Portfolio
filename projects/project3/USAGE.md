# Colorful Canvas: Usage Guide

This guide provides examples of how to use the consolidated `colorful_canvas.py` script to create various 3D visual effects.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Shadow Box Effect

```bash
python src/colorful_canvas.py --effect shadow_box --input_image path/to/your/image.jpg
```

### Run Screen Pop Effect

```bash
python src/colorful_canvas.py --effect screen_pop --input_image path/to/your/image.jpg --depth_factor 2.0
```

## Examples with Sample Images

If you have sample images in the `examples` directory, you can try:

```bash
# Create a shadow box effect
python src/colorful_canvas.py --effect shadow_box --input_image examples/original.jpg --output_dir examples/output

# Create a screen pop effect
python src/colorful_canvas.py --effect screen_pop --input_image examples/original.jpg --output_dir examples/output
```

## Advanced Usage

### Adjusting Depth Factor

The depth factor controls how pronounced the 3D effect is:

```bash
# Subtle effect
python src/colorful_canvas.py --effect screen_pop --input_image examples/original.jpg --depth_factor 1.0

# Strong effect
python src/colorful_canvas.py --effect screen_pop --input_image examples/original.jpg --depth_factor 3.0
```

### Custom Output Directory

Specify a custom output directory:

```bash
python src/colorful_canvas.py --effect shadow_box --input_image examples/original.jpg --output_dir my_custom_effects
```

## Troubleshooting

### Memory Issues

If you encounter memory issues with large images, try:
1. Resize your input image to a smaller resolution
2. Use a device with more available memory (GPU recommended)

### No Depth Model Error

If you see an error about the depth estimation model not loading:
1. Ensure you have an internet connection (model is downloaded on first use)
2. Make sure you have sufficient disk space
3. Try running with a smaller image first

### Poor Quality Results

For best results:
1. Use high-resolution images
2. Choose images with clear foreground/background separation
3. Use images with good lighting and contrast