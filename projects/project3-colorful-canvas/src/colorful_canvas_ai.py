"""
Colorful Canvas: AI Art Studio
A comprehensive toolkit for creating stunning 3D visual illusions from 2D images

Features:
- Shadow Box Effect: Creates realistic display case illusions with depth-based 3D enhancement
- Screen Pop Effect: Makes objects appear to come out of the screen with chromatic aberration
- Anamorphic Billboard Effect: Creates urban advertising-style 3D illusions
- Depth estimation using state-of-the-art neural networks
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

try:
    import torch
    from transformers import DPTForDepthEstimation, DPTImageProcessor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not available. Using OpenCV for depth estimation.")

def load_image(image_path):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image '{image_path}': {e}")
        return None

def save_image(image, output_path):
    """
    Save an image to a file.
    
    Args:
        image (PIL.Image or numpy.ndarray): Image to save
        output_path (str): Path to save the image to
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(
                np.clip(image, 0, 255).astype(np.uint8)
            )
        
        # Save the image
        image.save(output_path)
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"Error saving image to '{output_path}': {e}")

def resize_image(image, max_size=1024):
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image (PIL.Image): Image to resize
        max_size (int): Maximum size of the longest dimension
        
    Returns:
        PIL.Image: Resized image
    """
    # Get current size
    width, height = image.size
    
    # If image is already smaller than max_size, return as is
    if width <= max_size and height <= max_size:
        return image
    
    # Calculate new size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # Resize image
    return image.resize((new_width, new_height), Image.LANCZOS)

def normalize_depth_map(depth_map):
    """
    Normalize a depth map to 0-1 range.
    
    Args:
        depth_map (numpy.ndarray): Depth map to normalize
        
    Returns:
        numpy.ndarray: Normalized depth map
    """
    depth_map = depth_map.astype(np.float32)
    min_val = depth_map.min()
    max_val = depth_map.max()
    
    # Avoid division by zero
    if max_val - min_val < 1e-8:
        return np.zeros_like(depth_map)
    
    # Normalize to 0-1 range
    normalized = (depth_map - min_val) / (max_val - min_val)
    
    return normalized

def generate_depth_map(image_path, use_torch=True):
    """
    Generate a depth map for the given image
    
    Args:
        image_path: Path to the input image
        use_torch: Whether to use PyTorch/DPT or OpenCV
        
    Returns:
        Depth map image
    """
    print("Generating depth map...")
    
    if use_torch and TORCH_AVAILABLE:
        print("Loading depth estimation model...")
        
        # Load model and processor
        model_name = "Intel/dpt-large"
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Device set to use cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Device set to use mps")
        else:
            device = torch.device("cpu")
            print("Device set to use cpu")
            
        model.to(device)
        print("Depth estimation model loaded successfully")
        
        # Load image and process
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_map = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
    else:
        # Fallback to OpenCV
        print("Using OpenCV for depth estimation (less accurate)")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply filters to estimate depth (this is a simple approximation)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        depth_map = cv2.Laplacian(blurred, cv2.CV_8U)
        depth_map = 255 - depth_map  # Invert
    
    return Image.fromarray(depth_map)

def create_shadow_box_effect(image, depth_map, frame_width=0.1, glass_opacity=0.2, light_angle=45):
    """
    Create a shadow box effect that makes the image appear as if it's in a glass display case
    
    Args:
        image: Original RGB image
        depth_map: Corresponding depth map
        frame_width: Width of the frame as a proportion of the image width
        glass_opacity: Opacity of the glass effect (0-1)
        light_angle: Angle of the light source (degrees)
    
    Returns:
        Enhanced image with shadow box effect, enhanced image, visualization
    """
    print("Creating 3D object illusion...")
    
    # Convert to numpy arrays
    img = np.array(image)
    depth = np.array(depth_map)
    
    # Create glass reflection effect
    light_direction = np.deg2rad(light_angle)
    height, width = img.shape[:2]
    
    # Create frame
    print("Creating shadow box frame...")
    frame_size = int(min(width, height) * frame_width)
    framed_img = cv2.copyMakeBorder(img, frame_size, frame_size, frame_size, frame_size, 
                                   cv2.BORDER_CONSTANT, value=[15, 15, 15])
    framed_depth = cv2.copyMakeBorder(depth, frame_size, frame_size, frame_size, frame_size, 
                                     cv2.BORDER_CONSTANT, value=255)
    
    # Enhance depth effect
    enhanced_img = framed_img.copy()
    max_shift = int(min(width, height) * 0.03)
    
    for y in range(frame_size, height + frame_size):
        for x in range(frame_size, width + frame_size):
            depth_value = framed_depth[y, x] / 255.0
            shift_x = int(max_shift * depth_value * np.cos(light_direction))
            shift_y = int(max_shift * depth_value * np.sin(light_direction))
            
            if 0 <= y + shift_y < framed_img.shape[0] and 0 <= x + shift_x < framed_img.shape[1]:
                enhanced_img[y, x] = framed_img[y + shift_y, x + shift_x]
    
    # Add glass effect
    glass = np.ones_like(enhanced_img) * 255
    glass_effect = cv2.addWeighted(enhanced_img, 1 - glass_opacity, glass, glass_opacity, 0)
    
    # Add subtle vignette for realism
    y, x = np.ogrid[:glass_effect.shape[0], :glass_effect.shape[1]]
    center_y, center_x = glass_effect.shape[0] / 2, glass_effect.shape[1] / 2
    mask = ((x - center_x)**2 / (center_x)**2 + (y - center_y)**2 / (center_y)**2) > 0.8
    glass_effect[mask] = glass_effect[mask] * 0.9  # Darken edges
    
    # Create visualization image
    depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    img_resized = cv2.resize(img, (width // 2, height // 2))
    depth_colored_resized = cv2.resize(depth_colored, (width // 2, height // 2))
    
    top_row = np.hstack([img_resized, depth_colored_resized])
    enhanced_resized = cv2.resize(enhanced_img[frame_size:frame_size+height, frame_size:frame_size+width], (width // 2, height // 2))
    glass_effect_resized = cv2.resize(glass_effect[frame_size:frame_size+height, frame_size:frame_size+width], (width // 2, height // 2))
    bottom_row = np.hstack([enhanced_resized, glass_effect_resized])
    
    visualization = np.vstack([top_row, bottom_row])
    
    return Image.fromarray(glass_effect), Image.fromarray(enhanced_img), Image.fromarray(visualization)

def create_screen_pop_effect(image, depth_map, strength=1.5, chromatic=True):
    """
    Create a screen pop effect that makes objects appear to come out of the screen
    
    Args:
        image: Original RGB image
        depth_map: Corresponding depth map
        strength: Strength of the 3D effect
        chromatic: Whether to add chromatic aberration
    
    Returns:
        Enhanced image with screen pop effect
    """
    print("Creating screen pop effect...")
    
    # Convert to numpy arrays
    img = np.array(image)
    depth = np.array(depth_map)
    
    # Normalize depth
    depth_norm = depth.astype(float) / 255.0
    
    # Create offset effect
    height, width = img.shape[:2]
    max_offset = int(min(width, height) * 0.05 * strength)
    
    # Initialize result
    result = np.zeros_like(img)
    
    # Apply offset based on depth
    if chromatic:
        # Create RGB channel separation for chromatic aberration
        channels = cv2.split(img)
        results = []
        
        # Different offsets for different channels creates chromatic aberration
        offsets = [0.7, 1.0, 1.3]  # R, G, B
        
        for i, channel in enumerate(channels):
            channel_result = np.zeros_like(channel)
            channel_offset = max_offset * offsets[i]
            
            for y in range(height):
                for x in range(width):
                    depth_val = depth_norm[y, x]
                    offset = int(depth_val * channel_offset)
                    
                    # Move pixels "outward" based on depth
                    if 0 <= y - offset < height and 0 <= x < width:
                        channel_result[y - offset, x] = channel[y, x]
            
            results.append(channel_result)
        
        # Merge channels
        result = cv2.merge(results)
    else:
        # Standard offset without chromatic aberration
        for y in range(height):
            for x in range(width):
                depth_val = depth_norm[y, x]
                offset = int(depth_val * max_offset)
                
                if 0 <= y - offset < height and 0 <= x < width:
                    result[y - offset, x] = img[y, x]
    
    # Add glow effect to foreground objects
    glow_mask = (depth_norm > 0.5).astype(np.float32)
    glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)
    
    for c in range(3):
        result[:,:,c] = cv2.addWeighted(result[:,:,c], 1, result[:,:,c], 0.3 * glow_mask, 0)
    
    return Image.fromarray(result)

def create_anamorphic_billboard(image, depth_map, strength=1.5, perspective=30, light_intensity=0.8):
    """
    Creates an anamorphic billboard effect similar to 3D LED displays seen in urban advertising.
    
    Args:
        image: Original RGB image
        depth_map: Corresponding depth map
        strength: How pronounced the 3D effect should be
        perspective: Viewing angle perspective (degrees)
        light_intensity: Intensity of virtual lighting
        
    Returns:
        Enhanced image with anamorphic billboard effect
    """
    print("Creating anamorphic billboard effect...")
    
    # Convert inputs to numpy arrays
    img = np.array(image)
    depth = np.array(depth_map)
    
    # Normalize depth map
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Create perspective transform matrix
    height, width = img.shape[:2]
    angle_rad = np.deg2rad(perspective)
    d = np.sqrt(width**2 + height**2)
    
    # Calculate perspective points
    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    shift = int(d * np.sin(angle_rad) * 0.15)
    pts2 = np.float32([[shift, shift], [width-shift, shift], 
                       [width, height], [0, height]])
    
    # Apply perspective transform to both image and depth
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped = cv2.warpPerspective(img, M, (width, height))
    depth_warped = cv2.warpPerspective(depth_norm, M, (width, height))
    
    # Create parallax effect based on depth
    result = img_warped.copy()
    max_shift = int(width * 0.05 * strength)
    
    for y in range(height):
        for x in range(width):
            if x < width-max_shift and y < height-max_shift:
                # Calculate shift based on depth
                depth_val = depth_warped[y, x]
                shift_x = int(depth_val * max_shift)
                shift_y = int(depth_val * max_shift * 0.5)  # Less vertical shift
                
                # Apply shift to create 3D effect
                if 0 <= x+shift_x < width and 0 <= y+shift_y < height:
                    result[y, x] = img_warped[y+shift_y, x+shift_x]
    
    # Add dramatic lighting/glow effect
    light_effect = np.ones_like(result) * 255
    light_mask = 1 - depth_warped**2  # Inverse square falloff
    light_mask = cv2.GaussianBlur(light_mask, (21, 21), 0)
    
    # Convert to proper format for addWeighted
    light_mask_3d = np.zeros_like(result, dtype=np.float32)
    for c in range(3):
        light_mask_3d[:,:,c] = light_mask * light_intensity
    
    # Apply light effect
    result = cv2.addWeighted(result, 1.0, light_effect, 1.0, 0) * light_mask_3d + result * (1 - light_mask_3d)
    result = result.astype(np.uint8)
    
    # Add subtle vignette
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height/2, width/2
    mask = ((x - center_x)**2 / (width/2)**2 + (y - center_y)**2 / (height/2)**2) > 1
    result[mask] = result[mask] * 0.8  # Darken edges
    
    return Image.fromarray(result)

def create_side_by_side_comparison(original, processed):
    """Create side-by-side comparison of original and processed images"""
    width, height = original.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original, (0, 0))
    comparison.paste(processed, (width, 0))
    return comparison

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Colorful Canvas: AI Art Studio")
    parser.add_argument('--effect', type=str, required=True, 
                      choices=['shadow_box', 'screen_pop', 'anamorphic_billboard'],
                      help='Type of effect to generate')
    parser.add_argument('--input_image', type=str, required=True, 
                      help='Path to the input image')
    parser.add_argument('--strength', type=float, default=1.5,
                      help='Strength of the 3D effect (default: 1.5)')
    parser.add_argument('--perspective', type=float, default=30,
                      help='Perspective angle in degrees (for anamorphic effect, default: 30)')
    parser.add_argument('--light', type=float, default=0.8,
                      help='Light intensity (for anamorphic and shadow box, default: 0.8)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory (default: effect_name_effects/)')
    parser.add_argument('--use_opencv', action='store_true', 
                      help='Use OpenCV instead of PyTorch for depth estimation')
    parser.add_argument('--resize', type=int, default=None,
                      help='Resize image to this maximum dimension before processing')
    
    args = parser.parse_args()
    
    print("=== Colorful Canvas: AI Art Studio ===")
    
    # Check if input file exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input file '{args.input_image}' not found")
        sys.exit(1)
    
    # Load the image
    print(f"Loaded image: {args.input_image}")
    image = load_image(args.input_image)
    
    # Resize if requested
    if args.resize:
        image = resize_image(image, args.resize)
        print(f"Resized image to max dimension: {args.resize}px")
    
    # Convert to RGB if needed
    image = image.convert("RGB")
    
    # Generate depth map
    depth_map = generate_depth_map(args.input_image, use_torch=not args.use_opencv)
    
    # Extract the image name without extension
    image_name = os.path.splitext(os.path.basename(args.input_image))[0]
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"{args.effect}_effects"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save depth map
    depth_path = f"{output_dir}/{image_name}_depth.png"
    save_image(depth_map, depth_path)
    print(f"Saved depth map to {depth_path}")
    
    # Process based on selected effect
    start_time = time.time()
    
    if args.effect == "shadow_box":
        # Process image with shadow box effect
        glass_effect, enhanced, visualization = create_shadow_box_effect(
            image, 
            depth_map,
            frame_width=0.1,
            glass_opacity=0.2,
            light_angle=45
        )
        
        # Save enhanced image
        enhanced_path = f"{output_dir}/{image_name}_enhanced.png"
        save_image(enhanced, enhanced_path)
        
        # Save glass effect
        output_path = f"{output_dir}/{image_name}_shadow_box.png"
        save_image(glass_effect, output_path)
        
        # Save visualization
        vis_path = f"{output_dir}/{image_name}_process.png"
        save_image(visualization, vis_path)
        
        print(f"Saved visualization to {vis_path}")
        result_image = glass_effect
        
    elif args.effect == "screen_pop":
        # Process image with screen pop effect
        pop_effect = create_screen_pop_effect(
            image, 
            depth_map,
            strength=args.strength,
            chromatic=True
        )
        
        # Save result
        output_path = f"{output_dir}/{image_name}_screen_pop.png"
        save_image(pop_effect, output_path)
        result_image = pop_effect
        
    elif args.effect == "anamorphic_billboard":
        # Process image with billboard effect
        enhanced_image = create_anamorphic_billboard(
            image, 
            depth_map, 
            strength=args.strength,
            perspective=args.perspective,
            light_intensity=args.light
        )
        
        # Save result
        output_path = f"{output_dir}/{image_name}_billboard.png"
        save_image(enhanced_image, output_path)
        result_image = enhanced_image
    
    processing_time = time.time() - start_time
    
    # Create and save side-by-side comparison
    comparison = create_side_by_side_comparison(image, result_image)
    comparison_path = f"{output_dir}/{image_name}_comparison.png"
    save_image(comparison, comparison_path)
    print(f"Side-by-side comparison saved to {comparison_path}")
    
    print(f"\n=== Effect Generation Complete ===")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Check {output_path} for the final result")

if __name__ == "__main__":
    main() 