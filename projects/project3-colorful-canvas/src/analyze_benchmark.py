#!/usr/bin/env python3
"""
Comprehensive analysis of benchmark image and anamorphic billboard output
to ensure all 3D elements, textures, and sizing are correct
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import os

def analyze_benchmark_content():
    """Analyze the benchmark image to identify key elements"""
    print("🔍 Analyzing Benchmark Image Content")
    print("=" * 50)
    
    # Load benchmark image
    benchmark = Image.open("../data/input/benchmark.jpg")
    benchmark_array = np.array(benchmark)
    
    print(f"📏 Benchmark Dimensions: {benchmark.size}")
    print(f"🎨 Color Mode: {benchmark.mode}")
    print(f"📊 Value Range: {benchmark_array.min()} - {benchmark_array.max()}")
    print(f"🌈 Mean RGB: {benchmark_array.mean(axis=(0,1))}")
    
    # Analyze color distribution to identify objects
    print("\n🎯 Color Analysis:")
    
    # Convert to HSV for better object detection
    hsv = cv2.cvtColor(benchmark_array, cv2.COLOR_RGB2HSV)
    
    # Detect bright/colorful objects (likely the furry characters)
    bright_mask = benchmark_array.mean(axis=2) > 100
    bright_pixels = np.sum(bright_mask)
    print(f"Bright areas (characters): {bright_pixels} pixels ({bright_pixels/benchmark_array.size*100:.1f}%)")
    
    # Detect architectural elements (likely darker, more structured)
    gray = cv2.cvtColor(benchmark_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    print(f"Edge pixels (architecture): {edge_pixels} pixels ({edge_pixels/gray.size*100:.1f}%)")
    
    # Detect different color regions
    colors = {
        'Red': np.sum((benchmark_array[:,:,0] > 150) & (benchmark_array[:,:,1] < 100)),
        'Green': np.sum((benchmark_array[:,:,1] > 150) & (benchmark_array[:,:,0] < 100)),
        'Blue': np.sum((benchmark_array[:,:,2] > 150) & (benchmark_array[:,:,0] < 100)),
        'Yellow': np.sum((benchmark_array[:,:,0] > 150) & (benchmark_array[:,:,1] > 150) & (benchmark_array[:,:,2] < 100)),
        'Purple': np.sum((benchmark_array[:,:,0] > 100) & (benchmark_array[:,:,2] > 150) & (benchmark_array[:,:,1] < 100))
    }
    
    print("\n🌈 Color Distribution:")
    for color, pixels in colors.items():
        percentage = pixels / benchmark_array.size * 100
        print(f"{color}: {pixels} pixels ({percentage:.1f}%)")
    
    return benchmark, benchmark_array

def analyze_billboard_output():
    """Analyze the anamorphic billboard output"""
    print("\n🖼️ Analyzing Billboard Output")
    print("=" * 50)
    
    # Load billboard output
    billboard = Image.open("proper_billboard_output/anamorphic_billboard.png")
    billboard_array = np.array(billboard)
    
    print(f"📏 Billboard Dimensions: {billboard.size}")
    print(f"🎨 Color Mode: {billboard.mode}")
    print(f"📊 Value Range: {billboard_array.min()} - {billboard_array.max()}")
    print(f"🌈 Mean RGB: {billboard_array.mean(axis=(0,1))}")
    
    # Check if content is properly distributed
    non_black_pixels = np.sum(billboard_array.sum(axis=2) > 30)
    total_pixels = billboard_array.shape[0] * billboard_array.shape[1]
    content_percentage = non_black_pixels / total_pixels * 100
    
    print(f"📊 Content Coverage: {content_percentage:.1f}% of billboard has content")
    
    if content_percentage < 20:
        print("⚠️ Low content coverage - may indicate transformation issues")
    elif content_percentage > 60:
        print("✅ Good content coverage")
    else:
        print("🔶 Moderate content coverage")
    
    return billboard, billboard_array

def compare_content_preservation():
    """Compare if billboard preserves the original content"""
    print("\n🔄 Content Preservation Analysis")
    print("=" * 50)
    
    # Load images
    original = np.array(Image.open("proper_billboard_output/original.png"))
    billboard = np.array(Image.open("proper_billboard_output/anamorphic_billboard.png"))
    
    # Analyze color preservation
    orig_colors = original.mean(axis=(0,1))
    bill_colors = billboard.mean(axis=(0,1))
    
    print(f"Original mean RGB: {orig_colors}")
    print(f"Billboard mean RGB: {bill_colors}")
    
    color_diff = np.abs(orig_colors - bill_colors)
    print(f"Color difference: {color_diff}")
    
    if np.all(color_diff < 50):
        print("✅ Colors well preserved")
    else:
        print("⚠️ Significant color changes detected")
    
    # Check if bright elements are preserved
    orig_bright = np.sum(original.mean(axis=2) > 100)
    bill_bright = np.sum(billboard.mean(axis=2) > 100)
    
    print(f"Original bright pixels: {orig_bright}")
    print(f"Billboard bright pixels: {bill_bright}")
    
    if bill_bright > orig_bright * 0.5:
        print("✅ Bright elements preserved")
    else:
        print("⚠️ Loss of bright elements")

def check_3d_elements():
    """Check if 3D elements are properly represented"""
    print("\n🎲 3D Elements Analysis")
    print("=" * 50)
    
    # Load depth map
    depth_map = np.array(Image.open("proper_billboard_output/depth_map.png"))
    
    # Analyze depth distribution
    depth_stats = {
        'min': depth_map.min(),
        'max': depth_map.max(),
        'mean': depth_map.mean(),
        'std': depth_map.std()
    }
    
    print(f"Depth range: {depth_stats['min']} - {depth_stats['max']}")
    print(f"Depth mean: {depth_stats['mean']:.2f}")
    print(f"Depth variation: {depth_stats['std']:.2f}")
    
    # Check for proper depth layers
    near_objects = np.sum(depth_map > 200)  # Very close objects
    mid_objects = np.sum((depth_map > 100) & (depth_map <= 200))  # Mid-distance
    far_objects = np.sum(depth_map <= 100)  # Far objects
    
    total_pixels = depth_map.size
    print(f"Near objects: {near_objects/total_pixels*100:.1f}%")
    print(f"Mid objects: {mid_objects/total_pixels*100:.1f}%")
    print(f"Far objects: {far_objects/total_pixels*100:.1f}%")
    
    if depth_stats['std'] > 50:
        print("✅ Good depth variation for 3D effect")
    else:
        print("⚠️ Limited depth variation")

def check_deprecated_modules():
    """Check what deprecated modules we're using and suggest alternatives"""
    print("\n⚠️ Deprecated Module Analysis")
    print("=" * 50)
    
    deprecated_issues = []
    
    # Check timm deprecation
    try:
        import timm.models.layers
        deprecated_issues.append("timm.models.layers - should use timm.layers")
    except:
        pass
    
    # Check torch hub usage
    print("🔍 Current dependencies:")
    print("- MiDaS: Using torch.hub (stable)")
    print("- OpenCV: Using cv2 (stable)")
    print("- PIL: Using Pillow (stable)")
    print("- NumPy: Using numpy (stable)")
    
    if deprecated_issues:
        print("\n⚠️ Deprecated modules found:")
        for issue in deprecated_issues:
            print(f"  - {issue}")
        print("\n💡 These don't affect functionality but should be updated")
    else:
        print("✅ No critical deprecated modules")

def create_visual_comparison():
    """Create a visual comparison of original vs billboard"""
    print("\n📸 Creating Visual Comparison")
    print("=" * 50)
    
    try:
        # Load images
        original = Image.open("proper_billboard_output/original.png")
        billboard = Image.open("proper_billboard_output/anamorphic_billboard.png")
        depth_map = Image.open("proper_billboard_output/depth_map.png")
        
        # Resize for comparison
        target_height = 400
        orig_resized = original.resize((int(original.width * target_height / original.height), target_height))
        bill_resized = billboard.resize((int(billboard.width * target_height / billboard.height), target_height))
        depth_resized = depth_map.resize((int(depth_map.width * target_height / depth_map.height), target_height))
        
        # Create comparison image
        total_width = orig_resized.width + bill_resized.width + depth_resized.width + 40
        comparison = Image.new('RGB', (total_width, target_height + 60), color=(50, 50, 50))
        
        # Paste images
        comparison.paste(orig_resized, (10, 50))
        comparison.paste(bill_resized, (orig_resized.width + 20, 50))
        comparison.paste(depth_resized, (orig_resized.width + bill_resized.width + 30, 50))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Original", fill=(255, 255, 255), font=font)
        draw.text((orig_resized.width + 20, 10), "Anamorphic Billboard", fill=(255, 255, 255), font=font)
        draw.text((orig_resized.width + bill_resized.width + 30, 10), "Depth Map", fill=(255, 255, 255), font=font)
        
        comparison.save("visual_comparison.png")
        print("✅ Saved visual_comparison.png")
        
    except Exception as e:
        print(f"❌ Error creating visual comparison: {e}")

def main():
    """Main analysis function"""
    print("🔍 Comprehensive Anamorphic Billboard Analysis")
    print("=" * 60)
    
    # Analyze benchmark content
    benchmark, benchmark_array = analyze_benchmark_content()
    
    # Analyze billboard output
    billboard, billboard_array = analyze_billboard_output()
    
    # Compare content preservation
    compare_content_preservation()
    
    # Check 3D elements
    check_3d_elements()
    
    # Check deprecated modules
    check_deprecated_modules()
    
    # Create visual comparison
    create_visual_comparison()
    
    print("\n" + "=" * 60)
    print("📋 Analysis Summary:")
    print("1. Check visual_comparison.png for side-by-side view")
    print("2. Verify all colorful characters are visible in billboard")
    print("3. Confirm architectural elements show proper depth")
    print("4. Ensure textures and details are preserved")
    print("5. Check that 3D effect is convincing from viewing angle")

if __name__ == "__main__":
    main() 