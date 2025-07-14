#!/usr/bin/env python3
"""
Verification script to check if the anamorphic billboard is working correctly
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

def analyze_images():
    """Analyze the generated images to verify the anamorphic effect"""
    print("🔍 Analyzing Anamorphic Billboard Results")
    print("=" * 50)
    
    # Check if files exist
    files_to_check = [
        "proper_billboard_output/original.png",
        "proper_billboard_output/depth_map.png", 
        "proper_billboard_output/anamorphic_billboard.png"
    ]
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Load images
    original = Image.open("proper_billboard_output/original.png")
    depth_map = Image.open("proper_billboard_output/depth_map.png")
    billboard = Image.open("proper_billboard_output/anamorphic_billboard.png")
    
    print(f"\n📏 Image Dimensions:")
    print(f"Original: {original.size}")
    print(f"Depth Map: {depth_map.size}")
    print(f"Billboard: {billboard.size}")
    
    # Check if billboard is larger (should be for anamorphic effect)
    if billboard.size[0] > original.size[0] or billboard.size[1] > original.size[1]:
        print("✅ Billboard is larger than original (good for anamorphic effect)")
    else:
        print("⚠️ Billboard is not larger than original")
    
    # Analyze depth map
    depth_array = np.array(depth_map.convert('L'))
    depth_stats = {
        'min': depth_array.min(),
        'max': depth_array.max(),
        'mean': depth_array.mean(),
        'std': depth_array.std()
    }
    
    print(f"\n🎯 Depth Map Analysis:")
    print(f"Min depth: {depth_stats['min']}")
    print(f"Max depth: {depth_stats['max']}")
    print(f"Mean depth: {depth_stats['mean']:.2f}")
    print(f"Std deviation: {depth_stats['std']:.2f}")
    
    # Check if depth map has variation (not flat)
    if depth_stats['std'] > 10:
        print("✅ Depth map has good variation")
    else:
        print("⚠️ Depth map appears flat - may not be working correctly")
    
    # Analyze billboard transformation
    original_array = np.array(original)
    billboard_array = np.array(billboard)
    
    # Check if billboard has content (not mostly black)
    billboard_mean = billboard_array.mean()
    print(f"\n🖼️ Billboard Analysis:")
    print(f"Billboard mean brightness: {billboard_mean:.2f}")
    
    if billboard_mean > 20:
        print("✅ Billboard has content")
    else:
        print("❌ Billboard appears mostly black")
    
    # Check for anamorphic distortion by comparing aspect ratios
    original_aspect = original.size[0] / original.size[1]
    billboard_aspect = billboard.size[0] / billboard.size[1]
    
    print(f"Original aspect ratio: {original_aspect:.2f}")
    print(f"Billboard aspect ratio: {billboard_aspect:.2f}")
    
    if abs(billboard_aspect - original_aspect) > 0.1:
        print("✅ Aspect ratio changed (indicates anamorphic transformation)")
    else:
        print("⚠️ Aspect ratio unchanged - transformation may be minimal")
    
    return True

def test_depth_estimation():
    """Test if depth estimation is actually working"""
    print("\n🧠 Testing Depth Estimation")
    print("=" * 30)
    
    try:
        from core.proper_anamorphic_billboard import ProperAnamorphicBillboard, BillboardConfig
        
        # Create system
        config = BillboardConfig()
        system = ProperAnamorphicBillboard(config)
        
        # Test with a simple image
        test_image = Image.new('RGB', (400, 300), color=(100, 100, 100))
        
        # Add some objects at different depths
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        
        # Bright object (should be close)
        draw.ellipse([50, 50, 150, 150], fill=(255, 255, 255))
        
        # Medium object
        draw.rectangle([200, 100, 300, 200], fill=(150, 150, 150))
        
        # Dark object (should be far)
        draw.polygon([(320, 250), (370, 200), (370, 280)], fill=(50, 50, 50))
        
        test_image.save("test_depth_input.png")
        print("Created test image with objects at different depths")
        
        # Generate depth map
        depth_map = system.generate_depth_map(test_image)
        depth_map.save("test_depth_output.png")
        print("Generated depth map")
        
        # Analyze depth map
        depth_array = np.array(depth_map)
        
        # Check specific regions
        bright_region = depth_array[75:125, 75:125].mean()  # Bright circle
        medium_region = depth_array[125:175, 225:275].mean()  # Medium rectangle
        dark_region = depth_array[225:275, 345:370].mean()   # Dark triangle
        
        print(f"Bright object depth: {bright_region:.2f}")
        print(f"Medium object depth: {medium_region:.2f}")
        print(f"Dark object depth: {dark_region:.2f}")
        
        # Check if depth correlates with brightness (fallback method)
        if bright_region > dark_region:
            print("✅ Depth estimation working (bright objects closer)")
        else:
            print("⚠️ Depth estimation may not be working correctly")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing depth estimation: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 Anamorphic Billboard Verification")
    print("=" * 50)
    
    # Analyze existing results
    if analyze_images():
        print("\n" + "=" * 50)
        
        # Test depth estimation
        test_depth_estimation()
        
        print("\n" + "=" * 50)
        print("📋 Verification Summary:")
        print("1. Check the generated images visually")
        print("2. Verify the billboard shows proper distortion")
        print("3. Confirm depth map has variation")
        print("4. Look for perspective transformation effects")
        
        print("\n💡 To view images:")
        print("open proper_billboard_output/")
        
    else:
        print("❌ Verification failed - missing files")

if __name__ == "__main__":
    main() 