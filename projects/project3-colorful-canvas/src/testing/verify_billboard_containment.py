#!/usr/bin/env python3
"""
Billboard Containment and Anamorphic Quality Verification
Checks that content stays within billboard boundaries and anamorphic effect works correctly
"""

import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
    import numpy as np
    AVAILABLE = True
    print("✅ Image processing available")
except ImportError:
    AVAILABLE = False
    print("❌ Install Pillow and numpy for verification")

def analyze_billboard_containment(image_path: str):
    """
    Analyze if content is properly contained within billboard boundaries
    """
    if not AVAILABLE:
        return False
    
    print(f"🔍 Analyzing billboard containment: {image_path}")
    
    # Load the rendered image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    height, width = img_array.shape[:2]
    print(f"📐 Image dimensions: {width}x{height}")
    
    # Expected billboard area (approximate based on our camera setup)
    # Billboard should be roughly in the center-left of the frame
    billboard_left = int(width * 0.2)    # 20% from left
    billboard_right = int(width * 0.6)   # 60% from left  
    billboard_top = int(height * 0.2)    # 20% from top
    billboard_bottom = int(height * 0.8) # 80% from top
    
    print(f"🎯 Expected billboard area:")
    print(f"   Left: {billboard_left}, Right: {billboard_right}")
    print(f"   Top: {billboard_top}, Bottom: {billboard_bottom}")
    
    # Extract billboard region
    billboard_region = img_array[billboard_top:billboard_bottom, billboard_left:billboard_right]
    
    # Check for content brightness/activity in billboard area
    billboard_brightness = np.mean(billboard_region)
    
    # Check for content spilling outside billboard area
    outside_left = img_array[:, :billboard_left]
    outside_right = img_array[:, billboard_right:]
    outside_top = img_array[:billboard_top, :]
    outside_bottom = img_array[billboard_bottom:, :]
    
    outside_brightness = np.mean([
        np.mean(outside_left) if outside_left.size > 0 else 0,
        np.mean(outside_right) if outside_right.size > 0 else 0,
        np.mean(outside_top) if outside_top.size > 0 else 0,
        np.mean(outside_bottom) if outside_bottom.size > 0 else 0
    ])
    
    print(f"📊 Analysis Results:")
    print(f"   Billboard area brightness: {billboard_brightness:.1f}")
    print(f"   Outside area brightness: {outside_brightness:.1f}")
    
    # Content containment check
    containment_ratio = billboard_brightness / (outside_brightness + 1)  # +1 to avoid division by zero
    
    if containment_ratio > 1.2:  # Billboard should be significantly brighter
        print("✅ GOOD: Content appears contained in billboard area")
        containment_status = "CONTAINED"
    else:
        print("⚠️ WARNING: Content may be spilling outside billboard")
        containment_status = "SPILLOVER"
    
    return {
        'containment_status': containment_status,
        'billboard_brightness': billboard_brightness,
        'outside_brightness': outside_brightness,
        'containment_ratio': containment_ratio,
        'billboard_area': (billboard_left, billboard_top, billboard_right, billboard_bottom)
    }

def check_anamorphic_distortion(image_path: str):
    """
    Check if the anamorphic distortion is working properly
    """
    if not AVAILABLE:
        return False
    
    print(f"🎭 Checking anamorphic distortion: {image_path}")
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Look for perspective distortion characteristics
    height, width = img_array.shape[:2]
    
    # Sample horizontal lines at different heights to check for perspective
    top_line = img_array[height//4, :]      # Upper quarter
    mid_line = img_array[height//2, :]      # Middle
    bottom_line = img_array[3*height//4, :] # Lower quarter
    
    # Calculate variance in brightness across horizontal lines
    # Anamorphic billboards should show perspective distortion
    top_variance = np.var(top_line)
    mid_variance = np.var(mid_line)
    bottom_variance = np.var(bottom_line)
    
    print(f"📈 Perspective Analysis:")
    print(f"   Top line variance: {top_variance:.1f}")
    print(f"   Mid line variance: {mid_variance:.1f}")
    print(f"   Bottom line variance: {bottom_variance:.1f}")
    
    # Check for proper perspective gradient
    if mid_variance > top_variance * 0.5 and mid_variance > bottom_variance * 0.5:
        print("✅ GOOD: Perspective distortion detected")
        distortion_status = "PROPER_ANAMORPHIC"
    else:
        print("⚠️ WARNING: May lack proper anamorphic distortion")
        distortion_status = "WEAK_DISTORTION"
    
    return {
        'distortion_status': distortion_status,
        'variance_gradient': [top_variance, mid_variance, bottom_variance]
    }

def check_billboard_frame_visibility(image_path: str):
    """
    Check if billboard frame is visible and properly positioned
    """
    if not AVAILABLE:
        return False
    
    print(f"🖼️ Checking billboard frame visibility: {image_path}")
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale for edge detection
    gray = np.mean(img_array, axis=2)
    
    # Simple edge detection - look for rectangular frame
    height, width = gray.shape
    
    # Check edges of expected billboard area
    billboard_left = int(width * 0.2)
    billboard_right = int(width * 0.6)
    billboard_top = int(height * 0.2)
    billboard_bottom = int(height * 0.8)
    
    # Sample frame edges
    left_edge = gray[:, billboard_left:billboard_left+5]
    right_edge = gray[:, billboard_right-5:billboard_right]
    top_edge = gray[billboard_top:billboard_top+5, :]
    bottom_edge = gray[billboard_bottom-5:billboard_bottom, :]
    
    # Check for high contrast indicating frame presence
    left_contrast = np.max(left_edge) - np.min(left_edge)
    right_contrast = np.max(right_edge) - np.min(right_edge)
    top_contrast = np.max(top_edge) - np.min(top_edge)
    bottom_contrast = np.max(bottom_edge) - np.min(bottom_edge)
    
    avg_contrast = np.mean([left_contrast, right_contrast, top_contrast, bottom_contrast])
    
    print(f"🔲 Frame Analysis:")
    print(f"   Average edge contrast: {avg_contrast:.1f}")
    
    if avg_contrast > 50:  # Threshold for visible frame
        print("✅ GOOD: Billboard frame is visible")
        frame_status = "FRAME_VISIBLE"
    else:
        print("⚠️ WARNING: Billboard frame may not be clearly visible")
        frame_status = "FRAME_WEAK"
    
    return {
        'frame_status': frame_status,
        'average_contrast': avg_contrast,
        'edge_contrasts': [left_contrast, right_contrast, top_contrast, bottom_contrast]
    }

def create_verification_overlay(image_path: str, output_path: str, analysis_results: dict):
    """
    Create an overlay showing the analysis results
    """
    if not AVAILABLE:
        return
    
    print(f"🎨 Creating verification overlay: {output_path}")
    
    img = Image.open(image_path)
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    
    # Draw billboard boundary
    bbox = analysis_results['containment']['billboard_area']
    left, top, right, bottom = bbox
    
    # Draw billboard frame in green if contained, red if spillover
    color = "green" if analysis_results['containment']['containment_status'] == "CONTAINED" else "red"
    draw.rectangle([left, top, right, bottom], outline=color, width=3)
    
    # Add text annotations
    draw.text((10, 10), f"Containment: {analysis_results['containment']['containment_status']}", 
              fill=color, stroke_width=1, stroke_fill="black")
    draw.text((10, 30), f"Distortion: {analysis_results['distortion']['distortion_status']}", 
              fill="blue", stroke_width=1, stroke_fill="black")
    draw.text((10, 50), f"Frame: {analysis_results['frame']['frame_status']}", 
              fill="purple", stroke_width=1, stroke_fill="black")
    
    overlay.save(output_path)
    print(f"✅ Verification overlay saved: {output_path}")

def main():
    """
    Run full billboard verification
    """
    print("🚀 Billboard Containment & Anamorphic Quality Verification")
    print("=" * 60)
    
    # Check available output files
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ No output directory found")
        return
    
    # Find the most recent billboard render
    png_files = list(output_dir.glob("*.png"))
    if not png_files:
        print("❌ No PNG files found in output directory")
        return
    
    # Use the most recent file
    latest_file = max(png_files, key=lambda p: p.stat().st_mtime)
    print(f"📷 Analyzing: {latest_file}")
    
    # Run all verification checks
    containment_results = analyze_billboard_containment(str(latest_file))
    distortion_results = check_anamorphic_distortion(str(latest_file))
    frame_results = check_billboard_frame_visibility(str(latest_file))
    
    # Combine results
    analysis_results = {
        'containment': containment_results,
        'distortion': distortion_results,
        'frame': frame_results
    }
    
    # Create verification overlay
    overlay_path = output_dir / f"verification_{latest_file.name}"
    create_verification_overlay(str(latest_file), str(overlay_path), analysis_results)
    
    # Final assessment
    print("\n" + "=" * 60)
    print("📋 FINAL ASSESSMENT:")
    
    issues = []
    if containment_results and containment_results['containment_status'] != "CONTAINED":
        issues.append("Content spillover detected")
    if distortion_results and distortion_results['distortion_status'] != "PROPER_ANAMORPHIC":
        issues.append("Weak anamorphic distortion")
    if frame_results and frame_results['frame_status'] != "FRAME_VISIBLE":
        issues.append("Billboard frame not clearly visible")
    
    if not issues:
        print("✅ EXCELLENT: Billboard is properly contained with good anamorphic effect!")
    else:
        print("⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
    
    print(f"\n📊 Verification overlay created: {overlay_path}")

if __name__ == "__main__":
    main() 