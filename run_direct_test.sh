#!/bin/bash
# Direct test script for running Blender with our anamorphic billboard script

# Path to Blender on macOS
BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create a simple test Python script for Blender to run
TMP_SCRIPT="$SCRIPT_DIR/temp_test_script.py"

cat > "$TMP_SCRIPT" <<EOF
import bpy
import sys
import os
from pathlib import Path
import datetime

# Add project directory to Python path
project_dir = "$SCRIPT_DIR"
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Clear existing scene first
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)

# Create a simple cube
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
cube = bpy.context.active_object

# Add a camera
bpy.ops.object.camera_add(location=(0, -10, 0))
camera = bpy.context.active_object
camera.rotation_euler = (1.5707, 0, 0)  # Point at cube

# Add a light
bpy.ops.object.light_add(type='POINT', location=(0, 0, 5))

# Set camera as active
bpy.context.scene.camera = camera

# Set up output path with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = "$SCRIPT_DIR/data/output/blender_anamorphic/test_render_" + timestamp + ".png"

# Set render settings
bpy.context.scene.render.filepath = output_path
bpy.context.scene.render.resolution_x = 800
bpy.context.scene.render.resolution_y = 600

# Render
bpy.ops.render.render(write_still=True)

print(f"✅ Test render saved to: {output_path}")
EOF

# Check if Blender exists
if [ ! -f "$BLENDER_PATH" ]; then
    echo "❌ Blender not found at $BLENDER_PATH"
    echo "Please edit this script with the correct path to your Blender installation"
    exit 1
fi

# Run Blender with our temporary script
echo "🎬 Running Blender with test script..."
"$BLENDER_PATH" --background --python "$TMP_SCRIPT"

# Don't remove the script so we can inspect it
echo "✅ Blender execution complete!"
echo "📝 Test script is at: $TMP_SCRIPT" 