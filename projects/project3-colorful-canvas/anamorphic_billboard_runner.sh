#!/bin/bash
# Anamorphic Billboard Direct Runner Script
# Runs the anamorphic billboard script directly in Blender
# Author: Komal Shahid
# Course: DSC680 - Bellevue University
# Date: June 1, 2025

# Path to Blender on macOS
BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
IMAGE_PATH="$SCRIPT_DIR/data/sample_images/sample2.jpg"
EFFECT_TYPE="shadow_box"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --effect)
      EFFECT_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--image /path/to/image.jpg] [--effect shadow_box|seoul_corner|screen_pop]"
      exit 1
      ;;
  esac
done

# Check if Blender exists
if [ ! -f "$BLENDER_PATH" ]; then
    echo "❌ Blender not found at $BLENDER_PATH"
    echo "Please edit this script with the correct path to your Blender installation"
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found at $IMAGE_PATH"
    exit 1
fi

# Generate unique filename based on image name and effect type
IMAGE_BASENAME=$(basename "${IMAGE_PATH%.*}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/data/output/blender_anamorphic"
mkdir -p "$OUTPUT_DIR"

# Create a Blender Python script
BLENDER_SCRIPT="$SCRIPT_DIR/temp_anamorphic_script.py"

cat > "$BLENDER_SCRIPT" <<EOF
import bpy
import sys
import os
from pathlib import Path
import datetime

# Add project directory to Python path
project_dir = "$SCRIPT_DIR"
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Load the main script
with open("$SCRIPT_DIR/src/blender/anamorphic_billboard_consolidated.py", 'r') as file:
    script_code = file.read()

# Execute the script
exec(script_code)

# Set up unique output paths with timestamp
image_name = os.path.basename("$IMAGE_PATH").split('.')[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_base = f"$OUTPUT_DIR/{image_name}_${EFFECT_TYPE}_{timestamp}"

# Run the appropriate effect function
image_path = "$IMAGE_PATH"
effect_type = "$EFFECT_TYPE"

# Call the main function directly with our parameters
if effect_type == "seoul_corner":
    output_path = output_base + "_seoul.png"
    main(image_path, output_path, "seoul_corner", 2.0)
elif effect_type == "screen_pop":
    output_path = output_base + "_screen.png"
    main(image_path, output_path, "screen_pop", 1.8)
else:  # shadow_box
    output_path = output_base + "_shadow.png"
    main(image_path, output_path, "shadow_box", 1.5)

print(f"✅ Anamorphic billboard render complete")
print(f"📁 Output files saved with prefix: {output_base}")
EOF

# Run Blender with our script
echo "🎬 Running Blender anamorphic billboard with $EFFECT_TYPE effect on $(basename "$IMAGE_PATH")..."
"$BLENDER_PATH" --background --python "$BLENDER_SCRIPT"

# Clean up
echo "✅ Blender execution complete!"
echo "📝 Check output directory: $OUTPUT_DIR" 