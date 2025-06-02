#!/bin/bash
# Blender runner script with parameters
# Runs Blender with our anamorphic billboard script directly

# Path to Blender on macOS
BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default values
IMAGE_PATH="$SCRIPT_DIR/data/sample_images/sample1.jpg"
EFFECT_TYPE="shadow_box"
OUTPUT_DIR="$SCRIPT_DIR/data/output/blender_anamorphic"

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
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--image /path/to/image.jpg] [--effect shadow_box|seoul_corner|screen_pop] [--output /output/directory]"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate a unique base name for output files based on input image and effect
IMAGE_BASENAME=$(basename "${IMAGE_PATH%.*}")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/${IMAGE_BASENAME}_${EFFECT_TYPE}_${TIMESTAMP}_"

# Check if Blender exists
if [ ! -f "$BLENDER_PATH" ]; then
    echo "❌ Blender not found at $BLENDER_PATH"
    echo "Please edit this script with the correct path to your Blender installation"
    exit 1
fi

# Create a temporary Python script to run in Blender
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" <<EOF
import bpy
import sys
import os
from pathlib import Path

# Add project directory to Python path
project_dir = "$SCRIPT_DIR"
if project_dir not in sys.path:
    sys.path.append(project_dir)

# Load the main script
script_path = "$SCRIPT_DIR/anamorphic_billboard_consolidated.py"
with open(script_path, 'r') as file:
    script_code = file.read()

# Execute the script
exec(script_code)

# Run the specified effect
image_path = "$IMAGE_PATH"
effect_type = "$EFFECT_TYPE"

if effect_type == "seoul_corner":
    create_seoul_style_billboard(image_path)
elif effect_type == "screen_pop":
    create_screen_popup_billboard(image_path)
else:  # shadow_box
    create_shadow_box_billboard(image_path)
EOF

# Run Blender with our temporary script
echo "🎬 Running Blender with $EFFECT_TYPE effect on $(basename "$IMAGE_PATH")..."
"$BLENDER_PATH" --background --python "$TMP_SCRIPT" --render-output "$OUTPUT_FILE"

# Clean up
rm "$TMP_SCRIPT"

echo "✅ Blender execution complete!"
echo "📁 Output files saved with prefix: $(basename "${OUTPUT_FILE}")" 