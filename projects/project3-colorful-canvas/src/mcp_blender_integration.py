"""
MCP Integration Module for Blender Anamorphic Billboard Generation
This module integrates the Colorful Canvas AI with Blender through MCP
"""

import os
import sys
import subprocess
from pathlib import Path

# Get project directory
PROJECT_DIR = Path(__file__).parent.parent.absolute()

class MCPBlenderIntegration:
    """Class for integrating MCP with Blender anamorphic billboard generation"""
    
    def __init__(self):
        """Initialize the integration with default values"""
        self.blender_script_path = PROJECT_DIR / "anamorphic_billboard_consolidated.py"
        self.wrapper_script_path = PROJECT_DIR / "run_blender_anamorphic.py"
        self.image_dir = PROJECT_DIR / "data/sample_images"
        self.output_dir = PROJECT_DIR / "data/output/blender_anamorphic"
        
        # Create directories if they don't exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for required files
        if not self.blender_script_path.exists():
            print(f"⚠️ Warning: Blender script not found at {self.blender_script_path}")
        
        if not self.wrapper_script_path.exists():
            print(f"⚠️ Warning: Wrapper script not found at {self.wrapper_script_path}")
    
    def generate_anamorphic_billboard(self, image_path, effect_type="shadow_box", output_name=None):
        """
        Generate an anamorphic billboard effect using Blender
        
        Args:
            image_path (str): Path to the input image
            effect_type (str): Type of effect ('shadow_box', 'seoul_corner', or 'screen_pop')
            output_name (str, optional): Name for the output file (without extension)
            
        Returns:
            str: Path to the output render or None if generation failed
        """
        # Validate image path
        if not os.path.exists(image_path):
            print(f"❌ Error: Input image not found at {image_path}")
            return None
        
        # Generate output filename
        if output_name is None:
            input_filename = os.path.basename(image_path)
            base_name = os.path.splitext(input_filename)[0]
            output_name = f"{base_name}_{effect_type}"
        
        output_path = self.output_dir / f"{output_name}.png"
        
        # Run the wrapper script
        cmd = [
            sys.executable,  # Current Python interpreter
            str(self.wrapper_script_path),
            "--script", str(self.blender_script_path),
            "--image", str(image_path),
            "--output", str(output_path),
            "--effect", effect_type
        ]
        
        print(f"🎬 Running Blender anamorphic generator with {effect_type} effect...")
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            # Get return code
            return_code = process.poll()
            
            # Check for errors
            if return_code != 0:
                stderr = process.stderr.read()
                print(f"❌ Error in Blender execution (code {return_code}):")
                print(stderr)
                return None
            
            if output_path.exists():
                print(f"✅ Anamorphic billboard generated: {output_path}")
                return str(output_path)
            else:
                print(f"❌ Output file not created: {output_path}")
                return None
                
        except Exception as e:
            print(f"❌ Failed to run Blender: {e}")
            return None
    
    def process_image_batch(self, image_list, effect_type="shadow_box"):
        """
        Process a batch of images with the same effect
        
        Args:
            image_list (list): List of image paths to process
            effect_type (str): Type of effect to apply to all images
            
        Returns:
            list: List of output paths for successful renders
        """
        results = []
        for image_path in image_list:
            print(f"Processing image: {os.path.basename(image_path)}")
            result = self.generate_anamorphic_billboard(image_path, effect_type)
            if result:
                results.append(result)
            print("-" * 40)
        
        print(f"✅ Batch processing complete. {len(results)}/{len(image_list)} images processed successfully.")
        return results
    
    def run_demo(self):
        """Run a demonstration of all effects on a sample image"""
        # Find sample images
        sample_images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        
        if not sample_images:
            print("❌ No sample images found in", self.image_dir)
            return
        
        # Use the first sample image
        sample_image = sample_images[0]
        print(f"🖼️ Using sample image: {sample_image}")
        
        # Run all effects
        effects = ["shadow_box", "seoul_corner", "screen_pop"]
        results = []
        
        for effect in effects:
            print(f"\n🎬 Generating {effect.replace('_', ' ')} effect...")
            output = self.generate_anamorphic_billboard(
                str(sample_image),
                effect_type=effect,
                output_name=f"demo_{effect}"
            )
            if output:
                results.append((effect, output))
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 DEMO RESULTS SUMMARY")
        print("=" * 50)
        for effect, path in results:
            print(f"- {effect.replace('_', ' ')}: {path}")
        print("=" * 50)

# Example usage
if __name__ == "__main__":
    mcp = MCPBlenderIntegration()
    mcp.run_demo() 