import os
import sys
from safetensors import safe_open

def validate_safetensors(file_path):
    """Validate a safetensors file by attempting to open it and check basic metadata."""
    try:
        print(f"Validating {file_path}...")
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Size in GB
        print(f"File size: {file_size:.2f} GB")
        
        # Try to open the file to verify it's a valid safetensors file
        with safe_open(file_path, framework="pt") as f:
            # Get metadata
            metadata = f.metadata()
            if metadata:
                print("Metadata:", metadata)
            
            # Get tensor names and shapes to verify structure
            tensor_info = {}
            for key in f.keys():
                try:
                    shape = f.get_tensor(key).shape
                    tensor_info[key] = shape
                except Exception as e:
                    print(f"Error reading tensor {key}: {e}")
            
            print(f"Number of tensors: {len(tensor_info)}")
            print("First few tensors:")
            for i, (key, shape) in enumerate(list(tensor_info.items())[:5]):
                print(f"  {key}: {shape}")
                
            print("Validation successful!")
            return True
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    model_path = "models/sdxl/sd_xl_base_1.0.safetensors"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    validate_safetensors(model_path) 