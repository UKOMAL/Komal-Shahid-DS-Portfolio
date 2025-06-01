import os
import sys
from huggingface_hub import login, snapshot_download

# Check if token was provided as command line argument
if len(sys.argv) < 2:
    print("Please provide your Hugging Face token as an argument:")
    print("python3 download_models.py YOUR_TOKEN_HERE")
    sys.exit(1)

# Get token from command line arguments
token = sys.argv[1]

# Login to Hugging Face
print(f"Logging in to Hugging Face...")
login(token)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)

# Define models to download - using the latest trending and safe models
models = [
    {
        "name": "FLUX.1-dev-ControlNet-Union-Pro",
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
        "dir": "models/flux-controlnet-union-pro",
        "description": "Trending unified ControlNet with 7 control modes (canny, tile, depth, blur, pose, gray, low quality)"
    },
    {
        "name": "Stable Diffusion Safe",
        "repo": "AIML-TUDA/stable-diffusion-safe",
        "dir": "models/stable-diffusion-safe",
        "description": "Safer version of Stable Diffusion that suppresses inappropriate content"
    },
    {
        "name": "Stable Diffusion XL Base",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "dir": "models/stable-diffusion-xl",
        "description": "High-quality base model for the anamorphic engine"
    },
    # Optional additional model
    {
        "name": "Stable Diffusion 3.5 ControlNet",
        "repo": "stabilityai/stable-diffusion-3.5-controlnets",
        "dir": "models/stable-diffusion-3.5-controlnet",
        "description": "Latest ControlNet from StabilityAI (only if the user has access)"
    }
]

# Download each model
for model in models:
    print(f"\n===== Downloading {model['name']} =====")
    print(f"Description: {model['description']}")
    try:
        # Create model directory
        os.makedirs(model['dir'], exist_ok=True)
        
        # Attempt to download the model
        print(f"Downloading {model['repo']} to {model['dir']}...")
        path = snapshot_download(
            model['repo'],
            local_dir=model['dir'],
            local_dir_use_symlinks=False,
            # Prefer safetensors format when available
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt", "*.md", "*.yaml", "*.yml"],
        )
        print(f"✅ {model['name']} downloaded to: {path}")
    except Exception as e:
        print(f"❌ Error downloading {model['name']}: {e}")
        print("You may need to manually accept the model's terms of use at:")
        print(f"https://huggingface.co/{model['repo']}")

print("\n===== Download Summary =====")
print("The downloaded models can be used with the anamorphic_engine.py by updating the paths:")
print("sd_model_path = 'models/stable-diffusion-xl'  # or 'models/stable-diffusion-safe'")
print("control_net_path = 'models/flux-controlnet-union-pro'  # For advanced control")
print("\nFor the Anamorphic Illusion Engine, these models provide:")
print("1. High-quality image generation (Stable Diffusion XL or SD Safe)")
print("2. Advanced control over perspectives and transformations (FLUX ControlNet)")
print("3. Safer content generation with appropriate filtering")
print("\nDone!") 