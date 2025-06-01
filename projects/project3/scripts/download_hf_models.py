import os
import sys
from huggingface_hub import login, snapshot_download

print("Hugging Face Model Downloader")
print("-----------------------------")

# Get token interactively if not provided as argument
if len(sys.argv) > 1:
    token = sys.argv[1]
    print("Using token provided as argument")
else:
    token = input("Enter your Hugging Face token: ")
    print("Token received")

# Login to Hugging Face
print("Logging in to Hugging Face...")
login(token)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)

# Define models to download - latest trending safe models
models = [
    {
        "name": "SDXL 1.0",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "dir": "models/sdxl"
    },
    {
        "name": "FLUX ControlNet",
        "repo": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
        "dir": "models/flux-controlnet"
    },
    {
        "name": "RunDiffusion XL",
        "repo": "rundiffusion/RunDiffusion-XL",
        "dir": "models/rundiffusion-xl"
    }
]

# Download each model
for model in models:
    print(f"\nDownloading {model['name']}...")
    print(f"Repository: {model['repo']}")
    try:
        path = snapshot_download(
            repo_id=model['repo'],
            local_dir=model['dir'],
            ignore_patterns=["*.bin", "*.onnx"],  # Prefer safetensors when available
            resume_download=True
        )
        print(f"✅ Successfully downloaded to: {path}")
    except Exception as e:
        print(f"❌ Error downloading {model['name']}: {str(e)}")

print("\nDownload process completed!") 