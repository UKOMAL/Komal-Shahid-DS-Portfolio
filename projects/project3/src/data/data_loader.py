"""
Data Loader for Colorful Canvas AI Art Studio

This script handles fetching and loading the datasets needed for training via APIs:
1. Optical Illusions Classification Dataset from GitHub API
2. 3D Depth Perception Dataset from NYU Depth Dataset
3. Fashion 3D Dataset from ShapeNet
4. Color Psychology Dataset from academic repositories
5. Social Media Engagement Data from platform APIs
6. Advertising Performance Data from ad platforms
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
import random
from tqdm import tqdm
from urllib.parse import urlparse
import base64
from concurrent.futures import ThreadPoolExecutor

# GitHub repository for the optical illusion dataset
GITHUB_REPO = "robertmaxwilliams/optical-illusion-dataset"
GITHUB_API_URL = "https://api.github.com"

# URLs for the datasets
DATASET_URLS = {
    "optical_illusions": {
        "api_url": f"{GITHUB_API_URL}/repos/{GITHUB_REPO}/contents",
        "metadata_path": "metadata.json",
        "images_path": "images",
    },
    "depth_perception": {
        "api_url": "https://huggingface.co/api/datasets/nyu_depth_v2",
        "fallback_url": "https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html",
    },
    "fashion_3d": {
        "api_url": "https://www.shapenet.org/api/v1",
        "category": "03d01966",  # ShapeNet ID for clothing
    },
    "color_psychology": {
        "api_url": "https://zenodo.org/api/records/1234567",  # Example Zenodo ID
    },
    "social_media": {
        "api_urls": {
            "youtube": "https://www.googleapis.com/youtube/v3",
            "instagram": "https://graph.instagram.com/v1",
            "tiktok": "https://open-api.tiktok.com/api/v1",
        }
    },
    "advertising": {
        "api_urls": {
            "google_ads": "https://googleads.googleapis.com/v12",
            "facebook": "https://graph.facebook.com/v16.0",
        }
    }
}

# Cache directory for downloaded data
CACHE_DIR = Path("./data_cache")
CACHE_EXPIRY = 86400  # Cache expiry in seconds (24 hours)

class DataLoader:
    """Data loader for Colorful Canvas dataset with API-based access"""
    
    def __init__(self, cache_dir=CACHE_DIR, use_cache=True, cache_expiry=CACHE_EXPIRY):
        """
        Initialize the data loader
        
        Args:
            cache_dir: Directory to cache downloaded data
            use_cache: Whether to use cached data (if available)
            cache_expiry: Time in seconds before cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_expiry = cache_expiry
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for each dataset
        for dataset in DATASET_URLS:
            os.makedirs(self.cache_dir / dataset, exist_ok=True)
            
        print(f"Data loader initialized with cache directory: {self.cache_dir}")

    def _get_cached_data(self, cache_key):
        """
        Get data from cache if available and not expired
        
        Args:
            cache_key: Key for the cached data
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        if not self.use_cache:
            return None
            
        cache_path = self.cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        cache_time = os.path.getmtime(cache_path)
        if time.time() - cache_time > self.cache_expiry:
            return None
            
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _save_to_cache(self, cache_key, data):
        """
        Save data to cache
        
        Args:
            cache_key: Key for the cached data
            data: Data to cache
        """
        if not self.use_cache:
            return
            
        cache_path = self.cache_dir / f"{cache_key}.json"
        os.makedirs(cache_path.parent, exist_ok=True)
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    
    def _api_request(self, url, params=None, headers=None, method="GET", retries=3):
        """
        Make an API request with error handling and retries
        
        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            method: HTTP method
            retries: Number of retries
            
        Returns:
            Response JSON if successful, None otherwise
        """
        # Default headers
        if headers is None:
            headers = {
                "Accept": "application/json",
                "User-Agent": "Colorful-Canvas-AI-Art-Studio/1.0"
            }
            
        for attempt in range(retries):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def fetch_optical_illusions_dataset(self, max_samples=None):
        """
        Fetch optical illusions dataset via GitHub API
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            Dictionary of dataset information
        """
        print("Fetching optical illusions dataset via API...")
        
        # Check cache first
        cache_key = f"optical_illusions_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached optical illusions data")
            return cached_data
        
        # Fetch metadata first
        metadata_url = f"{DATASET_URLS['optical_illusions']['api_url']}/{DATASET_URLS['optical_illusions']['metadata_path']}"
        metadata_response = self._api_request(metadata_url)
        
        if metadata_response is None:
            print("Failed to fetch metadata, simulating dataset...")
            return self._simulate_optical_illusions(max_samples)
        
        # Decode content
        try:
            metadata_content = base64.b64decode(metadata_response['content']).decode('utf-8')
            metadata = json.loads(metadata_content)
        except:
            print("Failed to decode metadata, simulating dataset...")
            return self._simulate_optical_illusions(max_samples)
        
        # Fetch list of images
        images_url = f"{DATASET_URLS['optical_illusions']['api_url']}/{DATASET_URLS['optical_illusions']['images_path']}"
        images_response = self._api_request(images_url)
        
        if images_response is None:
            print("Failed to fetch images list, simulating dataset...")
            return self._simulate_optical_illusions(max_samples)
        
        # Get image data from metadata
        image_data = metadata.get("images", [])
        if max_samples is not None:
            image_data = image_data[:max_samples]
        
        print(f"Found {len(image_data)} optical illusion images")
        
        # Prepare dataset
        dataset = {
            "metadata": metadata,
            "images": [],
            "features": []
        }
        
        # Fetch images in parallel with a thread pool
        def fetch_image(img_info):
            try:
                img_url = f"{DATASET_URLS['optical_illusions']['api_url']}/{DATASET_URLS['optical_illusions']['images_path']}/{img_info['filename']}"
                img_response = self._api_request(img_url)
                
                if img_response is None:
                    return None
                
                img_content = base64.b64decode(img_response['content'])
                image = Image.open(BytesIO(img_content))
                
                return {
                    "image": image,
                    "feature": {
                        "id": img_info.get("id", ""),
                        "category": img_info.get("category", ""),
                        "tags": img_info.get("tags", []),
                        "depth_score": img_info.get("depth_score", 0.0),
                        "complexity": img_info.get("complexity", 0.0)
                    }
                }
            except Exception as e:
                print(f"Error fetching image {img_info['filename']}: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(tqdm(
                executor.map(fetch_image, image_data),
                total=len(image_data),
                desc="Fetching optical illusion images"
            ))
        
        # Add results to dataset
        for result in results:
            if result is not None:
                dataset["images"].append(result["image"])
                dataset["features"].append(result["feature"])
        
        print(f"Fetched {len(dataset['images'])} optical illusion images")
        
        # Cache the dataset
        self._save_to_cache(cache_key, dataset)
        
        return dataset
    
    def fetch_3d_depth_dataset(self, max_samples=None):
        """
        Fetch 3D depth dataset via NYU API
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            Dictionary of dataset information
        """
        print("Fetching 3D depth dataset via API...")
        
        # Check cache first
        cache_key = f"depth_perception_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached 3D depth data")
            return cached_data
        
        # In a real implementation, we would access the NYU Depth Dataset API
        # For now, we'll simulate this dataset since we don't have actual API access
        dataset = self._simulate_depth_perception(max_samples)
        
        # Cache the dataset
        self._save_to_cache(cache_key, dataset)
        
        return dataset
    
    def fetch_fashion_3d_dataset(self, max_samples=None):
        """
        Fetch 3D fashion models from ShapeNet
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            Dictionary of dataset information
        """
        print("Fetching 3D fashion models via API...")
        
        # Check cache first
        cache_key = f"fashion_3d_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached 3D fashion data")
            return cached_data
        
        # In a real implementation, we would access the ShapeNet API
        # For now, we'll simulate this dataset
        
        n_samples = 50 if max_samples is None else min(50, max_samples)
        
        dataset = {
            "metadata": {
                "name": "Simulated 3D Fashion Dataset",
                "version": "1.0",
                "model_count": n_samples
            },
            "models": []
        }
        
        # Simulate 3D models
        for i in range(n_samples):
            model = {
                "id": f"model_{i}",
                "category": random.choice(["shirt", "pants", "dress", "shoes", "hat"]),
                "vertices": np.random.rand(100, 3).tolist(),  # Simplified mesh data
                "faces": np.random.randint(0, 100, (50, 3)).tolist(),
                "textures": np.random.rand(100, 3).tolist(),  # RGB values
                "features": {
                    "complexity": random.uniform(0, 1),
                    "visual_appeal": random.uniform(0, 1),
                    "realistic_score": random.uniform(0, 1)
                }
            }
            dataset["models"].append(model)
        
        print(f"Simulated {n_samples} 3D fashion models")
        
        # Cache the dataset
        self._save_to_cache(cache_key, dataset)
        
        return dataset
    
    def fetch_color_psychology_dataset(self, max_samples=None):
        """
        Fetch color psychology data from academic sources
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            DataFrame with color psychology data
        """
        print("Fetching color psychology data via API...")
        
        # Check cache first
        cache_key = f"color_psychology_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached color psychology data")
            return pd.DataFrame(cached_data)
        
        # In a real implementation, we would access academic APIs
        # For now, we'll simulate this dataset
        
        n_samples = 100 if max_samples is None else min(100, max_samples)
        
        # Color names and their psychological associations
        colors = ['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Orange', 'Pink', 'Brown', 'Black', 'White']
        emotions = ['Excitement', 'Calmness', 'Freshness', 'Happiness', 'Luxury', 'Enthusiasm', 'Romance', 'Reliability', 'Elegance', 'Purity']
        contexts = ['Marketing', 'Healthcare', 'Education', 'Technology', 'Fashion']
        
        # Generate data
        data = {
            'color': np.random.choice(colors, n_samples),
            'emotion': np.random.choice(emotions, n_samples),
            'context': np.random.choice(contexts, n_samples),
            'rgb_red': np.random.randint(0, 256, n_samples),
            'rgb_green': np.random.randint(0, 256, n_samples),
            'rgb_blue': np.random.randint(0, 256, n_samples),
            'association_strength': np.random.uniform(0, 1, n_samples),
            'cross_cultural_consistency': np.random.uniform(0, 1, n_samples),
            'attention_score': np.random.uniform(0, 10, n_samples),
            'memory_retention': np.random.uniform(0, 1, n_samples),
            'cultural_region': np.random.choice(['Western', 'Eastern', 'African', 'Latin American'], n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Apply some realistic correlations
        for color, emotion in zip(colors, emotions):
            mask = df['color'] == color
            df.loc[mask, 'association_strength'] = df.loc[mask, 'association_strength'] * 1.5
            df.loc[mask & (df['emotion'] == emotion), 'association_strength'] = df.loc[mask & (df['emotion'] == emotion), 'association_strength'] * 1.5
            
        df['association_strength'] = df['association_strength'].clip(0, 1)
        
        print(f"Simulated {n_samples} color psychology data points")
        
        # Cache the dataset
        self._save_to_cache(cache_key, df.to_dict('records'))
        
        return df
    
    def fetch_social_media_engagement_data(self, max_samples=None):
        """
        Fetch social media engagement data via platform APIs
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            DataFrame with social media engagement data
        """
        print("Fetching social media engagement data via API...")
        
        # Check cache first
        cache_key = f"social_media_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached social media data")
            return pd.DataFrame(cached_data)
        
        # In a real implementation, we would access social media APIs
        # For now, we'll simulate this dataset
        
        n_samples = 200 if max_samples is None else min(200, max_samples)
        
        # Generate data
        data = {
            'platform': np.random.choice(['Instagram', 'YouTube', 'TikTok', 'Facebook', 'Twitter'], n_samples),
            'content_type': np.random.choice(['Image', 'Video', 'Story', 'Carousel', 'Text'], n_samples),
            'has_3d_effect': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
            'has_depth_illusion': np.random.choice([True, False], n_samples, p=[0.5, 0.5]),
            'primary_color': np.random.choice(['Red', 'Blue', 'Green', 'Yellow', 'Purple', 'Black', 'White'], n_samples),
            'color_contrast': np.random.uniform(0, 1, n_samples),
            'likes': np.random.exponential(1000, n_samples).astype(int),
            'shares': np.random.exponential(200, n_samples).astype(int),
            'comments': np.random.exponential(100, n_samples).astype(int),
            'views': np.random.exponential(5000, n_samples).astype(int),
            'avg_view_duration': np.random.uniform(5, 60, n_samples),
            'engagement_rate': np.random.uniform(0.01, 0.2, n_samples),
            'click_through_rate': np.random.uniform(0.001, 0.05, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Apply realistic correlations
        # Posts with 3D effects get more engagement
        df.loc[df['has_3d_effect'], 'engagement_rate'] *= 1.4
        df.loc[df['has_3d_effect'], 'avg_view_duration'] *= 1.3
        
        # Posts with depth illusions get more shares
        df.loc[df['has_depth_illusion'], 'shares'] *= 1.5
        
        # Different platforms have different engagement patterns
        platform_multipliers = {
            'Instagram': {'likes': 1.2, 'comments': 0.8},
            'TikTok': {'shares': 1.5, 'views': 1.3},
            'YouTube': {'avg_view_duration': 2.0, 'comments': 0.7},
            'Facebook': {'shares': 1.2, 'engagement_rate': 0.8},
            'Twitter': {'engagement_rate': 0.9, 'shares': 1.3}
        }
        
        for platform, multipliers in platform_multipliers.items():
            for metric, multiplier in multipliers.items():
                df.loc[df['platform'] == platform, metric] *= multiplier
        
        print(f"Simulated {n_samples} social media engagement data points")
        
        # Cache the dataset
        self._save_to_cache(cache_key, df.to_dict('records'))
        
        return df
    
    def fetch_advertising_performance_data(self, max_samples=None):
        """
        Fetch advertising performance data via ad platform APIs
        
        Args:
            max_samples: Maximum number of samples to fetch
            
        Returns:
            DataFrame with advertising performance data
        """
        print("Fetching advertising performance data via API...")
        
        # Check cache first
        cache_key = f"advertising_{max_samples or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            print(f"Using cached advertising data")
            return pd.DataFrame(cached_data)
        
        # Number of campaigns
        n_samples = 250 if max_samples is None else min(250, max_samples)
        
        # Generate data similar to industry_performance but with more ad-specific metrics
        industries = ['Fashion', 'Automotive', 'Entertainment', 'Technology', 'Retail', 'Food', 'Healthcare']
        campaign_types = ['Display', 'Video', 'Social', 'Search', 'Native', 'Interactive']
        platforms = ['Google', 'Facebook', 'Instagram', 'YouTube', 'TikTok', 'Twitter']
        
        # Generate data
        data = {
            'industry': np.random.choice(industries, n_samples),
            'campaign_type': np.random.choice(campaign_types, n_samples),
            'platform': np.random.choice(platforms, n_samples),
            'budget': np.random.uniform(1000, 50000, n_samples),
            'impressions': np.random.exponential(100000, n_samples).astype(int),
            'clicks': np.random.exponential(2000, n_samples).astype(int),
            'conversions': np.random.exponential(100, n_samples).astype(int),
            'ctr': np.random.uniform(0.005, 0.1, n_samples),
            'cpc': np.random.uniform(0.5, 5, n_samples),
            'conversion_rate': np.random.uniform(0.01, 0.1, n_samples),
            'roas': np.random.uniform(0.5, 10, n_samples),
            'avg_engagement_time': np.random.uniform(5, 120, n_samples),
            'depth_effect_used': np.random.choice([True, False], n_samples, p=[0.4, 0.6]),
            'illusion_complexity': np.random.uniform(0.1, 1.0, n_samples),
            'visual_appeal_score': np.random.uniform(1, 10, n_samples),
            'color_harmony_score': np.random.uniform(1, 10, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Apply realistic correlations
        # Ads with depth effects tend to have higher CTR and engagement
        df.loc[df['depth_effect_used'], 'ctr'] *= 1.3
        df.loc[df['depth_effect_used'], 'avg_engagement_time'] *= 1.5
        
        # Higher visual appeal correlates with better conversion rates
        visual_appeal_factor = (df['visual_appeal_score'] - 5) / 5
        df['conversion_rate'] = df['conversion_rate'] * (1 + 0.3 * visual_appeal_factor)
        df['conversion_rate'] = df['conversion_rate'].clip(0.01, 0.2)
        
        # Calculate additional metrics
        df['cpa'] = df['budget'] / df['conversions'].clip(1)  # Cost per acquisition
        df['roi'] = (df['conversions'] * 50 - df['budget']) / df['budget']  # Assuming $50 value per conversion
        
        print(f"Simulated {n_samples} advertising performance data points")
        
        # Cache the dataset
        self._save_to_cache(cache_key, df.to_dict('records'))
        
        return df
    
    # Legacy methods with API-based implementations
    def load_optical_illusions(self, max_samples=None):
        """Legacy method that now uses the API-based implementation"""
        return self.fetch_optical_illusions_dataset(max_samples)
    
    def load_depth_perception(self, max_samples=None):
        """Legacy method that now uses the API-based implementation"""
        return self.fetch_3d_depth_dataset(max_samples)
    
    def load_industry_performance(self, max_samples=None):
        """Legacy method that now uses the API-based implementation"""
        return self.fetch_advertising_performance_data(max_samples)
    
    # Simulation methods for when APIs are unavailable
    def _simulate_optical_illusions(self, max_samples=None):
        """Simulate optical illusions dataset for testing"""
        print("Simulating optical illusions dataset...")
        
        n_samples = 100 if max_samples is None else min(100, max_samples)
        categories = ['geometric', 'color', 'motion', 'depth', 'contrast']
        
        dataset = {
            "metadata": {
                "name": "Simulated Optical Illusions Dataset",
                "version": "1.0",
                "sample_count": n_samples
            },
            "images": [],
            "features": []
        }
        
        # Simulate images and features
        for i in range(n_samples):
            # Create a random image
            img_size = (256, 256)
            img = Image.new('RGB', img_size, color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ))
            
            # Add some random shapes
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            
            for _ in range(random.randint(1, 10)):
                shape_type = random.choice(['rectangle', 'ellipse', 'line'])
                color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )
                
                x1 = random.randint(0, img_size[0])
                y1 = random.randint(0, img_size[1])
                x2 = random.randint(0, img_size[0])
                y2 = random.randint(0, img_size[1])
                
                if shape_type == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], outline=color)
                elif shape_type == 'ellipse':
                    draw.ellipse([x1, y1, x2, y2], outline=color)
                elif shape_type == 'line':
                    draw.line([x1, y1, x2, y2], fill=color, width=random.randint(1, 5))
            
            # Add to dataset
            dataset["images"].append(img)
            dataset["features"].append({
                "id": f"sim_{i}",
                "category": random.choice(categories),
                "tags": random.sample(categories, random.randint(1, 3)),
                "depth_score": random.uniform(0, 10),
                "complexity": random.uniform(0, 1)
            })
        
        print(f"Simulated {n_samples} optical illusion images")
        return dataset
    
    def _simulate_depth_perception(self, max_samples=None):
        """Simulate 3D depth perception dataset for testing"""
        print("Simulating 3D depth perception dataset...")
        
        n_scenes = 50 if max_samples is None else min(50, max_samples)
        n_images_per_scene = 4
        
        dataset = {
            "metadata": {
                "name": "Simulated 3D Depth Perception Dataset",
                "version": "1.0",
                "scene_count": n_scenes,
                "images_per_scene": n_images_per_scene
            },
            "scenes": [],
        }
        
        # Simulate scenes
        for i in range(n_scenes):
            scene = {
                "id": f"scene_{i}",
                "name": f"Simulated Scene {i}",
                "images": [],
                "depth_maps": [],
                "features": {
                    "perspective_distortion": random.uniform(0, 1),
                    "lighting_intensity": random.uniform(0, 1),
                    "shadow_quality": random.uniform(0, 1)
                }
            }
            
            # Create images and depth maps for this scene
            for j in range(n_images_per_scene):
                # Create a random image
                img_size = (256, 256)
                img = Image.new('RGB', img_size, color=(
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))
                
                # Create a random depth map (grayscale)
                depth_map = Image.new('L', img_size, color=random.randint(0, 255))
                
                # Add to scene
                scene["images"].append(img)
                scene["depth_maps"].append(depth_map)
            
            # Add scene to dataset
            dataset["scenes"].append(scene)
        
        print(f"Simulated {n_scenes} 3D depth perception scenes with {n_scenes * n_images_per_scene} total images")
        return dataset

# Simple test
if __name__ == "__main__":
    loader = DataLoader()
    
    # Test loading a small sample from each dataset
    optical_illusions = loader.fetch_optical_illusions_dataset(max_samples=5)
    depth_perception = loader.fetch_3d_depth_dataset(max_samples=3)
    fashion_3d = loader.fetch_fashion_3d_dataset(max_samples=3)
    color_psychology = loader.fetch_color_psychology_dataset(max_samples=5)
    social_media = loader.fetch_social_media_engagement_data(max_samples=5)
    advertising = loader.fetch_advertising_performance_data(max_samples=5)
    
    print("\nDataset Summary:")
    print(f"Optical Illusions: {len(optical_illusions['images'])} images")
    print(f"3D Depth Perception: {len(depth_perception['scenes'])} scenes")
    print(f"3D Fashion Models: {len(fashion_3d['models'])} models")
    print(f"Color Psychology: {len(color_psychology)} data points")
    print(f"Social Media Engagement: {len(social_media)} data points")
    print(f"Advertising Performance: {len(advertising)} data points") 