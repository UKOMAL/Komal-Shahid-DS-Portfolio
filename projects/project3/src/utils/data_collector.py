import os
import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
import hashlib

class AnamorphicDataCollector:
    def __init__(self, output_dir="projects/project3/dataset/anamorphic"):
        """Initialize the data collector with the output directory."""
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, "raw")
        self.processed_dir = os.path.join(output_dir, "processed")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.categories_dir = os.path.join(output_dir, "categories")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.categories_dir, exist_ok=True)
        
        # List of sources to scrape
        self.sources = [
            {
                "name": "times_square_examples",
                "url": "https://www.linsnled.com/3d-billboard-time-square.html",
                "parser": self._parse_times_square
            },
            {
                "name": "groove_jones_examples",
                "url": "https://www.groovejones.com/3d-billboards-101-anamorphic-and-forced-perspective-ooh-campaigns",
                "parser": self._parse_groove_jones
            },
            {
                "name": "las_vegas_examples",
                "url": "https://www.vegas.com/attractions/on-the-strip/digital-billboards/",
                "parser": self._parse_vegas_examples
            },
            {
                "name": "museum_installations",
                "url": "https://www.thisiscolossal.com/tags/anamorphic/",
                "parser": self._parse_colossal_examples
            },
            {
                "name": "creative_3d_billboards",
                "url": "https://www.adsoftheworld.com/collection/3d_billboards",
                "parser": self._parse_ads_of_world
            },
            {
                "name": "dstrict_examples",
                "url": "https://www.dstrict.com/en/portfolio.php",
                "parser": self._parse_dstrict
            }
        ]
        
        self.metadata = {
            "dataset_name": "3D Anamorphic Illusions Dataset",
            "version": "1.0",
            "sources": [],
            "images": []
        }
    
    def _parse_times_square(self, soup, base_url):
        """Parse the Times Square examples page."""
        images = []
        # Look for image elements in the page
        img_elements = soup.select("img")
        
        for img in img_elements:
            src = img.get("src")
            if src and ("billboard" in src.lower() or "3d" in src.lower()):
                # Extract alt text or nearby headings as description
                alt = img.get("alt", "")
                parent = img.parent
                heading = parent.find_previous(["h1", "h2", "h3", "h4"])
                description = heading.text.strip() if heading else alt
                
                # Add to images list
                images.append({
                    "url": urljoin(base_url, src),
                    "description": description,
                    "category": "billboard",
                    "source": "times_square_examples"
                })
        
        return images
    
    def _parse_groove_jones(self, soup, base_url):
        """Parse the Groove Jones examples page."""
        images = []
        # Look for image elements in the page
        img_elements = soup.select("img")
        
        for img in img_elements:
            src = img.get("src")
            if src and ("billboard" in src.lower() or "3d" in src.lower()):
                # Extract alt text or nearby headings as description
                alt = img.get("alt", "")
                parent = img.parent
                heading = parent.find_previous(["h1", "h2", "h3", "h4"])
                description = heading.text.strip() if heading else alt
                
                # Add to images list
                images.append({
                    "url": urljoin(base_url, src),
                    "description": description,
                    "category": "billboard",
                    "source": "groove_jones_examples"
                })
        
        return images
    
    def _parse_vegas_examples(self, soup, base_url):
        """Parse Las Vegas digital billboards and art installations."""
        images = []
        # Look for high-quality images on the page
        img_elements = soup.select("img.img-responsive, img.featured-image, div.image-container img")
        
        for img in img_elements:
            src = img.get("src")
            if src:
                # Extract alt text or nearby headings as description
                alt = img.get("alt", "")
                parent = img.parent
                heading = parent.find_previous(["h1", "h2", "h3", "h4"])
                description = heading.text.strip() if heading else alt
                
                # Add to images list
                images.append({
                    "url": urljoin(base_url, src),
                    "description": description,
                    "category": "las_vegas_billboard",
                    "source": "las_vegas_examples"
                })
        
        return images
    
    def _parse_colossal_examples(self, soup, base_url):
        """Parse Colossal art website for anamorphic installations."""
        images = []
        # Look for article images
        articles = soup.select("article.post")
        
        for article in articles:
            img_elements = article.select("img")
            title_elem = article.select_one("h1, h2")
            title = title_elem.text.strip() if title_elem else "Art Installation"
            
            for img in img_elements:
                src = img.get("src")
                if src:
                    # Add to images list
                    images.append({
                        "url": urljoin(base_url, src),
                        "description": title,
                        "category": "art_installation",
                        "source": "museum_installations"
                    })
        
        return images
    
    def _parse_ads_of_world(self, soup, base_url):
        """Parse Ads of the World for creative 3D billboards."""
        images = []
        # Look for showcase images
        img_elements = soup.select("div.showcase img, div.item img")
        
        for img in img_elements:
            src = img.get("src") or img.get("data-src")
            if src:
                # Try to find associated title
                parent = img.parent
                while parent and not parent.select_one("h2, h3, .title"):
                    parent = parent.parent
                
                title_elem = parent.select_one("h2, h3, .title") if parent else None
                title = title_elem.text.strip() if title_elem else "3D Billboard"
                
                # Add to images list
                images.append({
                    "url": urljoin(base_url, src),
                    "description": title,
                    "category": "creative_billboard",
                    "source": "creative_3d_billboards"
                })
        
        return images
    
    def _parse_dstrict(self, soup, base_url):
        """Parse D'strict portfolio for high-quality 3D anamorphic displays."""
        images = []
        # D'strict is known for high-quality anamorphic displays
        img_elements = soup.select("div.portfolio-item img, div.project-item img")
        
        for img in img_elements:
            src = img.get("src") or img.get("data-src")
            if src:
                # Try to find associated title
                parent = img.parent
                while parent and not parent.select_one("h3, h4, .item-title"):
                    parent = parent.parent
                
                title_elem = parent.select_one("h3, h4, .item-title") if parent else None
                title = title_elem.text.strip() if title_elem else "3D Anamorphic Display"
                
                # Add to images list
                images.append({
                    "url": urljoin(base_url, src),
                    "description": title,
                    "category": "anamorphic_display",
                    "source": "dstrict_examples"
                })
        
        return images
    
    def download_image(self, url, save_path):
        """Download an image from a URL and save it to the specified path."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def collect_data(self):
        """Collect data from all sources."""
        for source in self.sources:
            print(f"Collecting data from {source['name']}...")
            
            try:
                response = requests.get(source["url"], timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                images = source["parser"](soup, source["url"])
                
                print(f"Found {len(images)} images from {source['name']}")
                
                # Add to metadata
                self.metadata["sources"].append({
                    "name": source["name"],
                    "url": source["url"],
                    "count": len(images)
                })
                
                # Download images
                for img in images:
                    # Generate filename from URL hash
                    url_hash = hashlib.md5(img["url"].encode()).hexdigest()
                    extension = os.path.splitext(img["url"])[1]
                    if not extension:
                        extension = ".jpg"  # Default extension
                    
                    filename = f"{source['name']}_{url_hash}{extension}"
                    save_path = os.path.join(self.raw_dir, filename)
                    
                    # Download the image
                    if self.download_image(img["url"], save_path):
                        # Add to metadata
                        img_metadata = {
                            "id": url_hash,
                            "filename": filename,
                            "url": img["url"],
                            "description": img["description"],
                            "category": img["category"],
                            "source": img["source"]
                        }
                        
                        self.metadata["images"].append(img_metadata)
                        
                        # Create symlink in categories directory
                        category_dir = os.path.join(self.categories_dir, img["category"])
                        os.makedirs(category_dir, exist_ok=True)
                        
                        category_path = os.path.join(category_dir, filename)
                        if os.path.exists(category_path):
                            os.remove(category_path)
                        
                        # Create relative path to the raw file
                        rel_path = os.path.relpath(save_path, category_dir)
                        os.symlink(rel_path, category_path)
                    
                    # Sleep to avoid hammering the server
                    time.sleep(0.5)
            
            except Exception as e:
                print(f"Error processing {source['name']}: {e}")
        
        # Save metadata
        metadata_path = os.path.join(self.metadata_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"Collected {len(self.metadata['images'])} images in total")
        print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    collector = AnamorphicDataCollector()
    collector.collect_data() 