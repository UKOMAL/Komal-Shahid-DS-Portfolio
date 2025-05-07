/**
 * Builder.io Integration for Portfolio Website
 * This script fetches and renders content from Builder.io
 */

document.addEventListener('DOMContentLoaded', function() {
  // Builder.io configuration
  const BUILDER_API_KEY = 'f947ced278234b4d8dc12c19358844bf'; // Your Builder.io API key
  const BUILDER_CONTENT_ID = '9bd80b306a124819add107e98c61d45d';
  const BUILDER_BLOCK_ID = 'builder-dc76033cf6044e39a8482b935d844130';
  
  // Fetch the Builder.io content
  async function fetchBuilderContent() {
    try {
      const response = await fetch(
        `https://cdn.builder.io/api/v2/content/${BUILDER_CONTENT_ID}?apiKey=${BUILDER_API_KEY}&includeRefs=true`
      );
      
      if (!response.ok) {
        throw new Error(`Builder.io API error: ${response.status}`);
      }
      
      const data = await response.json();
      if (!data.data || !data.data.blocks) {
        throw new Error('Invalid response format from Builder.io');
      }
      
      // Find the specific block by ID
      const targetBlock = findBlockById(data.data.blocks, BUILDER_BLOCK_ID);
      
      if (targetBlock) {
        console.log('Builder.io block found:', targetBlock);
        applyBlockToWebsite(targetBlock);
      } else {
        // Fallback to the hardcoded version if block not found
        console.warn('Specific Builder.io block not found, using fallback');
        applyFallbackStyles();
      }
    } catch (error) {
      console.error('Error fetching Builder.io content:', error);
      // Use fallback if fetch fails
      applyFallbackStyles();
    }
  }
  
  // Helper function to find a block by ID
  function findBlockById(blocks, id) {
    for (const block of blocks) {
      if (block.id === id) {
        return block;
      }
      
      // Check children recursively
      if (block.blocks && block.blocks.length > 0) {
        const foundInChildren = findBlockById(block.blocks, id);
        if (foundInChildren) {
          return foundInChildren;
        }
      }
    }
    
    return null;
  }
  
  // Apply the Builder.io block to the website
  function applyBlockToWebsite(block) {
    // For now we'll focus on profile image styling
    const profileContainer = document.querySelector('.hero-image');
    if (!profileContainer) return;
    
    const profileImage = profileContainer.querySelector('img');
    if (!profileImage) return;
    
    // Apply properties from the block if it's an image
    if (block.tagName === 'img' && block.properties) {
      Object.entries(block.properties).forEach(([key, value]) => {
        if (key !== 'src') { // Keep the GitHub profile image
          profileImage.setAttribute(key, value);
        }
      });
    }
    
    // Apply styles from the block
    if (block.responsiveStyles && block.responsiveStyles.large) {
      Object.entries(block.responsiveStyles.large).forEach(([property, value]) => {
        profileImage.style[property] = value;
      });
    }
    
    console.log('Builder.io styles applied to profile image');
  }
  
  // Fallback to hardcoded styling if API fetch fails
  function applyFallbackStyles() {
    const profileImageComponent = {
      "tagName": "img",
      "properties": {
        "src": "https://github.com/UKOMAL.png",
        "alt": "Komal Shahid"
      },
      "responsiveStyles": {
        "large": {
          "position": "relative",
          "zIndex": "10",
          "borderRadius": "40px",
          "width": "100%",
          "height": "100%",
          "objectFit": "cover",
          "borderWidth": "2px",
          "borderColor": "#2a2a2a15",
          "boxShadow": "0 8px 32px rgba(0,0,0,0.1)"
        }
      }
    };
    
    const profileContainer = document.querySelector('.hero-image');
    if (!profileContainer) return;
    
    const profileImage = profileContainer.querySelector('img');
    if (!profileImage) return;
    
    // Apply properties from fallback
    const properties = profileImageComponent.properties || {};
    const styles = profileImageComponent.responsiveStyles?.large || {};
    
    Object.entries(properties).forEach(([key, value]) => {
      if (key !== 'src') { // Keep the GitHub profile image
        profileImage.setAttribute(key, value);
      }
    });
    
    // Apply styles from fallback
    Object.entries(styles).forEach(([property, value]) => {
      profileImage.style[property] = value;
    });
    
    console.log('Fallback Builder.io styles applied to profile image');
  }
  
  // Initialize
  fetchBuilderContent();
}); 