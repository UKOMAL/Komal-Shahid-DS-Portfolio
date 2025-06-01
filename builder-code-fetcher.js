#!/usr/bin/env node

/**
 * Builder.io Code Fetcher
 * 
 * This script fetches code from a Builder.io content item and saves it locally.
 * It's a simplified version of the "npx builder.io@latest code" command.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
let url = '';

for (let i = 0; i < args.length; i++) {
  if (args[i] === '--url' && args[i+1]) {
    url = args[i+1];
    break;
  }
}

if (!url) {
  console.error('Error: URL is required. Use --url "https://builder.io/content/..."');
  process.exit(1);
}

// Extract content ID from URL
let contentId = '';
const contentIdMatch = url.match(/\/content\/([a-f0-9]+)/i);
if (contentIdMatch && contentIdMatch[1]) {
  contentId = contentIdMatch[1];
} else {
  console.error('Error: Could not extract content ID from URL');
  process.exit(1);
}

// Extract block ID from URL if available
let blockId = '';
const blockIdMatch = url.match(/selectedBlock=([a-f0-9-]+)/i);
if (blockIdMatch && blockIdMatch[1]) {
  blockId = blockIdMatch[1];
}

console.log(`Fetching content ID: ${contentId} ${blockId ? `(block: ${blockId})` : ''}`);

// Fetch content from Builder.io API
const apiUrl = `https://cdn.builder.io/api/v2/content/${contentId}?apiKey=YOUR_BUILDER_API_KEY&includeRefs=true`;

https.get(apiUrl, (res) => {
  let data = '';

  res.on('data', (chunk) => {
    data += chunk;
  });

  res.on('end', () => {
    try {
      const parsedData = JSON.parse(data);
      
      if (!parsedData.data) {
        console.error('Error: Content not found or invalid response from Builder.io');
        process.exit(1);
      }

      // Extract the blocks from the content
      const blocks = parsedData.data.blocks || [];
      
      // If a specific block ID was provided, find only that block
      let targetBlock = null;
      if (blockId) {
        targetBlock = findBlockById(blocks, blockId);
      } else {
        // Otherwise, just use the first block
        targetBlock = blocks[0];
      }
      
      if (!targetBlock) {
        console.error('Error: Block not found in content');
        process.exit(1);
      }
      
      // Save the block data to a file
      const outputFile = `builder-block-${contentId.substring(0, 8)}.json`;
      fs.writeFileSync(outputFile, JSON.stringify(targetBlock, null, 2));
      
      console.log(`Block data saved to ${outputFile}`);
      
      // Print summary of the block
      console.log('\nBlock Summary:');
      console.log(`Type: ${targetBlock.tagName || 'component'}`);
      if (targetBlock.component) {
        console.log(`Component: ${targetBlock.component.name}`);
      }
      if (targetBlock.properties) {
        console.log('Properties:');
        Object.keys(targetBlock.properties).forEach(key => {
          const value = targetBlock.properties[key];
          console.log(`  ${key}: ${typeof value === 'object' ? JSON.stringify(value).substring(0, 50) + '...' : value}`);
        });
      }
      
    } catch (error) {
      console.error('Error parsing API response:', error.message);
      process.exit(1);
    }
  });
}).on('error', (err) => {
  console.error('Error fetching from Builder.io API:', err.message);
  process.exit(1);
});

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