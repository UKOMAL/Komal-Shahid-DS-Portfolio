/**
 * Figma Portfolio Integration
 * This script enhances the Figma embeds with additional functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get all Figma embeds
    const figmaEmbeds = document.querySelectorAll('.figma-embed');
    
    // Add loading state
    figmaEmbeds.forEach(embed => {
        // Create loading overlay
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'figma-loading';
        loadingOverlay.innerHTML = '<div class="figma-spinner"></div><p>Loading Figma Design...</p>';
        
        // Add loading overlay to parent
        embed.parentNode.insertBefore(loadingOverlay, embed);
        
        // Remove loading state when iframe loads
        embed.addEventListener('load', function() {
            loadingOverlay.style.opacity = '0';
            setTimeout(() => {
                loadingOverlay.remove();
            }, 500);
        });
    });
    
    // Responsive height adjustment for mobile
    function adjustFigmaHeight() {
        figmaEmbeds.forEach(embed => {
            if (window.innerWidth < 768) {
                embed.style.height = '300px';
            } else {
                embed.style.height = '450px';
            }
        });
    }
    
    // Initial call and event listener
    adjustFigmaHeight();
    window.addEventListener('resize', adjustFigmaHeight);
    
    // Handle Figma tool buttons
    const figmaToolButtons = document.querySelectorAll('.figma-tool-button');
    figmaToolButtons.forEach(button => {
        button.addEventListener('click', function() {
            const action = this.getAttribute('title');
            
            if (action === 'View full screen') {
                const container = this.closest('.figma-container');
                const embed = container.querySelector('.figma-embed');
                
                if (container.classList.contains('fullscreen')) {
                    // Exit fullscreen
                    container.classList.remove('fullscreen');
                    document.body.style.overflow = 'auto';
                    this.innerHTML = '<i class="fas fa-expand"></i>';
                    this.setAttribute('title', 'View full screen');
                    
                    // Reset height
                    if (window.innerWidth < 768) {
                        embed.style.height = '300px';
                    } else {
                        embed.style.height = '450px';
                    }
                } else {
                    // Enter fullscreen
                    container.classList.add('fullscreen');
                    document.body.style.overflow = 'hidden';
                    this.innerHTML = '<i class="fas fa-compress"></i>';
                    this.setAttribute('title', 'Exit full screen');
                    
                    // Adjust height for fullscreen
                    embed.style.height = (window.innerHeight - 100) + 'px';
                }
            }
            
            if (action === 'View comments') {
                // Mock Figma comments UI
                const container = this.closest('.figma-container');
                let commentsPanel = container.querySelector('.figma-comments');
                
                if (commentsPanel) {
                    // Toggle existing panel
                    commentsPanel.classList.toggle('visible');
                } else {
                    // Create comments panel
                    commentsPanel = document.createElement('div');
                    commentsPanel.className = 'figma-comments visible';
                    commentsPanel.innerHTML = `
                        <div class="figma-comments-header">
                            <h4>Comments</h4>
                            <button class="figma-close-btn"><i class="fas fa-times"></i></button>
                        </div>
                        <div class="figma-comments-list">
                            <div class="figma-comment">
                                <div class="figma-comment-avatar">KS</div>
                                <div class="figma-comment-content">
                                    <div class="figma-comment-author">Komal Shahid</div>
                                    <div class="figma-comment-text">The color contrast on this chart could be improved for better accessibility.</div>
                                    <div class="figma-comment-time">2 days ago</div>
                                </div>
                            </div>
                            <div class="figma-comment">
                                <div class="figma-comment-avatar">JD</div>
                                <div class="figma-comment-content">
                                    <div class="figma-comment-author">John Doe</div>
                                    <div class="figma-comment-text">I like the layout of the dashboard. Very intuitive!</div>
                                    <div class="figma-comment-time">1 week ago</div>
                                </div>
                            </div>
                        </div>
                        <div class="figma-comments-input">
                            <input type="text" placeholder="Add comment...">
                            <button class="figma-send-btn"><i class="fas fa-paper-plane"></i></button>
                        </div>
                    `;
                    
                    container.appendChild(commentsPanel);
                    
                    // Add close button functionality
                    const closeBtn = commentsPanel.querySelector('.figma-close-btn');
                    closeBtn.addEventListener('click', function() {
                        commentsPanel.classList.remove('visible');
                    });
                    
                    // Add mock send button functionality
                    const sendBtn = commentsPanel.querySelector('.figma-send-btn');
                    const commentInput = commentsPanel.querySelector('input');
                    
                    sendBtn.addEventListener('click', function() {
                        if (commentInput.value.trim() !== '') {
                            const commentList = commentsPanel.querySelector('.figma-comments-list');
                            const newComment = document.createElement('div');
                            newComment.className = 'figma-comment new-comment';
                            newComment.innerHTML = `
                                <div class="figma-comment-avatar">KS</div>
                                <div class="figma-comment-content">
                                    <div class="figma-comment-author">Komal Shahid</div>
                                    <div class="figma-comment-text">${commentInput.value}</div>
                                    <div class="figma-comment-time">Just now</div>
                                </div>
                            `;
                            
                            commentList.appendChild(newComment);
                            commentInput.value = '';
                            
                            // Animate the new comment
                            setTimeout(() => {
                                newComment.classList.add('visible');
                            }, 10);
                        }
                    });
                }
            }
            
            if (action === 'View code') {
                // Mock Figma code export UI
                const container = this.closest('.figma-container');
                let codePanel = container.querySelector('.figma-code-panel');
                
                if (codePanel) {
                    // Toggle existing panel
                    codePanel.classList.toggle('visible');
                } else {
                    // Create code panel
                    codePanel = document.createElement('div');
                    codePanel.className = 'figma-code-panel visible';
                    codePanel.innerHTML = `
                        <div class="figma-code-header">
                            <h4>Design to Code</h4>
                            <button class="figma-close-btn"><i class="fas fa-times"></i></button>
                        </div>
                        <div class="figma-code-tabs">
                            <button class="figma-code-tab active" data-tab="html">HTML</button>
                            <button class="figma-code-tab" data-tab="css">CSS</button>
                            <button class="figma-code-tab" data-tab="react">React</button>
                        </div>
                        <div class="figma-code-content">
                            <pre class="figma-code-block html active"><code>&lt;div class="dashboard-container"&gt;
  &lt;header class="dashboard-header"&gt;
    &lt;h1&gt;Healthcare Analytics Dashboard&lt;/h1&gt;
    &lt;div class="dashboard-controls"&gt;
      &lt;div class="date-selector"&gt;
        &lt;label&gt;Date Range&lt;/label&gt;
        &lt;select&gt;
          &lt;option&gt;Last 7 Days&lt;/option&gt;
          &lt;option&gt;Last 30 Days&lt;/option&gt;
          &lt;option&gt;Custom...&lt;/option&gt;
        &lt;/select&gt;
      &lt;/div&gt;
    &lt;/div&gt;
  &lt;/header&gt;
  &lt;div class="dashboard-grid"&gt;
    &lt;!-- Charts and visualizations --&gt;
  &lt;/div&gt;
&lt;/div&gt;</code></pre>
                            <pre class="figma-code-block css"><code>.dashboard-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Inter', sans-serif;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
}

.dashboard-header h1 {
  font-size: 1.8rem;
  font-weight: 600;
  color: #333;
}

.dashboard-controls {
  display: flex;
  gap: 1rem;
}</code></pre>
                            <pre class="figma-code-block react"><code>import React, { useState } from 'react';
import './Dashboard.css';

const Dashboard = () => {
  const [dateRange, setDateRange] = useState('last7Days');
  
  return (
    &lt;div className="dashboard-container"&gt;
      &lt;header className="dashboard-header"&gt;
        &lt;h1&gt;Healthcare Analytics Dashboard&lt;/h1&gt;
        &lt;div className="dashboard-controls"&gt;
          &lt;div className="date-selector"&gt;
            &lt;label&gt;Date Range&lt;/label&gt;
            &lt;select 
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value)}
            &gt;
              &lt;option value="last7Days"&gt;Last 7 Days&lt;/option&gt;
              &lt;option value="last30Days"&gt;Last 30 Days&lt;/option&gt;
              &lt;option value="custom"&gt;Custom...&lt;/option&gt;
            &lt;/select&gt;
          &lt;/div&gt;
        &lt;/div&gt;
      &lt;/header&gt;
      &lt;div className="dashboard-grid"&gt;
        {/* Charts and visualizations */}
      &lt;/div&gt;
    &lt;/div&gt;
  );
};

export default Dashboard;</code></pre>
                        </div>
                        <div class="figma-code-footer">
                            <button class="figma-copy-btn"><i class="fas fa-copy"></i> Copy Code</button>
                            <button class="figma-download-btn"><i class="fas fa-download"></i> Download</button>
                        </div>
                    `;
                    
                    container.appendChild(codePanel);
                    
                    // Add close button functionality
                    const closeBtn = codePanel.querySelector('.figma-close-btn');
                    closeBtn.addEventListener('click', function() {
                        codePanel.classList.remove('visible');
                    });
                    
                    // Add tab functionality
                    const codeTabs = codePanel.querySelectorAll('.figma-code-tab');
                    const codeBlocks = codePanel.querySelectorAll('.figma-code-block');
                    
                    codeTabs.forEach(tab => {
                        tab.addEventListener('click', function() {
                            const tabType = this.getAttribute('data-tab');
                            
                            // Update active tab
                            codeTabs.forEach(t => t.classList.remove('active'));
                            this.classList.add('active');
                            
                            // Show corresponding code block
                            codeBlocks.forEach(block => {
                                block.classList.remove('active');
                                if (block.classList.contains(tabType)) {
                                    block.classList.add('active');
                                }
                            });
                        });
                    });
                    
                    // Add copy functionality
                    const copyBtn = codePanel.querySelector('.figma-copy-btn');
                    copyBtn.addEventListener('click', function() {
                        const activeCode = codePanel.querySelector('.figma-code-block.active code');
                        const textToCopy = activeCode.textContent;
                        
                        navigator.clipboard.writeText(textToCopy).then(() => {
                            // Show copied notification
                            const originalText = this.innerHTML;
                            this.innerHTML = '<i class="fas fa-check"></i> Copied!';
                            
                            setTimeout(() => {
                                this.innerHTML = originalText;
                            }, 2000);
                        });
                    });
                }
            }
        });
    });
    
    // Handle Figma prototype link
    const prototypeLinks = document.querySelectorAll('.figma-prototype-link');
    prototypeLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Find the associated iframe
            const container = this.closest('.figma-container');
            const embed = container.querySelector('.figma-embed');
            
            // Replace the src with a prototype URL (this is just a demo)
            const currentSrc = embed.getAttribute('src');
            if (currentSrc.includes('&node-id=')) {
                // Already showing prototype, revert to design
                const baseUrl = currentSrc.split('&node-id=')[0];
                embed.setAttribute('src', baseUrl);
                this.innerHTML = '<i class="fas fa-play-circle"></i> View Prototype';
            } else {
                // Show prototype
                const prototypeUrl = currentSrc + '&node-id=2%3A2&scaling=contain&starting-point-node-id=2%3A2';
                embed.setAttribute('src', prototypeUrl);
                this.innerHTML = '<i class="fas fa-object-group"></i> View Design';
                
                // Reset the loading state
                const container = this.closest('.figma-container');
                const loadingOverlay = document.createElement('div');
                loadingOverlay.className = 'figma-loading';
                loadingOverlay.innerHTML = '<div class="figma-spinner"></div><p>Loading Prototype...</p>';
                container.appendChild(loadingOverlay);
                
                // Remove loading state when iframe loads
                embed.addEventListener('load', function onceLoaded() {
                    loadingOverlay.style.opacity = '0';
                    setTimeout(() => {
                        loadingOverlay.remove();
                    }, 500);
                    embed.removeEventListener('load', onceLoaded);
                });
            }
        });
    });
}); 