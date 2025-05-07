# Depression Detection System - Interactive Demo

This folder contains a fully client-side interactive demo of the Depression Detection System that can be easily integrated into your portfolio website or GitHub Pages.

## Overview

This demo allows users to:
- Enter text or select sample texts
- See a simulated analysis of depression indicators
- View visualized results including severity assessment and confidence scores
- Receive appropriate guidance based on the analysis

## Files

- `index.html` - Main HTML structure
- `styles.css` - CSS styles with robot theme matching your portfolio
- `script.js` - JavaScript code that handles the demo functionality

## Integration with GitHub Pages

### Option 1: Link directly from your projects page

1. Upload this entire `web` folder to your GitHub repository
2. In your `projects.md` file, update the "Interactive Demo" link to point to this web folder:

```markdown
<a href="projects/project1-depression-detection/demo/web/index.html" class="project-link" target="_blank">Interactive Demo</a>
```

### Option 2: Embed in an iframe

You can embed the demo directly into your portfolio page by adding an iframe:

```html
<iframe src="projects/project1-depression-detection/demo/web/index.html" width="100%" height="800px" frameborder="0"></iframe>
```

### Option 3: Create a dedicated page

1. Copy the HTML content into a new file called `depression-demo.md` in your repository root
2. Add the appropriate front matter:

```markdown
---
layout: default
title: Depression Detection Demo - Komal Shahid
---

<!-- Demo HTML content here -->
```

3. Link to this page from your projects page

## Customization

- **Colors**: Edit the CSS variables at the top of `styles.css` to match your portfolio's color scheme
- **Animation**: Adjust the robot animations in the CSS file to match your preferences
- **Sample Texts**: Modify the sample texts in the HTML file to demonstrate different scenarios

## Important Notes

- This is a **simulated demo** that runs entirely in the browser without a backend
- The text analysis algorithm is simplified for demonstration purposes
- Make sure to clearly communicate to users that this is a demonstration only and not a clinical tool

## Technical Details

- Pure HTML, CSS, and JavaScript (no dependencies)
- Responsive design that works on mobile devices
- Robot theme consistent with your portfolio's visual identity 