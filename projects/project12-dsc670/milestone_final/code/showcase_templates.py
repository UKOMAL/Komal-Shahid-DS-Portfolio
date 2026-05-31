#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Notely Template Showcase

This script creates all four template types with different content examples,
including images and charts to demonstrate their full capabilities.
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Helper functions (previously from creative_templates.py)
def get_common_html_header(title="NoteMorphAI Template"):
    """Generate common HTML header with styles for all templates"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;600;700&family=Kalam:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Kalam', cursive;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .notebook-page {{
            max-width: 800px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            position: relative;
            border: 3px solid #e1e8ed;
        }}
        
        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 16px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 18px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        
        .main-title {{
            font-family: 'Caveat', cursive;
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            transform: rotate(-1deg);
        }}
        
        .subtitle {{
            font-size: 18px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
            font-style: italic;
        }}
        
        .highlight-purple {{
            background-color: rgba(155, 89, 182, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .highlight-green {{
            background-color: rgba(39, 174, 96, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .highlight-pink {{
            background-color: rgba(233, 30, 99, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
        
        .highlight-yellow {{
            background-color: rgba(241, 196, 15, 0.3);
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }}
    </style>"""

def open_in_browser(filepath):
    """Open HTML file in default browser"""
    abs_path = os.path.abspath(filepath)
    webbrowser.open(f"file://{abs_path}")
    print(f"✅ Template created: {abs_path}")

# Import template functions
# from creative_templates import get_common_html_header, open_in_browser

def create_climate_change_infographic(output_path):
    """Create a climate change themed infographic"""
    title = "Climate Change Impact"
    
    # Custom styles for infographic
    INFOGRAPHIC_STYLES = """
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        .definition-box {
            background-color: rgba(129, 212, 250, 0.3);
            border: 2px solid #81d4fa;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .definition-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: #81d4fa;
            border-radius: 10px;
            z-index: -1;
        }
        
        .section-title {
            font-family: 'Caveat', cursive;
            font-size: 32px;
            font-weight: bold;
            color: #01579b;
            margin-bottom: 15px;
            display: inline-block;
            border-bottom: 3px dotted #0288d1;
            padding-bottom: 5px;
        }
        
        .definition-content {
            font-size: 18px;
            line-height: 1.7;
        }
        
        /* Issue cards */
        .issues-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin-top: 30px;
        }
        
        .issue-card {
            background-color: rgba(200, 230, 201, 0.3);
            border: 2px solid #a5d6a7;
            border-radius: 15px;
            padding: 0;
            overflow: hidden;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
            transform: rotate(1deg);
            transition: transform 0.3s ease;
        }
        
        .issue-card:nth-child(even) {
            background-color: rgba(179, 229, 252, 0.3);
            border-left-color: #29b6f6;
            transform: rotate(-1deg);
        }
        
        .issue-card:hover {
            transform: rotate(0) scale(1.02);
        }
        
        .issue-header {
            background-color: #a5d6a7;
            padding: 12px 15px;
            font-family: 'Caveat', cursive;
            font-size: 24px;
            font-weight: bold;
            color: #1b5e20;
            display: flex;
            align-items: center;
        }
        
        .issue-card:nth-child(even) .issue-header {
            background-color: #29b6f6;
            color: #01579b;
        }
        
        .issue-icon {
            margin-right: 10px;
            font-size: 24px;
        }
        
        .issue-content {
            padding: 15px;
            font-size: 16px;
        }
        
        .issue-content p {
            margin-bottom: 8px;
            position: relative;
            padding-left: 20px;
        }
        
        .issue-content p::before {
            content: '•';
            position: absolute;
            left: 5px;
            color: #4caf50;
            font-size: 18px;
        }
        
        .issue-card:nth-child(even) .issue-content p::before {
            color: #0288d1;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{INFOGRAPHIC_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #039be5;">
        <i class="fas fa-cloud-sun-rain"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #4caf50;">
        <i class="fas fa-leaf"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #03a9f4;">
        <i class="fas fa-wind"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #00bcd4;">
        <i class="fas fa-temperature-high"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #b3e5fc;"></div>
        <div class="washi-tape green" style="top: 10px; right: 80px; width: 100px; background-color: #b2dfdb;"></div>
        <div class="washi-tape purple" style="bottom: 20px; right: 70px; width: 150px; background-color: #bbdefb;"></div>
        <div class="washi-tape green" style="bottom: 30px; left: 40px; width: 120px; background-color: #c8e6c9;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">{title}</h1>
            <p class="subtitle">Environmental & Social Consequences</p>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="definition-box">
                <h2 class="section-title">What is Climate Change? <i class="fas fa-globe-americas"></i></h2>
                <p class="definition-content">
                    Climate change refers to <span class="highlight-purple">long-term shifts in temperatures and weather patterns</span>. 
                    These shifts may be natural, but since the 1800s, human activities have been the 
                    <span class="highlight-green">main driver of climate change</span>, primarily due to the burning of fossil fuels.
                </p>
            </div>
            
            <!-- Sample chart -->
            <div class="hand-drawn-chart">
                <h3 class="chart-title">Global Temperature Rise (°C) Since 1880</h3>
                <div class="chart-container">
                    <svg width="100%" height="250" style="overflow: visible;">
                        <!-- Hand-drawn axes -->
                        <path d="M 40 220 L 40 30 L 700 30" stroke="#0288d1" stroke-width="3" fill="none" 
                              style="stroke-dasharray: 5,5;" />
                        
                        <!-- Hand-drawn line -->
                        <path d="M 70 200 C 150 190, 250 180, 350 150 S 550 80, 650 40" 
                              stroke="#f44336" stroke-width="4" fill="none" />
                        
                        <!-- Dots on the line -->
                        <circle cx="70" cy="200" r="6" fill="#ffcdd2" stroke="#f44336" stroke-width="2" />
                        <circle cx="210" cy="180" r="6" fill="#ffcdd2" stroke="#f44336" stroke-width="2" />
                        <circle cx="350" cy="150" r="6" fill="#ffcdd2" stroke="#f44336" stroke-width="2" />
                        <circle cx="490" cy="100" r="6" fill="#ffcdd2" stroke="#f44336" stroke-width="2" />
                        <circle cx="650" cy="40" r="6" fill="#ffcdd2" stroke="#f44336" stroke-width="2" />
                        
                        <!-- Y-axis labels -->
                        <text x="30" y="200" text-anchor="end" font-family="Caveat" font-size="16">0.0°</text>
                        <text x="30" y="160" text-anchor="end" font-family="Caveat" font-size="16">0.5°</text>
                        <text x="30" y="120" text-anchor="end" font-family="Caveat" font-size="16">1.0°</text>
                        <text x="30" y="80" text-anchor="end" font-family="Caveat" font-size="16">1.5°</text>
                        <text x="30" y="40" text-anchor="end" font-family="Caveat" font-size="16">2.0°</text>
                        
                        <!-- X-axis labels -->
                        <text x="70" y="240" text-anchor="middle" font-family="Caveat" font-size="16">1880</text>
                        <text x="210" y="240" text-anchor="middle" font-family="Caveat" font-size="16">1920</text>
                        <text x="350" y="240" text-anchor="middle" font-family="Caveat" font-size="16">1960</text>
                        <text x="490" y="240" text-anchor="middle" font-family="Caveat" font-size="16">2000</text>
                        <text x="650" y="240" text-anchor="middle" font-family="Caveat" font-size="16">2020</text>
                    </svg>
                </div>
            </div>
            
            <!-- Hand-drawn arrows -->
            <div class="arrow" style="bottom: 220px; right: 50px; transform: rotate(45deg); color: #0288d1;">
                <i class="fas fa-long-arrow-alt-right"></i>
            </div>
            <div class="arrow" style="top: 300px; left: 30px; transform: rotate(-45deg); color: #0288d1;">
                <i class="fas fa-long-arrow-alt-left"></i>
            </div>
            
            <h2 class="section-title">Key Impacts <i class="fas fa-exclamation-triangle"></i></h2>
            
            <!-- Issues grid -->
            <div class="issues-grid">
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-icon"><i class="fas fa-water"></i></span>
                        Rising Sea Levels
                    </div>
                    <div class="issue-content">
                        <p>Global sea level rose ~8-9 inches since 1880</p>
                        <p><span class="highlight-pink">Rate of rise is accelerating</span> - now 1.3 inches per decade</p>
                        <p>Threatens coastal communities and ecosystems</p>
                        <p>Could displace millions of people by 2050</p>
                    </div>
                </div>
                
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-icon"><i class="fas fa-temperature-high"></i></span>
                        Extreme Weather
                    </div>
                    <div class="issue-content">
                        <p>More frequent and intense heat waves</p>
                        <p><span class="highlight-green">Stronger hurricanes and storms</span> with heavier rainfall</p>
                        <p>Longer and more severe droughts</p>
                        <p>Increased wildfire frequency and intensity</p>
                    </div>
                </div>
                
                <!-- Image example -->
                <div style="grid-column: span 2; text-align: center; margin: 20px 0;">
                    <img src="https://images.unsplash.com/photo-1611273426858-450e7620370d?w=600&h=300&fit=crop&crop=focalpoint&q=80" 
                         alt="Climate Change Effects" class="doodle-image">
                </div>
                
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-icon"><i class="fas fa-biohazard"></i></span>
                        Biodiversity Loss
                    </div>
                    <div class="issue-content">
                        <p><span class="highlight-purple">One million species</span> at risk of extinction</p>
                        <p>Coral reefs severely threatened by ocean warming</p>
                        <p>Ecosystem disruption from changing seasons</p>
                        <p>Shifting wildlife habitats and migration patterns</p>
                    </div>
                </div>
                
                <div class="issue-card">
                    <div class="issue-header">
                        <span class="issue-icon"><i class="fas fa-lightbulb"></i></span>
                        Solutions
                    </div>
                    <div class="issue-content">
                        <p>Transition to <span class="highlight-green">renewable energy sources</span></p>
                        <p>Improve energy efficiency in buildings and transportation</p>
                        <p>Protect and restore forests and wetlands</p>
                        <p>Adopt more sustainable consumption patterns</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Climate Change infographic created at {output_path}")
    
    return output_path

def create_nutrition_stepbystep(output_path):
    """Create a nutrition themed step-by-step guide"""
    title = "Healthy Meal Prep"
    
    # Custom styles for step-by-step guide
    STEPBYSTEP_STYLES = """
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        /* Intro box */
        .intro-box {
            background-color: rgba(255, 204, 188, 0.3);
            border: 2px solid #ffccbc;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .intro-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: #ffccbc;
            border-radius: 10px;
            z-index: -1;
        }
        
        .section-title {
            font-family: 'Caveat', cursive;
            font-size: 32px;
            font-weight: bold;
            color: #d84315;
            margin-bottom: 15px;
            display: inline-block;
            border-bottom: 3px dotted #ff5722;
            padding-bottom: 5px;
        }
        
        /* Steps container */
        .steps-container {
            counter-reset: step-counter;
            margin: 30px 0;
        }
        
        .step-item {
            position: relative;
            background-color: rgba(255, 224, 178, 0.3);
            border-left: 4px solid #ffb74d;
            padding: 20px 20px 20px 70px;
            margin-bottom: 30px;
            border-radius: 0 15px 15px 0;
        }
        
        .step-item:nth-child(even) {
            background-color: rgba(255, 243, 224, 0.3);
            border-left-color: #ffe0b2;
        }
        
        .step-item:before {
            counter-increment: step-counter;
            content: counter(step-counter);
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            background-color: #ff9800;
            color: white;
            font-family: 'Permanent Marker', cursive;
            font-size: 24px;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            text-align: center;
            line-height: 40px;
            box-shadow: 3px 3px 5px rgba(0,0,0,0.1);
        }
        
        .step-item:nth-child(even):before {
            background-color: #fb8c00;
        }
        
        .step-title {
            font-family: 'Kalam', cursive;
            font-size: 22px;
            font-weight: bold;
            color: #e65100;
            margin-bottom: 10px;
        }
        
        .step-content {
            font-size: 18px;
            line-height: 1.6;
        }
        
        /* Tips box */
        .tips-box {
            background-color: rgba(255, 236, 179, 0.3);
            border: 2px dashed #ffca28;
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
            position: relative;
        }
        
        .tips-title {
            font-family: 'Caveat', cursive;
            font-size: 24px;
            font-weight: bold;
            color: #ff6f00;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .tips-title i {
            margin-right: 10px;
            font-size: 20px;
        }
        
        .tips-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .tips-list li {
            margin-bottom: 10px;
            position: relative;
            padding-left: 30px;
            font-size: 16px;
        }
        
        .tips-list li::before {
            content: '✨';
            position: absolute;
            left: 5px;
            color: #ff6f00;
            font-size: 16px;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{STEPBYSTEP_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #ff7043;">
        <i class="fas fa-apple-alt"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #ff9800;">
        <i class="fas fa-carrot"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #ff6d00;">
        <i class="fas fa-utensils"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #f57c00;">
        <i class="fas fa-lemon"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #ffccbc;"></div>
        <div class="washi-tape" style="top: 10px; right: 80px; width: 100px; background-color: #ffe0b2;"></div>
        <div class="washi-tape" style="bottom: 20px; right: 70px; width: 150px; background-color: #ffecb3;"></div>
        <div class="washi-tape" style="bottom: 30px; left: 40px; width: 120px; background-color: #fff8e1;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">{title}</h1>
            <p class="subtitle">A Week of Healthy Meals in One Day</p>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="intro-box">
                <h2 class="section-title">Introduction <i class="fas fa-clock"></i></h2>
                <p style="font-size: 18px; margin-bottom: 20px;">
                    Meal prepping helps you <span class="highlight-purple">save time and money</span> while making it easier 
                    to maintain a <span class="highlight-green">healthy diet</span>. This guide will walk you through 
                    preparing a week's worth of nutritious meals in just a few hours.
                </p>
            </div>
            
            <!-- Image example -->
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=600&h=300&fit=crop&crop=focalpoint&q=80" 
                     alt="Healthy Meal Prep" class="doodle-image">
            </div>
            
            <div class="steps-container">
                <div class="step-item">
                    <h3 class="step-title">Plan Your Menu</h3>
                    <p class="step-content">
                        Start by planning 5-7 meals that share common ingredients. Include a mix of proteins 
                        (chicken, fish, tofu, beans), whole grains (brown rice, quinoa), and plenty of vegetables. 
                        <span class="highlight-pink">Create a detailed shopping list</span> organized by grocery store sections.
                    </p>
                </div>
                
                <div class="step-item">
                    <h3 class="step-title">Shop Efficiently</h3>
                    <p class="step-content">
                        Shop with your list, focusing on the perimeter of the store where fresh foods are located. 
                        <span class="highlight-green">Buy in bulk when possible</span> for items you use frequently. 
                        Consider frozen vegetables as a convenient and nutritious option.
                    </p>
                </div>
                
                <div class="step-item">
                    <h3 class="step-title">Prep Ingredients</h3>
                    <p class="step-content">
                        Wash and chop all vegetables. Cook grains and proteins in batches. Roast a sheet pan of mixed 
                        vegetables with olive oil and simple seasonings. Store prepped ingredients in clear containers 
                        to easily see what you have available.
                    </p>
                </div>
                
                <div class="step-item">
                    <h3 class="step-title">Assemble Meals</h3>
                    <p class="step-content">
                        Use the <span class="highlight-purple">balanced plate method</span>: fill ¼ with protein, ¼ with whole grains, 
                        and ½ with vegetables. Prepare grab-and-go breakfasts like overnight oats or egg muffins. 
                        Store assembled meals in portion-sized containers.
                    </p>
                </div>
                
                <div class="step-item">
                    <h3 class="step-title">Store Properly</h3>
                    <p class="step-content">
                        Label containers with contents and date prepared. Most meals will stay fresh in the refrigerator 
                        for 3-4 days. Freeze extra portions for later in the week. Allow frozen meals to thaw overnight 
                        in the refrigerator before heating.
                    </p>
                </div>
            </div>
            
            <div class="tips-box">
                <h3 class="tips-title"><i class="fas fa-lightbulb"></i> Smart Prep Tips</h3>
                <ul class="tips-list">
                    <li>Use <span class="highlight-pink">one-pot meals</span> like chili, stews, and casseroles to simplify cooking</li>
                    <li>Invest in quality storage containers that are microwave, freezer, and dishwasher safe</li>
                    <li>Cook proteins in different seasonings to avoid meal fatigue</li>
                    <li>Keep fresh herbs, citrus, and spices on hand to quickly change flavors</li>
                    <li>Set aside 2-3 hours on a weekend day for efficient meal prepping</li>
                </ul>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Nutrition step-by-step guide created at {output_path}")
    
    return output_path

def create_history_academic(output_path):
    """Create a history-themed academic notes template"""
    title = "World War II"
    
    # Custom styles for academic notes
    ACADEMIC_STYLES = """
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        /* Topic box */
        .topic-box {
            background-color: rgba(207, 216, 220, 0.3);
            border: 2px solid #cfd8dc;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .topic-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: #cfd8dc;
            border-radius: 10px;
            z-index: -1;
        }
        
        .section-title {
            font-family: 'Caveat', cursive;
            font-size: 32px;
            font-weight: bold;
            color: #37474f;
            margin-bottom: 15px;
            display: inline-block;
            border-bottom: 3px dotted #607d8b;
            padding-bottom: 5px;
        }
        
        /* Notes list */
        .notes-list {
            list-style-type: none;
            padding-left: 0;
        }
        
        .notes-list li {
            margin-bottom: 15px;
            position: relative;
            padding-left: 30px;
            font-size: 18px;
        }
        
        .notes-list li::before {
            content: '✎';
            position: absolute;
            left: 5px;
            color: #546e7a;
            font-size: 18px;
        }
        
        /* Key concepts */
        .key-concepts {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin: 20px 0;
        }
        
        .concept-bubble {
            background-color: rgba(224, 224, 224, 0.5);
            border: 2px solid #e0e0e0;
            border-radius: 20px;
            padding: 10px 20px;
            font-family: 'Caveat', cursive;
            font-size: 18px;
            font-weight: bold;
            color: #455a64;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
            transform: rotate(-2deg);
        }
        
        .concept-bubble:nth-child(even) {
            background-color: rgba(207, 216, 220, 0.5);
            border-color: #cfd8dc;
            color: #263238;
            transform: rotate(2deg);
        }
        
        /* References section */
        .references {
            background-color: rgba(236, 239, 241, 0.3);
            border: 2px dashed #eceff1;
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
        }
        
        .reference-item {
            font-size: 16px;
            margin-bottom: 10px;
            font-style: italic;
        }
        
        /* Timeline */
        .timeline {
            position: relative;
            margin: 30px 0;
            padding-left: 30px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background-color: #78909c;
            border-radius: 2px;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 25px;
            padding-left: 25px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -13px;
            top: 5px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: #90a4ae;
            border: 3px solid #eceff1;
            z-index: 1;
        }
        
        .timeline-date {
            font-family: 'Permanent Marker', cursive;
            font-size: 20px;
            color: #455a64;
            margin-bottom: 5px;
        }
        
        .timeline-content {
            font-size: 16px;
            line-height: 1.5;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{ACADEMIC_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #607d8b;">
        <i class="fas fa-landmark"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #455a64;">
        <i class="fas fa-scroll"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #546e7a;">
        <i class="fas fa-globe-americas"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #37474f;">
        <i class="fas fa-book-open"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #cfd8dc;"></div>
        <div class="washi-tape" style="top: 10px; right: 80px; width: 100px; background-color: #b0bec5;"></div>
        <div class="washi-tape" style="bottom: 20px; right: 70px; width: 150px; background-color: #90a4ae;"></div>
        <div class="washi-tape" style="bottom: 30px; left: 40px; width: 120px; background-color: #78909c;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">{title}</h1>
            <p class="subtitle">Global Conflict (1939-1945)</p>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="topic-box">
                <h2 class="section-title">Overview <i class="fas fa-flag"></i></h2>
                <p style="font-size: 18px; margin-bottom: 20px;">
                    World War II was a <span class="highlight-purple">global conflict</span> that lasted from 1939 to 1945, 
                    involving the majority of the world's nations. It was the most widespread war in history, with more than 
                    <span class="highlight-pink">100 million military personnel mobilized</span> and an estimated 70-85 million fatalities.
                </p>
            </div>
            
            <h2 class="section-title">Key Figures <i class="fas fa-user-tie"></i></h2>
            
            <div class="key-concepts">
                <div class="concept-bubble">Franklin D. Roosevelt (USA)</div>
                <div class="concept-bubble">Winston Churchill (UK)</div>
                <div class="concept-bubble">Joseph Stalin (USSR)</div>
                <div class="concept-bubble">Adolf Hitler (Germany)</div>
                <div class="concept-bubble">Benito Mussolini (Italy)</div>
                <div class="concept-bubble">Hideki Tojo (Japan)</div>
                <div class="concept-bubble">Charles de Gaulle (Free France)</div>
                <div class="concept-bubble">Chiang Kai-shek (China)</div>
            </div>
            
            <!-- Sample chart -->
            <div class="hand-drawn-chart">
                <h3 class="chart-title">Military Deaths by Country (Millions)</h3>
                <div class="chart-container">
                    <svg width="100%" height="250" style="overflow: visible;">
                        <!-- Hand-drawn axes -->
                        <path d="M 40 220 L 40 30 L 700 30" stroke="#607d8b" stroke-width="3" fill="none" 
                              style="stroke-dasharray: 5,5;" />
                        
                        <!-- Hand-drawn bars -->
                        <rect x="70" y="100" width="40" height="120" rx="5" ry="5" fill="#cfd8dc" stroke="#607d8b" stroke-width="2" />
                        <rect x="170" y="50" width="40" height="170" rx="5" ry="5" fill="#b0bec5" stroke="#607d8b" stroke-width="2" />
                        <rect x="270" y="160" width="40" height="60" rx="5" ry="5" fill="#cfd8dc" stroke="#607d8b" stroke-width="2" />
                        <rect x="370" y="175" width="40" height="45" rx="5" ry="5" fill="#b0bec5" stroke="#607d8b" stroke-width="2" />
                        <rect x="470" y="190" width="40" height="30" rx="5" ry="5" fill="#cfd8dc" stroke="#607d8b" stroke-width="2" />
                        <rect x="570" y="205" width="40" height="15" rx="5" ry="5" fill="#b0bec5" stroke="#607d8b" stroke-width="2" />
                        
                        <!-- X-axis labels -->
                        <text x="90" y="240" text-anchor="middle" font-family="Caveat" font-size="16">USSR</text>
                        <text x="190" y="240" text-anchor="middle" font-family="Caveat" font-size="16">China</text>
                        <text x="290" y="240" text-anchor="middle" font-family="Caveat" font-size="16">Germany</text>
                        <text x="390" y="240" text-anchor="middle" font-family="Caveat" font-size="16">Japan</text>
                        <text x="490" y="240" text-anchor="middle" font-family="Caveat" font-size="16">Poland</text>
                        <text x="590" y="240" text-anchor="middle" font-family="Caveat" font-size="16">USA</text>
                        
                        <!-- Y-axis labels -->
                        <text x="30" y="220" text-anchor="end" font-family="Caveat" font-size="16">0</text>
                        <text x="30" y="170" text-anchor="end" font-family="Caveat" font-size="16">2</text>
                        <text x="30" y="120" text-anchor="end" font-family="Caveat" font-size="16">4</text>
                        <text x="30" y="70" text-anchor="end" font-family="Caveat" font-size="16">6</text>
                        <text x="30" y="40" text-anchor="end" font-family="Caveat" font-size="16">8+</text>
                    </svg>
                </div>
            </div>
            
            <h2 class="section-title">Major Events Timeline <i class="fas fa-calendar-alt"></i></h2>
            
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-date">September 1, 1939</div>
                    <div class="timeline-content">
                        <span class="highlight-green">Germany invades Poland</span>, beginning World War II in Europe. 
                        Britain and France declare war on Germany two days later.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">June 22, 1940</div>
                    <div class="timeline-content">
                        France signs an armistice with Germany, allowing Germany to occupy northern France.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">December 7, 1941</div>
                    <div class="timeline-content">
                        <span class="highlight-pink">Japan attacks Pearl Harbor</span>, leading to the United States' 
                        entry into the war. The U.S. declares war on Japan the next day.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">June 6, 1944</div>
                    <div class="timeline-content">
                        <span class="highlight-purple">D-Day landings</span> - Allied forces land on the beaches of Normandy, 
                        beginning the liberation of Western Europe from Nazi control.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">May 8, 1945</div>
                    <div class="timeline-content">
                        <span class="highlight-green">Victory in Europe Day (V-E Day)</span> - Germany surrenders unconditionally, 
                        ending the war in Europe.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">August 6 & 9, 1945</div>
                    <div class="timeline-content">
                        The United States drops atomic bombs on <span class="highlight-pink">Hiroshima and Nagasaki</span>.
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">September 2, 1945</div>
                    <div class="timeline-content">
                        <span class="highlight-purple">Japan formally surrenders</span>, bringing World War II to an end.
                    </div>
                </div>
            </div>
            
            <!-- Image example -->
            <div style="text-align: center; margin: 30px 0;">
                <img src="https://images.unsplash.com/photo-1580477667995-2b94f01c9516?w=600&h=300&fit=crop&crop=focalpoint&q=80" 
                     alt="World War II Map" class="doodle-image">
            </div>
            
            <div class="references">
                <h3 class="section-title" style="font-size: 24px;">References <i class="fas fa-book"></i></h3>
                <p class="reference-item">Beevor, A. (2012). The Second World War. Little, Brown and Company.</p>
                <p class="reference-item">Churchill, W. (1948-1953). The Second World War (6 vols).</p>
                <p class="reference-item">Gilbert, M. (1989). Second World War. Henry Holt and Company.</p>
                <p class="reference-item">Keegan, J. (1989). The Second World War. Viking Press.</p>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"History academic notes created at {output_path}")
    
    return output_path

def create_ai_cornell(output_path):
    """Create an AI-themed Cornell notes template"""
    title = "Artificial Intelligence"
    
    # Custom styles for Cornell notes
    CORNELL_STYLES = """
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        /* Topic box */
        .topic-box {
            background-color: rgba(197, 202, 233, 0.3);
            border: 2px solid #c5cae9;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .topic-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: #c5cae9;
            border-radius: 10px;
            z-index: -1;
        }
        
        .section-title {
            font-family: 'Caveat', cursive;
            font-size: 32px;
            font-weight: bold;
            color: #303f9f;
            margin-bottom: 15px;
            display: inline-block;
            border-bottom: 3px dotted #3949ab;
            padding-bottom: 5px;
        }
        
        /* Cornell notes layout */
        .cornell-container {
            display: grid;
            grid-template-columns: 30% 70%;
            grid-template-rows: auto 1fr auto;
            grid-template-areas:
                "header header"
                "cues notes"
                "summary summary";
            gap: 15px;
            margin-top: 30px;
            border-radius: 15px;
            overflow: hidden;
        }
        
        .cornell-header {
            grid-area: header;
            background-color: rgba(121, 134, 203, 0.2);
            padding: 15px;
            border-bottom: 2px dashed #7986cb;
            font-family: 'Kalam', cursive;
            font-size: 18px;
            display: flex;
            justify-content: space-between;
        }
        
        .cornell-cues {
            grid-area: cues;
            background-color: rgba(197, 202, 233, 0.3);
            padding: 15px;
            border-right: 2px dashed #9fa8da;
        }
        
        .cornell-notes {
            grid-area: notes;
            background-color: rgba(232, 234, 246, 0.2);
            padding: 15px;
        }
        
        .cornell-summary {
            grid-area: summary;
            background-color: rgba(121, 134, 203, 0.2);
            padding: 15px;
            border-top: 2px dashed #7986cb;
        }
        
        /* Section headings */
        .cornell-section-title {
            font-family: 'Caveat', cursive;
            font-size: 22px;
            font-weight: bold;
            color: #3949ab;
            margin-bottom: 10px;
            text-align: center;
        }
        
        /* Key questions */
        .key-question {
            margin-bottom: 15px;
            font-family: 'Kalam', cursive;
            font-size: 16px;
            color: #283593;
            position: relative;
            padding-left: 25px;
        }
        
        .key-question::before {
            content: '?';
            position: absolute;
            left: 5px;
            top: 0;
            font-family: 'Permanent Marker', cursive;
            font-size: 18px;
            color: #5c6bc0;
        }
        
        /* Notes content */
        .notes-paragraph {
            margin-bottom: 15px;
            font-size: 16px;
            line-height: 1.6;
        }
        
        /* Summary box */
        .summary-content {
            font-size: 16px;
            font-style: italic;
            line-height: 1.6;
        }
        
        /* Code blocks */
        .code-block {
            background-color: rgba(232, 234, 246, 0.7);
            border-left: 4px solid #7986cb;
            padding: 12px;
            margin: 15px 0;
            font-family: monospace;
            font-size: 14px;
            border-radius: 0 5px 5px 0;
            white-space: pre;
            overflow-x: auto;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{CORNELL_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #5c6bc0;">
        <i class="fas fa-robot"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #3949ab;">
        <i class="fas fa-brain"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #7986cb;">
        <i class="fas fa-microchip"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #3f51b5;">
        <i class="fas fa-cogs"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #c5cae9;"></div>
        <div class="washi-tape" style="top: 10px; right: 80px; width: 100px; background-color: #9fa8da;"></div>
        <div class="washi-tape" style="bottom: 20px; right: 70px; width: 150px; background-color: #7986cb;"></div>
        <div class="washi-tape" style="bottom: 30px; left: 40px; width: 120px; background-color: #5c6bc0;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">{title}</h1>
            <p class="subtitle">Machine Learning Foundations</p>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="topic-box">
                <h2 class="section-title">Introduction to AI <i class="fas fa-laptop-code"></i></h2>
                <p style="font-size: 18px; margin-bottom: 20px;">
                    Artificial Intelligence (AI) is the <span class="highlight-purple">simulation of human intelligence processes</span> 
                    by machines. The field encompasses machine learning, deep learning, natural language processing, 
                    computer vision, and <span class="highlight-pink">autonomous systems</span>.
                </p>
            </div>
            
            <!-- Image example -->
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=600&h=300&fit=crop&crop=focalpoint&q=80" 
                     alt="AI Machine Learning Diagram" class="doodle-image">
            </div>
            
            <div class="cornell-container">
                <div class="cornell-header">
                    <div><strong>Topic:</strong> Machine Learning Basics</div>
                    <div><strong>Date:</strong> November 12, 2023</div>
                </div>
                
                <div class="cornell-cues">
                    <h3 class="cornell-section-title">Key Questions</h3>
                    <div class="key-question">What is machine learning?</div>
                    <div class="key-question">What are the main types of ML?</div>
                    <div class="key-question">How does supervised learning work?</div>
                    <div class="key-question">What is a neural network?</div>
                    <div class="key-question">What are common ML challenges?</div>
                    <div class="key-question">What is the difference between AI and ML?</div>
                </div>
                
                <div class="cornell-notes">
                    <h3 class="cornell-section-title">Notes</h3>
                    <p class="notes-paragraph">
                        <span class="highlight-green">Machine Learning (ML)</span> is a subset of AI that enables systems to learn 
                        and improve from experience without being explicitly programmed. ML algorithms build mathematical 
                        models based on sample data (training data) to make predictions or decisions.
                    </p>
                    
                    <p class="notes-paragraph">
                        <span class="highlight-pink">Three main types of machine learning:</span>
                        <br>• <strong>Supervised Learning:</strong> Algorithms learn from labeled training data
                        <br>• <strong>Unsupervised Learning:</strong> Algorithms find patterns in unlabeled data
                        <br>• <strong>Reinforcement Learning:</strong> Algorithms learn through trial and error with rewards/penalties
                    </p>
                    
                    <p class="notes-paragraph">
                        <span class="highlight-purple">Neural networks</span> are computing systems inspired by biological neural networks. 
                        They consist of layers of interconnected nodes (neurons) that process information. Deep learning 
                        uses neural networks with many layers (deep neural networks).
                    </p>
                    
                    <div class="code-block">
# Simple neural network in Python using TensorFlow
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(features,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)</div>
                    
                    <p class="notes-paragraph">
                        Common ML challenges include overfitting (model performs well on training data but poorly on new data), 
                        underfitting (model is too simple to capture patterns), and data quality issues (insufficient, 
                        biased, or noisy data).
                    </p>
                </div>
                
                <div class="cornell-summary">
                    <h3 class="cornell-section-title">Summary</h3>
                    <p class="summary-content">
                        Machine learning is a subset of AI focused on building systems that learn from data. The three main 
                        approaches are supervised learning (using labeled data), unsupervised learning (finding patterns in 
                        unlabeled data), and reinforcement learning (learning through interaction). Neural networks, 
                        especially deep neural networks, have revolutionized the field. Effective ML requires quality data, 
                        appropriate algorithms, and careful management of challenges like overfitting.
                    </p>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"AI Cornell notes created at {output_path}")
    
    return output_path

def create_fashion_infographic(output_path):
    """Create a fast fashion themed infographic with pink and green styling"""
    title = "Fast Fashion Impact"
    
    # Custom styles for fashion infographic
    FASHION_STYLES = """
        /* Pink and green color scheme */
        :root {
            --pink-light: #FFC0CB;
            --pink-medium: #FF69B4;
            --pink-dark: #C71585;
            --green-light: #CCFF99;
            --green-medium: #99CC66;
            --green-dark: #669933;
        }
        
        .notebook-page {
            background-color: #FFF0F5;
        }
        
        /* Title styling */
        .main-title {
            color: #333;
            font-size: 48px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        .subtitle {
            font-family: 'Caveat', cursive;
            font-size: 42px;
            font-style: italic;
            color: #444;
            margin-top: 0;
        }
        
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        /* Definition box */
        .definition-box {
            background-color: var(--pink-light);
            border: 2px solid var(--pink-medium);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .definition-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: var(--pink-medium);
            border-radius: 10px;
            z-index: -1;
        }
        
        /* Problem section */
        .problem-section {
            background-color: var(--green-light);
            border-top: 10px solid var(--pink-medium);
            border-bottom: 10px solid var(--pink-medium);
            padding: 20px 0;
            margin: 40px 0;
            position: relative;
        }
        
        .problem-title {
            font-family: 'Permanent Marker', cursive;
            font-size: 32px;
            color: #333;
            text-align: center;
            background-color: var(--pink-light);
            border: 3px solid var(--pink-medium);
            border-radius: 50px;
            padding: 10px 30px;
            width: fit-content;
            margin: 0 auto 30px auto;
            box-shadow: 5px 5px 10px rgba(0,0,0,0.1);
        }
        
        /* Issue cards */
        .issues-container {
            counter-reset: issue-counter;
        }
        
        .issue-item {
            position: relative;
            margin-bottom: 30px;
            padding-left: 80px;
        }
        
        .issue-item:before {
            counter-increment: issue-counter;
            content: counter(issue-counter);
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            background-color: white;
            color: var(--pink-dark);
            border: 3px solid var(--pink-medium);
            font-family: 'Permanent Marker', cursive;
            font-size: 24px;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            text-align: center;
            line-height: 50px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
        }
        
        .issue-content {
            background-color: white;
            border: 2px solid var(--green-medium);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        }
        
        .issue-title {
            font-family: 'Kalam', cursive;
            font-size: 26px;
            font-weight: bold;
            color: var(--green-dark);
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 2px dotted var(--green-medium);
            display: inline-block;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{FASHION_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #FF69B4;">
        <i class="fas fa-tshirt"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #99CC66;">
        <i class="fas fa-recycle"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #FF69B4;">
        <i class="fas fa-shopping-bag"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #99CC66;">
        <i class="fas fa-leaf"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #FF69B4;"></div>
        <div class="washi-tape" style="top: 10px; right: 80px; width: 100px; background-color: #99CC66;"></div>
        <div class="washi-tape" style="bottom: 20px; right: 70px; width: 150px; background-color: #FF69B4;"></div>
        <div class="washi-tape" style="bottom: 30px; left: 40px; width: 120px; background-color: #99CC66;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">WHY</h1>
            <p class="subtitle">fast fashion</p>
            <h1 class="main-title">IS AN ISSUE?</h1>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="definition-box">
                <h2 class="section-title">WHAT IS FAST FASHION? <i class="fas fa-question-circle"></i></h2>
                <p style="font-size: 18px; line-height: 1.6;">
                    Fast fashion describes affordable, stylish, mass-produced clothes that hugely impact the environment.
                    These garments quickly move from design to stores to follow the latest trends, but are often worn just a few times 
                    before being discarded.
                </p>
            </div>
            
            <!-- Image example -->
            <div style="text-align: center; margin: 20px 0;">
                <img src="https://images.unsplash.com/photo-1567401893414-76b7b1e5a7a5?w=600&h=300&fit=crop&crop=focalpoint&q=80" 
                     alt="Fast Fashion Clothing Rack" class="doodle-image">
            </div>
            
            <div class="problem-section">
                <h2 class="problem-title">WHY IS IT A PROBLEM?</h2>
                
                <div class="issues-container">
                    <div class="issue-item">
                        <div class="issue-content">
                            <h3 class="issue-title">TEXTILE WASTE</h3>
                            <p style="font-size: 18px; line-height: 1.6;">
                                Clothing has lower recycling rates than paper, glass, and plastic because short-fiber fabrics are cheaper,
                                thinner, and can't be rewoven. The average consumer throws away 70 pounds (32kg) of clothing per year,
                                with over 85% ending up in landfills or incinerators.
                            </p>
                        </div>
                    </div>
                    
                    <div class="issue-item">
                        <div class="issue-content">
                            <h3 class="issue-title">CO2 EMISSIONS</h3>
                            <p style="font-size: 18px; line-height: 1.6;">
                                Carbon emissions happen during transportation from factories to stores, consumer purchases, 
                                and when products are disposed of. The fashion industry is responsible for 10% of annual global 
                                carbon emissions – more than international flights and maritime shipping combined.
                            </p>
                        </div>
                    </div>
                    
                    <div class="issue-item">
                        <div class="issue-content">
                            <h3 class="issue-title">WATER POLLUTION</h3>
                            <p style="font-size: 18px; line-height: 1.6;">
                                Synthetic fabrics, even those made from recycled water bottles, can contain microplastics,
                                despite being seen as eco-friendly. Every time these clothes are washed, they release microfibers 
                                that eventually reach oceans and harm marine life.
                            </p>
                        </div>
                    </div>
                    
                    <div class="issue-item">
                        <div class="issue-content">
                            <h3 class="issue-title">LABOR CONCERNS</h3>
                            <p style="font-size: 18px; line-height: 1.6;">
                                Many fast fashion companies outsource production to countries with low labor costs and minimal regulations.
                                Workers often face unsafe conditions, excessively long hours, and wages below living standards
                                to keep production costs low and profits high.
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="background-color: #FFC0CB; border-radius: 15px; padding: 20px; margin-top: 30px;">
                <h2 class="section-title">WHAT CAN YOU DO? <i class="fas fa-lightbulb"></i></h2>
                <ul style="font-size: 18px; line-height: 1.8; padding-left: 20px;">
                    <li>Buy fewer, higher-quality clothes that last longer</li>
                    <li>Choose second-hand or vintage clothing when possible</li>
                    <li>Support sustainable and ethical fashion brands</li>
                    <li>Repair clothes instead of replacing them</li>
                    <li>Donate or recycle unwanted garments properly</li>
                </ul>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Fast Fashion infographic created at {output_path}")
    
    return output_path

def create_project_notes(output_path):
    """Create a project notes template in a visually engaging style"""
    title = "Project Planning Notes"
    
    # Custom styles for project notes
    PROJECT_STYLES = """
        /* Content sections */
        .content-section {
            position: relative;
            margin-bottom: 35px;
            z-index: 5;
        }
        
        /* Project box */
        .project-box {
            background-color: rgba(179, 157, 219, 0.3);
            border: 2px solid #b39ddb;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
            position: relative;
        }
        
        .project-box::before {
            content: '';
            position: absolute;
            top: -10px;
            left: 20px;
            width: 180px;
            height: 20px;
            background-color: #b39ddb;
            border-radius: 10px;
            z-index: -1;
        }
        
        .section-title {
            font-family: 'Caveat', cursive;
            font-size: 32px;
            font-weight: bold;
            color: #5e35b1;
            margin-bottom: 15px;
            display: inline-block;
            border-bottom: 3px dotted #7e57c2;
            padding-bottom: 5px;
        }
        
        /* Tasks section */
        .tasks-container {
            margin: 30px 0;
        }
        
        .task-item {
            display: flex;
            margin-bottom: 15px;
            align-items: flex-start;
        }
        
        .task-checkbox {
            width: 24px;
            height: 24px;
            border: 2px solid #7e57c2;
            border-radius: 5px;
            margin-right: 15px;
            flex-shrink: 0;
            position: relative;
        }
        
        .task-content {
            flex-grow: 1;
        }
        
        .task-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
            color: #5e35b1;
        }
        
        .task-description {
            font-size: 16px;
            margin-bottom: 5px;
        }
        
        .task-meta {
            display: flex;
            font-size: 14px;
            color: #7e57c2;
        }
        
        .task-due {
            margin-right: 20px;
        }
        
        .task-priority {
            font-weight: bold;
        }
        
        .task-priority.high {
            color: #d32f2f;
        }
        
        .task-priority.medium {
            color: #ff9800;
        }
        
        .task-priority.low {
            color: #4caf50;
        }
        
        /* Timeline section */
        .timeline {
            position: relative;
            margin: 30px 0;
            padding-left: 30px;
        }
        
        .timeline::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            background-color: #9575cd;
            border-radius: 2px;
        }
        
        .timeline-item {
            position: relative;
            margin-bottom: 25px;
            padding-left: 25px;
        }
        
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -13px;
            top: 5px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: #b39ddb;
            border: 3px solid #ede7f6;
            z-index: 1;
        }
        
        .timeline-date {
            font-family: 'Permanent Marker', cursive;
            font-size: 18px;
            color: #5e35b1;
            margin-bottom: 5px;
        }
        
        .timeline-content {
            font-size: 16px;
            line-height: 1.5;
        }
        
        /* Notes section */
        .notes-section {
            background-color: rgba(209, 196, 233, 0.3);
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
        }
        
        .notes-content {
            font-family: 'Kalam', cursive;
            font-size: 18px;
            line-height: 1.6;
            color: #333;
        }
    """
    
    # Combine the styles
    html_header = get_common_html_header(title)
    html_header = html_header.replace("</style>", f"{PROJECT_STYLES}</style>")
    
    # Create the HTML content
    html_content = f"""{html_header}
</head>
<body>
    <!-- Print Button -->
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i>
    </button>

    <!-- Decorative doodles -->
    <div class="doodle" style="top: 10px; right: 20px; font-size: 28px; transform: rotate(15deg); color: #7e57c2;">
        <i class="fas fa-tasks"></i>
    </div>
    <div class="doodle" style="bottom: 40px; left: 30px; font-size: 32px; transform: rotate(-10deg); color: #5e35b1;">
        <i class="fas fa-calendar-alt"></i>
    </div>
    <div class="doodle" style="top: 120px; left: 10px; font-size: 24px; transform: rotate(-5deg); color: #9575cd;">
        <i class="fas fa-lightbulb"></i>
    </div>
    <div class="doodle" style="bottom: 100px; right: 40px; font-size: 30px; transform: rotate(8deg); color: #673ab7;">
        <i class="fas fa-clipboard-check"></i>
    </div>

    <!-- Main notebook page -->
    <div class="notebook-page">
        <!-- Washi tape decorations -->
        <div class="washi-tape" style="top: 15px; left: 50px; width: 120px; background-color: #d1c4e9;"></div>
        <div class="washi-tape" style="top: 10px; right: 80px; width: 100px; background-color: #b39ddb;"></div>
        <div class="washi-tape" style="bottom: 20px; right: 70px; width: 150px; background-color: #9575cd;"></div>
        <div class="washi-tape" style="bottom: 30px; left: 40px; width: 120px; background-color: #7e57c2;"></div>
        
        <!-- Paper clip -->
        <div class="paper-clip">
            <i class="fas fa-paperclip"></i>
        </div>
        
        <!-- Title section -->
        <div class="title-area">
            <h1 class="main-title">{title}</h1>
            <p class="subtitle">Research & Development Roadmap</p>
        </div>
        
        <!-- Content -->
        <div class="content-section">
            <div class="project-box">
                <h2 class="section-title">Project Overview <i class="fas fa-project-diagram"></i></h2>
                <p style="font-size: 18px; margin-bottom: 20px;">
                    This <span class="highlight-purple">research project</span> aims to develop a new methodology for 
                    analyzing environmental impact of consumer products. The approach combines <span class="highlight-pink">data science</span> 
                    techniques with traditional lifecycle assessment to create more accurate sustainability metrics.
                </p>
            </div>
            
            <h2 class="section-title">Key Objectives <i class="fas fa-bullseye"></i></h2>
            
            <div class="tasks-container">
                <div class="task-item">
                    <div class="task-checkbox"></div>
                    <div class="task-content">
                        <div class="task-title">Literature Review</div>
                        <div class="task-description">Complete comprehensive review of existing methodologies and identify gaps</div>
                        <div class="task-meta">
                            <div class="task-due">Due: June 15, 2023</div>
                            <div class="task-priority high">Priority: High</div>
                        </div>
                    </div>
                </div>
                
                <div class="task-item">
                    <div class="task-checkbox"></div>
                    <div class="task-content">
                        <div class="task-title">Data Collection Framework</div>
                        <div class="task-description">Design protocol for gathering product lifecycle data across multiple categories</div>
                        <div class="task-meta">
                            <div class="task-due">Due: July 10, 2023</div>
                            <div class="task-priority high">Priority: High</div>
                        </div>
                    </div>
                </div>
                
                <div class="task-item">
                    <div class="task-checkbox"></div>
                    <div class="task-content">
                        <div class="task-title">Develop Initial Algorithm</div>
                        <div class="task-description">Create first version of impact assessment algorithm using Python/R</div>
                        <div class="task-meta">
                            <div class="task-due">Due: August 5, 2023</div>
                            <div class="task-priority medium">Priority: Medium</div>
                        </div>
                    </div>
                </div>
                
                <div class="task-item">
                    <div class="task-checkbox"></div>
                    <div class="task-content">
                        <div class="task-title">Pilot Testing</div>
                        <div class="task-description">Test methodology on sample products from three different categories</div>
                        <div class="task-meta">
                            <div class="task-due">Due: September 20, 2023</div>
                            <div class="task-priority medium">Priority: Medium</div>
                        </div>
                    </div>
                </div>
                
                <div class="task-item">
                    <div class="task-checkbox"></div>
                    <div class="task-content">
                        <div class="task-title">Draft Initial Findings</div>
                        <div class="task-description">Prepare draft paper with methodology and preliminary results</div>
                        <div class="task-meta">
                            <div class="task-due">Due: October 15, 2023</div>
                            <div class="task-priority low">Priority: Low</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <h2 class="section-title">Project Timeline <i class="fas fa-clock"></i></h2>
            
            <div class="timeline">
                <div class="timeline-item">
                    <div class="timeline-date">May 1, 2023</div>
                    <div class="timeline-content">
                        <span class="highlight-purple">Project kickoff</span> - Initial team meeting and role assignments
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">July 15, 2023</div>
                    <div class="timeline-content">
                        Phase 1 completion: Literature review and data collection framework finalized
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">September 30, 2023</div>
                    <div class="timeline-content">
                        <span class="highlight-pink">Phase 2 completion</span>: Algorithm development and pilot testing
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">November 15, 2023</div>
                    <div class="timeline-content">
                        Phase 3 completion: Draft paper and validation of results
                    </div>
                </div>
                
                <div class="timeline-item">
                    <div class="timeline-date">December 31, 2023</div>
                    <div class="timeline-content">
                        <span class="highlight-green">Project completion</span> - Final paper submission and presentation
                    </div>
                </div>
            </div>
            
            <div class="notes-section">
                <h3 class="section-title" style="font-size: 24px;">Meeting Notes <i class="fas fa-comment-alt"></i></h3>
                <div class="notes-content">
                    <p>Kickoff meeting (May 1):</p>
                    <p>- Team expressed concerns about data availability for certain product categories</p>
                    <p>- Need to research existing databases that could supplement our primary data collection</p>
                    <p>- Consider reaching out to Dr. Martinez for expertise on lifecycle assessment methods</p>
                    <p>- Budget adjustment needed if we want to include consumer testing phase</p>
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>Created with ♥ by Notely • Download as PDF using the print button</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write the HTML content to the file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Project notes template created at {output_path}")
    
    return output_path

def showcase_templates():
    """Create and display different template examples"""
    print("\n===== SHOWCASING NOTELY TEMPLATES WITH DIFFERENT CONTENT =====")
    
    # Create output directory
    output_dir = 'showcase'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create climate change infographic
    print("\nGenerating Climate Change infographic...")
    climate_path = create_climate_change_infographic(f"{output_dir}/climate_infographic.html")
    
    # Create nutrition step-by-step guide
    print("\nGenerating Nutrition step-by-step guide...")
    nutrition_path = create_nutrition_stepbystep(f"{output_dir}/nutrition_stepbystep.html")
    
    # Create history academic notes
    print("\nGenerating History academic notes...")
    history_path = create_history_academic(f"{output_dir}/history_academic.html")
    
    # Create AI Cornell notes
    print("\nGenerating AI Cornell notes...")
    ai_path = create_ai_cornell(f"{output_dir}/ai_cornell.html")
    
    # Create Fashion infographic
    print("\nGenerating Fast Fashion infographic...")
    fashion_path = create_fashion_infographic(f"{output_dir}/fashion_infographic.html")
    
    # Create Project notes
    print("\nGenerating Project Planning notes...")
    project_path = create_project_notes(f"{output_dir}/project_notes.html")
    
    # Show all templates
    print(f"\nTemplate showcases created in the {output_dir} directory:")
    print(f"  - Climate Change Infographic: {climate_path}")
    print(f"  - Nutrition Step-by-Step Guide: {nutrition_path}")
    print(f"  - History Academic Notes: {history_path}")
    print(f"  - AI Cornell Notes: {ai_path}")
    print(f"  - Fast Fashion Infographic: {fashion_path}")
    print(f"  - Project Planning Notes: {project_path}")
    
    # Open each template in the browser
    print("\nOpening templates in browser...")
    open_in_browser(climate_path)
    open_in_browser(nutrition_path)
    open_in_browser(history_path)
    open_in_browser(ai_path)
    open_in_browser(fashion_path)
    open_in_browser(project_path)
    
    print("\nINSTRUCTIONS:")
    print("1. All templates are now open in your browser")
    print("2. Each template has a different theme and content")
    print("3. Use the print button (bottom right) to save as PDF")

if __name__ == "__main__":
    showcase_templates() 