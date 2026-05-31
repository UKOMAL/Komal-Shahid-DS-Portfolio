# DSC680 Project 2: AI-Driven Content Strategy Engine

## Project Overview
Statistical analysis of 12,000 Reddit posts to identify what content variables — topic, format, and hook style — drive engagement. OLS regression and gradient boosting models quantify the lift from AI-conditioned content, with ANOVA confirming significance across all three dimensions.

## Course Information
- **Course**: DSC680 - Applied Data Science
- **Institution**: Bellevue University
- **Student**: Komal Shahid

## Key Results
- OLS + Gradient Boost: R² = 0.271 on 12K Reddit posts
- ANOVA: topic, format, and hook type all significant (p < .001)
- AI-conditioned content: **+37% predicted engagement lift** (95% CI: [+15%, +64%])

## Repository Structure
```
project2-content-strategy-engine/
├── code/                          # Analysis pipeline
│   ├── shahid_dsc680_project2_analysis.py
│   ├── shahid_dsc680_project2_data_collection.py
│   ├── shahid_dsc680_project2_features.py
│   ├── shahid_dsc680_project2_synthetic_corpus.py
│   ├── ai_detect.py
│   └── requirements.txt
├── figures/                       # Output visualizations
│   ├── fig01_topic_format_heatmap.png
│   ├── fig02_ols_coefficients.png
│   ├── fig03_permutation_importance.png
│   ├── fig04_trends_lead_lag.png
│   └── fig05_anova_topic.png
├── milestone1_proposal/           # Project proposal
├── milestone2_whitepaper/         # Research white paper
├── milestone3_final/              # Final deliverables
│   ├── shahid_dsc680_project2_notebook_FINAL.ipynb
│   ├── shahid_dsc680_project2_milestone3_whitepaper_FINAL.pdf
│   ├── shahid_dsc680_project2_milestone3_presentation_FINAL.pptx
│   ├── shahid_dsc680_project2_infographic.html
│   └── shahid_dsc680_project2_milestone3_qa.pdf
└── results/
    ├── corpus_sample.csv
    └── m2_results.json
```

## Methods
- **Data**: 12,000 Reddit posts with engagement metrics (upvotes, comments, shares)
- **Feature Engineering**: Topic classification, format tagging, hook-style extraction
- **Models**: OLS regression, Gradient Boosting (permutation importance)
- **Inference**: One-way ANOVA per content dimension, Tukey HSD post-hoc

## Deliverables
| Milestone | Deliverable |
|---|---|
| M1 | Research proposal |
| M2 | White paper (literature review + methodology) |
| M3 | Final notebook, white paper, 13-slide presentation, Q&A, infographic |
