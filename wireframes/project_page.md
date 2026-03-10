# Wireframe: Projects Index Page
**Route:** `/projects`
**Components:** Nav · ProjectCard × N · Badge (filters) · MetricPill

---

## Desktop Layout (1200px)

```
┌─────────────────────────────────────────────────────────────────────┐
│  NAV (sticky)                                                        │
└─────────────────────────────────────────────────────────────────────┘
│                   PAGE HEADER                                         │
│  [H1] Projects                                                       │
│  [subtitle] End-to-end AI/ML systems with reproducible artifacts    │
│                                                                       │
│  ─── FILTER TAGS ──────────────────────────────────────────────────── │
│  All  |  NLP  |  Fraud  |  Federated  |  Healthcare  |  MLOps       │
│  (active filter shown with filled accent bg)                          │
│                                                                       │
│  ─── PROJECT GRID (3 columns, desktop) ─────────────────────────────  │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐  │
│  │  ★ FEATURED         │  │  ★ FEATURED         │  │ ★ FEATURED   │  │
│  │  [thumbnail]        │  │  [thumbnail]        │  │ [thumbnail]  │  │
│  │  Fraud Detection    │  │  Depression NLP     │  │ Federated AI │  │
│  │  AUC 0.886          │  │  Accuracy 91%       │  │ ε = 1.0 DP   │  │
│  │  [LightGBM][SHAP]   │  │  [DistilBERT][NLP]  │  │ [Flower][DP] │  │
│  │  ─── hover ───      │  │                     │  │              │  │
│  │  • Bullet 1         │  │                     │  │              │  │
│  │  • Bullet 2         │  │                     │  │              │  │
│  │  • Bullet 3         │  │                     │  │              │  │
│  │  [View CS] [Demo]   │  │  [View CS] [Demo]   │  │[View CS][Demo│  │
│  └─────────────────────┘  └─────────────────────┘  └──────────────┘  │
│                                                                       │
│  ─── OTHER PROJECTS (2 columns) ────────────────────────────────────  │
│  ┌──────────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Data Visualisation       │  │ Predictive Analytics            │   │
│  │ [DSC640 badge]           │  │ [DSC670 badge]                  │   │
│  │ [View Project]           │  │ [View Project]                  │   │
│  └──────────────────────────┘  └─────────────────────────────────┘   │
│  ┌──────────────────────────┐  ┌─────────────────────────────────┐   │
│  │ Text Mining (DSC550)     │  │ Statistical Computing (DSC520)  │   │
│  │ ...                      │  │ ...                             │   │
│  └──────────────────────────┘  └─────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Mobile Layout (375px)

```
┌────────────────────────────────────┐
│ NAV [sticky]                       │
└────────────────────────────────────┘
│  [H1] Projects                     │
│  [subtitle — 1–2 lines]            │
│                                    │
│  ─── FILTER TAGS (horizontal scroll)│
│  All | NLP | Fraud | Federated ...  │
│                                    │
│  ─── PROJECT CARDS (1 column) ──── │
│  ┌──────────────────────────────┐  │
│  │ [thumbnail]                  │  │
│  │ Fraud Detection — AUC 0.886  │  │
│  │ [LightGBM][SHAP][Ethical AI] │  │
│  │ [View Case Study] [Run Demo] │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Depression Detection — 91%   │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Federated Healthcare — ε≤1.0 │  │
│  └──────────────────────────────┘  │
│  ... (other projects below)        │
└────────────────────────────────────┘
```

---

## Component Placement Rules

### Desktop
- **Page header:** max-width 820px, centred, 64px top padding
- **Filter tags:** Horizontal row, gap 8px, sticky below nav on scroll (`position: sticky; top: 64px`)
- **Featured projects:** 3-column grid, `minmax(300px, 1fr)`, gap 24px
- **Other projects:** 2-column grid, `minmax(400px, 1fr)`, gap 16px, lighter card style (no border)
- **Hover bullets:** Reveal on `ProjectCard:hover` and `:focus-within`

### Mobile
- **Filter tags:** `overflow-x: auto; white-space: nowrap` row
- **Project cards:** Full width, stacked
- **Card height:** Auto (no fixed height — content-driven)
