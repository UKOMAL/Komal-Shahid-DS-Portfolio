# Wireframe: Case Study Page
**Route:** `/projects/[slug]`
**Components:** Nav · CaseStudyLayout · MetricPill × 3 · Badge · CTAButton · Collapsible Repro

---

## Desktop Layout (1200px)

```
┌─────────────────────────────────────────────────────────────────────┐
│  NAV (sticky)                                                        │
└─────────────────────────────────────────────────────────────────────┘
│                    CASE STUDY HEADER (dark bg)                       │
│              ┌──────────────────────────────────────┐                │
│              │  [H1] Fraud Detection System         │                │
│              │  [subtitle] Detecting fraud in 800K+ │                │
│              │                                       │                │
│              │  [AUC-ROC 0.886 ▲+0.11 ⓘ]           │                │
│              │  [Transactions 800K+ ⓘ]              │ MetricPill strip│
│              │  [Precision@K 0.74 ▲+0.18 ⓘ]        │                │
│              │                                       │                │
│              │  [LightGBM][SHAP][Ethical AI][...]   │ Badge row       │
│              │                                       │                │
│              │  [View CS ↓] [Run Demo ▶]            │                │
│              │  [Download Report ↓] [GitHub →]      │ CTA strip       │
│              └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
│                                                                       │
│  STICKY SECTION NAV (scrolls with page)                              │
│  Problem | Dataset | Approach | Results | Fairness | Demo | Repro   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │  BODY CONTENT (max-width 820px, centred)                    │     │
│  │                                                              │     │
│  │  ## Problem {#problem}                                       │     │
│  │  [content block]                                             │     │
│  │                                                              │     │
│  │  ## Dataset {#dataset}                                       │     │
│  │  [table: source | size | fraud rate | split]                 │     │
│  │                                                              │     │
│  │  ## Approach {#approach}                                     │     │
│  │  [code block — model architecture]                           │     │
│  │                                                              │     │
│  │  ## Results {#results}                                       │     │
│  │  [table: metric | baseline | model]                          │     │
│  │                                                              │     │
│  │  ## Fairness & Interpretability {#fairness}                  │     │
│  │  [content block]                                             │     │
│  │                                                              │     │
│  │  ## Demo {#demo}                                             │     │
│  │  [code block: curl / docker command]                         │     │
│  │  [sample JSON input + output]                                │     │
│  │  [Binder badge]                                              │     │
│  │                                                              │     │
│  │  ## Lessons Learned {#lessons}                               │     │
│  │  [numbered list]                                             │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  ▼ REPRODUCIBILITY & ARTIFACTS  [click to expand]  ← toggle │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  (collapsed by default)                                               │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  Checklist:                                                   │    │
│  │  ☑ Seed control  ☑ Environment  ☑ Data provenance            │    │
│  │                                                               │    │
│  │  | Artifact          | Link      |                           │    │
│  │  | Jupyter notebooks | [link]    |                           │    │
│  │  | Source code       | [link]    |                           │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  (expanded on click)                                                  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Mobile Layout (375px)

```
┌────────────────────────────────────┐
│ NAV [sticky]                       │
└────────────────────────────────────┘
│  HEADER (dark bg)                  │
│  [H1 title — 2 lines max]          │
│  [subtitle]                        │
│                                    │
│  ┌──────────────────────────────┐  │
│  │ AUC-ROC   0.886   ▲+0.11 ⓘ  │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ Transactions  800K+       ⓘ  │  │
│  └──────────────────────────────┘  │
│                                    │
│  [LightGBM][SHAP][Ethical AI]      │
│  (wraps to 2 rows)                 │
│                                    │
│  [View Case Study ↓] (full width)  │
│  [Run Demo ▶]       (full width)   │
│  [GitHub →]         (text link)    │
│                                    │
│ ─── SECTION NAV (horizontal scroll)│
│  Problem|Dataset|Approach|Results  │
│                                    │
│ ─── BODY CONTENT ────────────────  │
│  (single column, 16px padding)     │
│                                    │
│ ─── REPRO TOGGLE ────────────────  │
│  [▼ Reproducibility] (full width)  │
│                                    │
└────────────────────────────────────┘
```

---

## Component Placement Rules

### Desktop
- **Header bg:** same dark gradient as hero (visual continuity from hero if navigated from home)
- **MetricPill strip:** flex-row, centre-aligned, wrap allowed, gap 12px
- **Sticky section nav:** `position: sticky; top: 64px` (below main nav); `overflow-x: auto` for long lists
- **Body content:** max-width 820px, 64px top/bottom padding, 16px horizontal padding
- **Tables:** full-width, alternating row bg at `rgba(11,61,145,0.04)` on light bg
- **Code blocks:** dark bg (`#0F1724`), monospace font, horizontal scroll
- **Repro toggle:** full-width button, primary color, collapsible with CSS `max-height` transition

### Mobile
- **MetricPills:** Stack vertically (flex-direction: column), full width
- **CTAs:** Stack vertically, full width
- **Section nav:** `overflow-x: auto; white-space: nowrap` — horizontal scroll
- **Code blocks:** `overflow-x: auto` — horizontal scroll, 14px font
- **Repro section:** Same full-width toggle, content padding 16px
