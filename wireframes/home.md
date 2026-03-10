# Wireframe: Home Page
**Route:** `/`
**Components:** Nav · Hero · ProjectCard × 3 · Skills · Resume blurb · CTA strip

---

## Desktop Layout (1200px)

```
┌─────────────────────────────────────────────────────────────────────┐
│  NAV                                                                 │
│  [KS Logo | AI Engineer]    Projects  About  Resume  Blog   [Sched] │
└─────────────────────────────────────────────────────────────────────┘
│                         HERO (full-bleed, dark bg)                   │
│              ┌──────────────────────────────────────┐                │
│              │  [role label — small caps, accent]   │                │
│              │  AI Engineer · ML Engineer ·          │                │
│              │  Data Scientist                       │                │
│              │                                       │                │
│              │  Building AI that detects,            │                │
│              │  protects, and predicts —             │                │
│              │  with measurable impact.     [H1]     │                │
│              │                                       │                │
│              │  ┌─────────────────────────────┐      │                │
│              │  │ AUC-ROC ● 0.886 ▲+0.11 [ⓘ] │  ← MetricPill      │
│              │  └─────────────────────────────┘      │                │
│              │                                       │                │
│              │  [View Case Study ▶]  [Run Demo ▶]    │                │
│              │    (primary CTA)      (secondary CTA) │                │
│              └──────────────────────────────────────┘                │
│                                                                       │
│ ─────────────── FEATURED PROJECTS ────────────────────────────────── │
│                                                                       │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌────────────────┐  │
│  │  PROJECT CARD 1     │ │  PROJECT CARD 2     │ │ PROJECT CARD 3 │  │
│  │  [thumbnail]        │ │  [thumbnail]        │ │ [thumbnail]    │  │
│  │  Fraud Detection    │ │  Depression Detect. │ │ Federated AI   │  │
│  │  AUC 0.886 ▲+0.11  │ │  Accuracy 91% ▲+20% │ │ ε ≤ 1.0 DP    │  │
│  │  [tag][tag][tag]    │ │  [tag][tag][tag]    │ │ [tag][tag]     │  │
│  │  ─── on hover ───  │ │                     │ │                │  │
│  │  • bullet 1        │ │  (hover reveals     │ │ (hover reveals │  │
│  │  • bullet 2        │ │   3 bullets)        │ │  3 bullets)    │  │
│  │  • bullet 3        │ │                     │ │                │  │
│  │  [View CS] [Demo]  │ │  [View CS] [Demo]   │ │ [View CS][Demo]│  │
│  └─────────────────────┘ └─────────────────────┘ └────────────────┘  │
│                                                                       │
│ ─────────────────── SKILLS ────────────────────────────────────────── │
│                                                                       │
│  ML / DL            NLP           Privacy/Fed      MLOps             │
│  [PyTorch][sklearn] [HuggingFace] [Flower][Opacus] [Docker][FastAPI] │
│  [LightGBM][ONNX]   [BERT][spaCy] [Diff. Privacy]  [GH Actions]     │
│                                                                       │
│ ─────────────────── ABOUT BLURB ───────────────────────────────────── │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  [Profile image?]  AI Engineer... [3-4 sentence summary]        │  │
│  │                    [Download Resume]  [Schedule Interview]       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│ ─────────────────── CTA STRIP ─────────────────────────────────────── │
│         [View All Projects →]  [Schedule Interview →]                │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Mobile Layout (375px)

```
┌────────────────────────────────────┐
│ NAV                                │
│ [KS]                      [☰ menu] │
└────────────────────────────────────┘
│          HERO (full-bleed)         │
│  AI Engineer · ML Engineer        │
│  Data Scientist                    │
│                                    │
│  Building AI that detects,         │
│  protects, and predicts...         │
│                                    │
│  ┌──────────────────────────────┐  │
│  │ AUC-ROC  0.886  ▲+0.11  [ⓘ] │  │
│  └──────────────────────────────┘  │
│                                    │
│  [View Case Study ▶]               │
│  [Run Demo ▶]                      │
│                                    │
│ ─── FEATURED PROJECTS ──────────── │
│  ┌──────────────────────────────┐  │
│  │ PROJECT CARD 1 (full width)  │  │
│  │ Fraud Detection              │  │
│  │ AUC 0.886  [tag][tag]        │  │
│  │ [View CS]  [Demo]            │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ PROJECT CARD 2               │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ PROJECT CARD 3               │  │
│  └──────────────────────────────┘  │
│                                    │
│ ─── SKILLS (scrollable tags) ───── │
│  [PyTorch][HuggingFace][LightGBM] │
│  [Flower][Docker][FastAPI]...      │
│                                    │
│ ─── ABOUT / RESUME BLURB ────────  │
│  [3-sentence summary]              │
│  [Download Resume]                 │
│  [Schedule Interview]              │
└────────────────────────────────────┘
```

---

## Component Placement Rules

### Desktop
- **Nav:** Sticky top, 64px height, max-width 1200px centred
- **Hero:** 100vh min-height, content max-width 780px centred, text centred
- **MetricPill:** Centred below tagline, loads with initial HTML (no lazy load)
- **Featured Projects:** 3-column grid, gap 24px, cards equal height
- **ProjectCard hover bullets:** Revealed via CSS `max-height` transition on `.project-card:hover` + `:focus-within`
- **Skills:** Flex-wrap tag cloud, 2 rows visible, 32px padding
- **CTA strip:** Centred, 96px vertical padding

### Mobile (< 768px)
- **Nav:** Hamburger menu; drawer slides in from top with backdrop
- **Hero CTAs:** Stack vertically, full width minus 32px padding
- **Projects:** Single column, full width
- **MetricPill:** Full width minus 32px padding
- **Skills:** Single row horizontal scroll (no wrap)
