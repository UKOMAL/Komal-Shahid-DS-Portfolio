# Analytics & Conversion Tracking Plan
**Portfolio:** Komal Shahid — AI & ML Portfolio
**Analytics platform:** Google Analytics 4 (GA4) — swap snippet for Plausible/Fathom if preferred

---

## Analytics Snippet (GA4)

Paste in `<head>` on every page:

```html
<!-- Google tag (gtag.js) — replace G-XXXXXXXXXX with your Measurement ID -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){ dataLayer.push(arguments); }
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX', {
    send_page_view: true,
    anonymize_ip: true,       // GDPR compliance
    cookie_flags: 'SameSite=None;Secure'
  });
</script>
```

**Privacy-first alternative (no cookies):**
```html
<!-- Plausible Analytics — fully GDPR compliant, no cookies -->
<script defer data-domain="ukomal.github.io" src="https://plausible.io/js/script.js"></script>
```

---

## CTA Event Tracking

Attach these `data-*` attributes to every CTA (already included in component templates):

| Attribute | Example Value | Purpose |
|-----------|--------------|---------|
| `data-analytics-event` | `"cta_click"` | Event name |
| `data-analytics-label` | `"View Case Study"` | CTA label |
| `data-analytics-project` | `"fraud-detection"` | Which project |

**Global event listener (add once in `_app.jsx` or `layout.jsx`):**

```javascript
// analytics.js — drop into /lib/analytics.js
export function trackEvent(eventName, params = {}) {
  if (typeof window === 'undefined' || !window.gtag) return;
  window.gtag('event', eventName, {
    event_category: params.category || 'engagement',
    event_label: params.label || '',
    value: params.value || 1,
    ...params,
  });
}

// Auto-track all [data-analytics-event] clicks
if (typeof document !== 'undefined') {
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-analytics-event]');
    if (!el) return;
    trackEvent(el.dataset.analyticsEvent, {
      label: el.dataset.analyticsLabel,
      project: el.dataset.analyticsProject,
      category: 'cta',
    });
  });
}
```

---

## Conversion KPI Definitions

| KPI | Definition | How to Measure | Target |
|-----|-----------|---------------|--------|
| **Case Study Click Rate** | % of home page visitors who click "View Case Study" | GA4: event `cta_click`, label = "View Case Study" / sessions | ≥ 15% |
| **Demo Run Rate** | % of case study visitors who click "Run Demo" | GA4: event `cta_click`, label = "Run Demo" / case study sessions | ≥ 10% |
| **GitHub Click Rate** | % of project page visitors who click GitHub link | GA4: event `github_click` / project page sessions | ≥ 20% |
| **Resume Download Rate** | % of sessions that download resume PDF | GA4: file_download event on `/assets/komal-shahid-resume.pdf` | ≥ 5% |
| **Schedule Interview Click** | % of sessions that click Calendly link | GA4: event `cta_click`, label = "Schedule Interview" | ≥ 3% |
| **Bounce Rate (Hero page)** | % of sessions that leave without interacting | GA4: engagement_rate < 10s | < 60% |
| **Session Duration** | Average time on site | GA4: average_session_duration | ≥ 90s |

---

## Event Tracking Plan

| Page | Event Name | Trigger | Parameters |
|------|-----------|---------|----------|
| Home hero | `cta_click` | "View Case Study" click | `label: "View Case Study"` |
| Home hero | `cta_click` | "Run Demo" click | `label: "Run Demo"` |
| ProjectCard | `cta_click` | "View Case Study" on card | `label: "View Case Study"`, `project: <title>` |
| ProjectCard | `cta_click` | "Run Demo" on card | `label: "Run Demo"`, `project: <title>` |
| ProjectCard | `github_click` | GitHub link click | `project: <title>` |
| Case study | `section_view` | Scroll to each section anchor | `section: "problem"` / "results" / "demo" / "reproducibility"` |
| Case study | `repro_open` | Reproducibility section expanded | `project: <title>` |
| Resume page | `file_download` | PDF download click | `file: "resume.pdf"` |
| Contact page | `cta_click` | Calendly link click | `label: "Schedule Interview"` |
| Nav | `cta_click` | Nav "Schedule Interview" click | `location: "nav"` |
| Blog post | `scroll_depth` | 25%, 50%, 75%, 100% scroll | `depth: 75`, `post: <title>` |

---

## A/B Test Ideas

### A/B Test 1: Hero CTA Text

**Hypothesis:** "View Case Study" converts better than "See How It Works"  
**What to test:**  
- Variant A (control): `"View Case Study"` (primary) + `"Run Demo"` (secondary)  
- Variant B (treatment): `"See Results →"` (primary) + `"Try the Demo"` (secondary)  
**Primary metric:** Click-through rate on primary CTA  
**Sample size:** ~500 unique visitors per variant (use 95% confidence, MDE = 3%)  
**Tool:** Next.js Edge Middleware (A/B by cookie) or Vercel Edge Config  
**Expected winner:** Variant A — "View Case Study" is more specific and signals depth of content  

```javascript
// pages/_middleware.js (Next.js Edge)
import { NextResponse } from 'next/server';
export function middleware(req) {
  const variant = req.cookies.get('ab_hero') || (Math.random() > 0.5 ? 'A' : 'B');
  const res = NextResponse.next();
  res.cookies.set('ab_hero', variant, { maxAge: 60 * 60 * 24 * 7 });
  res.headers.set('x-ab-variant', variant);
  return res;
}
```

---

### A/B Test 2: Hero Layout — Metric-First vs Story-First

**Hypothesis:** Showing the top metric (AUC 0.886) before the tagline converts better with technical recruiter audiences  
**Variant A (control — story-first):**  
```
[Role label]
[Tagline text]
[MetricPill: AUC 0.886]
[CTA buttons]
```
**Variant B (treatment — metric-first):**  
```
[MetricPill: AUC 0.886 — TOP RESULT]
[Role label]
[Tagline text]
[CTA buttons]
```
**Primary metric:** "View Case Study" click rate  
**Secondary metric:** Time-on-page (engagement proxy)  
**Expected winner:** Context-dependent — metric-first likely wins with technical recruiters; story-first with hiring managers. Segment by traffic source (LinkedIn vs Google).  

---

## Funnel Analysis

```
Home page visit
    ↓ (target: 15%)
Case Study click (any project)
    ↓ (target: 35%)
Reach "Results" section (scroll depth)
    ↓ (target: 25%)
Demo click or GitHub click
    ↓ (target: 8%)
Contact / Schedule Interview
```

**Weekly review:** Check drop-off at each funnel step. If < 15% reach Results section, the problem is content length or page load speed. If < 8% click Demo, the demo link is broken or unclear.

---

## UTM Parameter Conventions

Use these UTM parameters when sharing your portfolio:

| Channel | utm_source | utm_medium | utm_campaign |
|---------|-----------|------------|-------------|
| LinkedIn bio | linkedin | social | portfolio |
| Resume PDF link | resume-pdf | document | portfolio |
| Email signature | email-sig | email | portfolio |
| Job application | job-app | outreach | [company-name] |
| GitHub README | github-profile | referral | portfolio |
