# 6-Week Portfolio Build Plan
**Owner:** Komal Shahid
**Goal:** Launch a recruiter-optimised AI/ML portfolio at `ukomal.github.io/Komal-Shahid-DS-Portfolio`
**Stack recommendation:** Next.js 14 + Tailwind CSS + Vercel (Option A)
**Daily time budget:** ~2–3 hours

---

## Pre-Week 0: Decisions & Setup (Day 1, 2h)

- [ ] Choose stack: **Option A (Next.js + Tailwind + Vercel)** vs Option B (Gatsby + Netlify)
- [ ] Create GitHub repo `Komal-Shahid-DS-Portfolio` (if not exists) and enable GitHub Pages
- [ ] Set up Vercel project linked to the repo (free tier)
- [ ] Register custom domain (optional): `komalshahid.dev` or similar
- [ ] Set up Google Analytics 4 property and copy Measurement ID into `analytics/tracking_plan.md`
- [ ] Create Calendly free account at `calendly.com/komalshahid` (30-min intro call type)

---

## Week 1: Foundation & Hero (Days 1–7)

**Goal:** Hero communicates role + top metric within 3 seconds on mobile. CI pipeline running.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 1 | Scaffold Next.js 14 app with Tailwind | `/` route renders | 2h |
| 1 | Copy `assets/tokens.css` into Tailwind config as CSS variables | Token system live | 30m |
| 2 | Build `Nav.jsx` component (sticky, mobile hamburger, skip-to-main) | Nav on all pages | 2h |
| 3 | Build `MetricPill.jsx` with tooltip | Standalone render | 1h |
| 3 | Build `CTAButton.jsx` (primary, secondary, ghost variants) | Buttons render | 1h |
| 4 | Build `Hero.jsx` — role label + tagline + MetricPill + dual CTAs | Hero renders | 2h |
| 5 | Add hero to `pages/index.tsx` — mobile-first CSS | Hero on `/` | 1.5h |
| 5 | Lighthouse mobile test: Performance ≥ 90, Accessibility ≥ 90 | Audit screenshot | 30m |
| 6 | Set up GitHub Actions CI (`ci/github-actions.yml`) | CI runs on push | 1h |
| 7 | A/B variant B scaffold (metric-first hero) | Both variants ready | 1h |

**Acceptance criteria:**
- [ ] Hero metric "AUC-ROC 0.886" visible < 3s on Moto G4 (Lighthouse throttled)
- [ ] Primary CTA "View Case Study" and secondary CTA "Run Demo" both render
- [ ] Lighthouse mobile: Performance ≥ 90, Accessibility ≥ 90
- [ ] GitHub Actions CI passes on push to main
- [ ] `qa/qa_script.py` hero checks pass

---

## Week 2: Project Cards & Index (Days 8–14)

**Goal:** `/projects` page with 3 featured ProjectCards, hover-reveal bullets, and working filters.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 8 | Build `Badge.jsx` component | Badge renders | 45m |
| 8 | Build `ProjectCard.jsx` with hover-reveal bullets | Card renders | 2h |
| 9 | Build `/projects` page with 3 featured cards | Projects index live | 2h |
| 10 | Add filter tag row (All / NLP / Fraud / Federated / Healthcare) | Filters work | 1.5h |
| 10 | Add secondary project cards (DSC640–DSC530) | All 10+ projects listed | 1.5h |
| 11 | Add `ProjectCard` hover bullets for all 3 featured projects | Hover works | 1h |
| 12 | Add thumbnail placeholder images with proper `alt` text | Images accessible | 1h |
| 13 | WCAG keyboard nav test: Tab through all project cards | All cards focusable | 1h |
| 14 | Run `qa_script.py` against `/projects` | QA passes | 30m |

**Acceptance criteria:**
- [ ] All 3 featured ProjectCards show 3 bullets on hover and focus-within
- [ ] Filter tags work (CSS class toggle or state-based)
- [ ] All images have descriptive alt text
- [ ] Keyboard: Tab, Enter, and focus-visible outlines work on all cards
- [ ] Lighthouse Accessibility ≥ 90 on `/projects`

---

## Week 3: Case Study Pages (Days 15–21)

**Goal:** All 3 case study pages live with full content, collapsible repro section, demo links.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 15 | Build `CaseStudyLayout.jsx` — header + sticky section nav + repro toggle | Layout renders | 3h |
| 16 | Fraud detection case study — paste content from `pages/projects/fraud-detection.md` | `/projects/fraud-detection` live | 2h |
| 17 | Depression detection case study | `/projects/depression-detection` live | 2h |
| 18 | Federated healthcare case study | `/projects/federated-healthcare-ai` live | 2h |
| 19 | Add JSON-LD structured data from `seo/metadata.jsonld` to each case study `<head>` | Structured data validates | 1h |
| 20 | Test collapsible repro section keyboard accessibility (Enter toggles) | Repro accessible | 1h |
| 21 | Run `qa_script.py` against all 3 case study pages | All checks pass | 30m |

**Acceptance criteria:**
- [ ] Each case study has: problem, dataset, approach, results, fairness, reproducibility, demo, lessons
- [ ] Collapsible repro section works on click and keyboard
- [ ] Sticky section nav scrolls to correct anchor
- [ ] Demo section has code block + sample input/output
- [ ] Notebook link and GitHub link present on each case study
- [ ] `qa_script.py` case study checks pass

---

## Week 4: About, Resume, Contact & SEO (Days 22–28)

**Goal:** Supporting pages live, SEO metadata in place, sitemap submitted.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 22 | Build `/about` page from `pages/about.md` | About page live | 1.5h |
| 22 | Build `/resume` page from `pages/resume.md` | Resume page live | 1h |
| 23 | Build `/contact` page from `pages/contact.md` | Contact page live | 1h |
| 23 | Add Calendly embed to contact page | Calendly widget works | 30m |
| 24 | Add meta tags (title, description, OG) to all pages | SEO tags in head | 1.5h |
| 24 | Add JSON-LD Person schema to home page and about page | Person schema validates | 1h |
| 25 | Generate and submit `seo/sitemap.xml` to Google Search Console | Sitemap indexed | 1h |
| 26 | Add skip-to-main content link to Nav | Skip link works on Tab | 30m |
| 26 | Fix any `1.4.3c` contrast issue: accent bg → dark text | Contrast AA passes | 30m |
| 27 | PDF resume export from `content/resume.md` (Word → PDF) | PDF downloadable | 1h |
| 28 | Full WCAG checklist audit from `accessibility/wcag_checklist.md` | All items checked | 2h |

**Acceptance criteria:**
- [ ] All pages have unique `<title>` and `<meta description>`
- [ ] OG image set for social sharing
- [ ] JSON-LD validates at https://search.google.com/test/rich-results
- [ ] Sitemap submitted to Google Search Console
- [ ] WCAG checklist 100% marked ☑ or tracked for remediation
- [ ] PDF resume downloads correctly

---

## Week 5: Blog, Analytics, A/B Tests, QA (Days 29–35)

**Goal:** Blog live with 2 posts, analytics tracking, A/B test running, Lighthouse targets met.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 29 | Build `/blog` index from `pages/blog/index.md` | Blog page live | 1.5h |
| 29 | Write first blog post: "Why Conservative SMOTE Beats Aggressive Oversampling" | Post live | 2h |
| 30 | Write second blog post: "Differential Privacy in Plain English" | Post live | 2h |
| 31 | Add GA4 snippet to all pages | Events firing in GA4 | 1h |
| 31 | Add `data-analytics-event` attributes to all CTAs | CTA events tracked | 1h |
| 32 | Implement A/B test 1 (hero CTA text) via Next.js middleware | A/B test running | 2h |
| 33 | Set up GA4 conversion events and goals dashboard | KPIs visible | 1h |
| 34 | Full Lighthouse audit on all major pages (mobile + desktop) | Results documented | 1.5h |
| 35 | Fix any Performance or Accessibility failures | All targets met | 2h |

**Acceptance criteria:**
- [ ] Blog has ≥ 2 published posts with proper SEO meta
- [ ] GA4 firing `cta_click` events for all major CTAs
- [ ] A/B test 1 running; cookie set for variant assignment
- [ ] Lighthouse mobile: Performance ≥ 90, Accessibility ≥ 90 on `/` and first case study
- [ ] `qa_script.py` all checks green

---

## Week 6: Docker, Demo Endpoints, Polish & Launch (Days 36–42)

**Goal:** Portfolio deployed, demo endpoints live, Docker image built, site announced.

### Tasks

| Day | Task | Deliverable | Est. |
|-----|------|-------------|------|
| 36 | Build FastAPI demo endpoint for fraud detection (from `deploy/`) | `/predict` endpoint works | 3h |
| 37 | Dockerise FastAPI demo (`deploy/Dockerfile`) | `docker run` works locally | 1.5h |
| 37 | Deploy demo to Railway/Render free tier | Live demo URL | 1h |
| 38 | Add Binder badge to case study pages | Binder links work | 30m |
| 39 | Final content review: update placeholder `[DATE]`, `[EMAIL]`, certifications | All placeholders filled | 2h |
| 39 | Update GitHub profile README (`content/github_profile_readme.md`) | Profile README live | 1h |
| 40 | Pin 3 projects + portfolio on GitHub | GitHub profile polished | 30m |
| 40 | Update LinkedIn with new portfolio URL, headline, and About section | LinkedIn updated | 1h |
| 41 | Final `qa_script.py` run — all checks green | QA pass report | 30m |
| 41 | Final Lighthouse run — Performance ≥ 90, Accessibility ≥ 90 | Lighthouse pass | 30m |
| 42 | **LAUNCH:** Share portfolio on LinkedIn, Discord communities, relevant subreddits | Portfolio announced | 1h |

**Acceptance criteria:**
- [ ] FastAPI demo endpoint live and responds to sample JSON input
- [ ] Binder link works for at least one project
- [ ] Docker image builds with `docker build -f deploy/Dockerfile .`
- [ ] All placeholder content replaced with real content
- [ ] `qa_script.py` run against live URL: 0 failures
- [ ] GitHub profile README live at `github.com/UKOMAL`
- [ ] LinkedIn updated with new portfolio link and headline

---

## Tech Stack Quick-Start (Option A)

```bash
# 1. Scaffold Next.js app
npx create-next-app@latest portfolio --typescript --tailwind --app --no-src-dir
cd portfolio

# 2. Copy design system files
cp /path/to/assets/tokens.css ./app/globals.css  # merge tokens into globals.css

# 3. Install additional dependencies
npm install clsx           # conditional classnames
npm install lucide-react   # icon set

# 4. Copy component files from components/
cp /path/to/components/*.jsx ./app/components/

# 5. Start dev server
npm run dev  # http://localhost:3000

# 6. Or use the run script
./deploy/run.sh dev
```

---

## Weekly Check-ins Checklist

Each Friday, verify:
- [ ] Lighthouse mobile score ≥ 90 (Performance + Accessibility)
- [ ] `qa_script.py` 0 failures
- [ ] GA4 showing expected events
- [ ] All new pages have `<title>` and `<meta description>`
- [ ] No broken links (use `npm run check-links` or broken-link-checker)
- [ ] New content added since last week committed and deployed

---

## Post-Launch KPI Tracking (Week 7+)

Review monthly:
| KPI | Target | Source |
|-----|--------|--------|
| Case Study Click Rate | ≥ 15% | GA4 |
| Demo Run Rate | ≥ 10% | GA4 |
| GitHub Click Rate | ≥ 20% | GA4 |
| Resume Downloads | ≥ 5% | GA4 file_download |
| Calendly Clicks | ≥ 3% | GA4 |
| Lighthouse Performance | ≥ 90 | Monthly audit |
| Lighthouse Accessibility | ≥ 90 | Monthly audit |
