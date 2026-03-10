# WCAG AA Accessibility Checklist
**Portfolio:** Komal Shahid — AI & ML Portfolio
**Target:** WCAG 2.1 Level AA
**Lighthouse Accessibility Target:** ≥ 90 (mobile)

---

## 1. Perceivable

### 1.1 Text Alternatives
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 1.1.1 | Every `<img>` has a descriptive `alt` attribute | All pages | Screen reader announces meaningful description; decorative images use `alt=""` | ☐ |
| 1.1.2 | Hero background image uses CSS, not `<img>` | Hero | No alt needed; content not dependent on image | ☐ |
| 1.1.3 | Icons without visible text have `aria-label` | Nav, CTAButton | Icon purpose announced | ☐ |

### 1.2 Time-Based Media
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 1.2.1 | Any demo video has captions | Demo embeds | CC available | ☐ |
| 1.2.2 | No auto-playing audio | All pages | No audio starts without user action | ☐ |

### 1.3 Adaptable
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 1.3.1 | Semantic HTML heading hierarchy (h1→h2→h3) | All pages | No skipped heading levels | ☐ |
| 1.3.2 | Data tables use `<thead>`, `<th scope>` | Resume, Results tables | Screen reader announces column/row headers | ☐ |
| 1.3.3 | Instructions don't rely solely on visual cues | All forms | E.g., "Click the blue button" also says "labeled Schedule Interview" | ☐ |
| 1.3.4 | No content locked to orientation | Mobile views | Content readable in portrait and landscape | ☐ |
| 1.3.5 | Form inputs have `autocomplete` attributes | Contact form | Browser autocompletes name, email | ☐ |

### 1.4 Distinguishable
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 1.4.1 | Color is not the sole information carrier | Badge, MetricPill | Deltas show ▲/▼ symbol in addition to color | ☐ |
| 1.4.3 | Normal text contrast ≥ 4.5:1 | Body text | `#1A202C` on `#F7FAFC` = 15.3:1 ✓ | ☐ |
| 1.4.3b | Primary `#0B3D91` on white = 8.6:1 ✓ | Nav links, headings | Passes AA and AAA | ☐ |
| 1.4.3c | Accent `#00A3FF` on dark `#0F1724` | Hero role label | Contrast = 5.8:1 ✓ (AA) | ☐ |
| 1.4.3d | White text on accent `#00A3FF` background | CTAButton primary | Contrast = 1.9:1 ✗ — **use dark text on accent bg** | ☐ |
| 1.4.4 | Text resizable to 200% without loss of content | All pages | No horizontal scrolling at 200% zoom | ☐ |
| 1.4.5 | Text used instead of images of text | All pages | All text is real HTML text | ☐ |
| 1.4.10 | Reflow at 320px width | All pages | Single-column layout; no loss of content | ☐ |
| 1.4.11 | Non-text contrast ≥ 3:1 for UI components | CTAButton borders, inputs | Button border vs background ≥ 3:1 | ☐ |
| 1.4.12 | Text spacing overrides don't break layout | All pages | Apply bookmarklet: line-height 1.5×, letter-spacing 0.12em | ☐ |
| 1.4.13 | Hover/focus content dismissable and persistent | MetricPill tooltip, ProjectCard hover | Tooltip stays while hovering; Esc dismisses | ☐ |

---

## 2. Operable

### 2.1 Keyboard Accessible
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 2.1.1 | All interactive elements reachable by Tab | Nav, CTAButton, ProjectCard, CaseStudyLayout | Tab order follows visual layout | ☐ |
| 2.1.1b | ProjectCard hover bullets visible on keyboard focus | ProjectCard | `focus-within` CSS reveals bullets | ☐ |
| 2.1.1c | Mobile nav drawer keyboard accessible | Nav hamburger | Tab reaches all items when open; trapped when open | ☐ |
| 2.1.2 | No keyboard traps | Modal dialogs | Esc closes; focus returns to trigger | ☐ |
| 2.1.4 | No single-key shortcuts that conflict | All pages | No custom keyboard shortcuts | ☐ |

### 2.2 Enough Time
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 2.2.1 | No time limits on content | All pages | No timed auto-dismiss elements | ☐ |
| 2.2.2 | No blinking/scrolling content > 5 seconds | All pages | No marquees or auto-scrolling carousels | ☐ |

### 2.3 Seizures and Physical Reactions
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 2.3.1 | No flashing > 3 times/second | All pages | No flash animations | ☐ |

### 2.4 Navigable
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 2.4.1 | Skip-to-main-content link present | Nav | First focusable element; visible on focus | ☐ |
| 2.4.2 | Unique, descriptive page titles | All pages | See `site_manifest.json` title fields | ☐ |
| 2.4.3 | Focus order logical and meaningful | All pages | Tab sequence matches visual reading order | ☐ |
| 2.4.4 | Link purpose clear from text alone | All CTAs, nav | "View Case Study" identifies the action and context | ☐ |
| 2.4.5 | Multiple ways to navigate | Site | Nav + sitemap + search (if implemented) | ☐ |
| 2.4.6 | Headings and labels describe purpose | All pages | H1 = page topic; labels match input purpose | ☐ |
| 2.4.7 | Focus visible on all interactive elements | All | `:focus-visible` outlines defined in tokens.css | ☐ |

### 2.5 Input Modalities
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 2.5.1 | No path-based or multi-point gestures required | Touch devices | All gestures have single-tap equivalents | ☐ |
| 2.5.3 | Label matches accessible name | CTAButton | Visible text matches aria-label | ☐ |
| 2.5.5 | Touch targets ≥ 44×44px | CTAButton, Nav links | `min-height: 44px` in CTAButton CSS | ☐ |

---

## 3. Understandable

### 3.1 Readable
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 3.1.1 | `lang` attribute on `<html>` | index.html | `<html lang="en">` | ☐ |
| 3.1.2 | Language changes marked for mixed-language content | All pages | N/A — single language | ☐ |

### 3.2 Predictable
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 3.2.1 | No context change on focus | Nav, forms | Hover/focus doesn't trigger navigation | ☐ |
| 3.2.2 | No automatic form submission | Contact form | Submit only on explicit button press | ☐ |

### 3.3 Input Assistance
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 3.3.1 | Error messages identify the field | Contact form | "Email field: valid email required" | ☐ |
| 3.3.2 | Labels or instructions for all inputs | Contact form | `<label>` associated with every `<input>` | ☐ |

---

## 4. Robust

### 4.1 Compatible
| # | Test | Component | Expected Outcome | Status |
|---|------|-----------|-----------------|--------|
| 4.1.1 | Valid HTML (no parsing errors) | All pages | Run W3C Validator; 0 errors | ☐ |
| 4.1.2 | Name, role, value set for custom components | MetricPill tooltip, repro toggle | `aria-expanded`, `aria-controls`, `role="tooltip"` present | ☐ |
| 4.1.3 | Status messages programmatically determinable | Forms, demos | Success/error states use `aria-live="polite"` | ☐ |

---

## Quick Test Commands

```bash
# Lighthouse CLI audit (install once: npm i -g lighthouse)
lighthouse https://ukomal.github.io/Komal-Shahid-DS-Portfolio \
  --only-categories=accessibility \
  --output=html --output-path=./accessibility/lighthouse-report.html

# axe-core in browser console
# Paste into DevTools console on any page:
# npm i -g @axe-core/cli
axe https://ukomal.github.io/Komal-Shahid-DS-Portfolio --tags wcag2a,wcag2aa

# Color contrast checker
# Use https://webaim.org/resources/contrastchecker/
# Primary #0B3D91 on white #FFFFFF = 8.59:1 ✓ AAA
# Accent #00A3FF on dark #0F1724 = 5.8:1 ✓ AA
# ⚠️  White text #FFFFFF on accent #00A3FF = 1.9:1 ✗ — use dark text instead
```

---

## Contrast Reference Table

| Foreground | Background | Ratio | WCAG Level |
|-----------|-----------|-------|-----------|
| `#1A202C` | `#F7FAFC` | 15.3:1 | AAA ✓ |
| `#0B3D91` | `#FFFFFF` | 8.6:1 | AAA ✓ |
| `#FFFFFF` | `#0B3D91` | 8.6:1 | AAA ✓ |
| `#00A3FF` | `#0F1724` | 5.8:1 | AA ✓ |
| `#FFFFFF` | `#00A3FF` | 1.9:1 | FAIL ✗ — use `#0F1724` on accent |
| `#4A5568` | `#FFFFFF` | 5.9:1 | AA ✓ |

**Action required:** When using `--color-accent` (`#00A3FF`) as a button background, set text to `--color-neutral-dark` (`#0F1724`), not white.
