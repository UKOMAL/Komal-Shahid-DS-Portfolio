# Portfolio UX/UI Improvement Report — October 2026 Cycle

**Agent run:** Automated (scheduled task)  
**File modified:** `DSC680/website/portfolio.html`  
**File size after:** 234,062 chars · 5,034 lines  
**HTML validation:** PASS (no unclosed tags, no parse errors)

---

## 1. Evaluation Summary

The portfolio is already at an exceptionally high baseline — it includes a WebGL neural constellation, custom cursor with blend-mode, dark/light theme, horizontal scroll project panels, modal case studies with prev/next navigation, skill bars with proficiency percentages, animated timelines, toast notifications, keyboard shortcuts, print styles, and full `prefers-reduced-motion` support. The brand system (navy · gold · burgundy · sage · cream) is cohesive and applied consistently.

This evaluation cycle focused on three areas left unfilled by prior updates:

| Area | Finding |
|------|---------|
| Dark mode hero atmosphere | The background dot grid was the only visual texture in dark mode; the neural constellation floated on a flat `#0f0f1a` surface |
| Section heading arrival | Headings revealed cleanly but without a "data identity" hook — nothing linked the visual language to the ML/code focus of the work |
| Keyboard & focus accessibility | Skill cards were not focusable or keyboard-navigable; side-nav arrival had no spatial confirmation for keyboard users |

---

## 2. Enhancements Implemented

### Enhancement 1 — Dark Mode Aurora Background  
**Issue:** In dark mode the hero felt visually flat behind the constellation. No atmospheric depth.  
**Fix:** Added three animated radial-gradient blobs (`auroraFloat`, 22 s cycle) appended as `::after` on `.hero__bg-grid` — only active when `[data-theme="dark"]`. Blobs use the existing design-token palette (`--gold`, `--burgundy`, `--sage`), filtered at `blur(56px)`, so they read as atmospheric glow rather than solid colour.  
**Motion safety:** Animation disabled for `prefers-reduced-motion`.  
**Why it works:** The aurora lives under the constellation z-index, so it enriches depth without competing for attention. On theme toggle it appears/disappears instantly via the existing theme transition.

```css
[data-theme="dark"] .hero__bg-grid::after {
  /* three radial blobs at 14%, 84%, 54% horizontal positions */
  animation: auroraFloat 22s ease-in-out infinite alternate;
  filter: blur(56px);
}
```

---

### Enhancement 2 — Scramble-Decode Effect on Section Headings  
**Issue:** Section headings (`h2`) appeared via the standard fade-up reveal with no connection to the ML/data brand identity.  
**Fix:** An `IntersectionObserver` watches each `.section__head h2`. On first viewport entry, a `TreeWalker` extracts raw text nodes (preserving `<em>`/`<strong>` child markup) and applies a frame-by-frame character scramble: characters unlock left-to-right using a `1 - (1-p)^2.5` ease-out curve over 620 ms. Scramble characters are drawn from `A–Z 0–9 ∑ ∆ ≈ ≡ ∂ ∫ §` — mathematical symbols that reinforce the data-science framing.  
**Motion safety:** Guarded by `PRM` (reduced-motion flag) — headings remain static for motion-sensitive users.  
**Why it works:** The effect fires only once per heading (observer disconnects after trigger) and resolves cleanly to the original markup, so it's a reward for first-time readers without becoming repetitive.

---

### Enhancement 3 — Section Arrival Pulse on Side-Nav Click  
**Issue:** Clicking a side-nav dot scrolled the page but gave no spatial confirmation of arrival — users had to visually locate the heading themselves.  
**Fix:** A 620 ms delayed callback (timed to smooth-scroll completion) adds `.flash-pulse` to the target section's `.section__head`, triggering a CSS `box-shadow` outward ripple animation. The class self-removes on `animationend`.  
**Why it works:** Closes the feedback loop between navigation intent and arrival — a spatial cue that would otherwise require JavaScript focus management.

---

### Enhancement 4 — Skill Card Keyboard Navigation  
**Issue:** Skill cards had no `tabindex` and could not be reached or navigated by keyboard. Focus management jumped over the entire skills grid.  
**Fix:** All `.skill-card` elements receive `tabindex="0"`. Arrow keys (`↑ ↓ ← →`) move focus between cards in DOM order. `Enter`/`Space` while a card has focus triggers its primary interactive child (button or link). Enhanced `:focus-within` CSS outline (2px gold, 4px offset) ensures visible focus state.  
**Why it works:** WCAG 2.1 SC 2.1.1 compliance — keyboard users can now fully explore the skills section without mouse or touch.

---

## 3. Visual & Motion Design Guide

| Token | Value | Role |
|-------|-------|------|
| Aurora blob 1 | `--gold` @ 9% opacity, `blur(56px)` | Warm atmospheric anchor (left-hero) |
| Aurora blob 2 | `--burgundy` @ 11% opacity | Cool counterpoint (right-hero) |
| Aurora blob 3 | `--sage` @ 7% opacity | Subtle upper hint |
| Scramble chars | `A-Z 0-9 ∑∆≈≡∂∫§¶` | Reinforces data/ML brand |
| Scramble easing | `1 - (1-p)^2.5` | Accelerating reveal (feels like decryption) |
| Pulse ring | `--accent` @ 40% → 0% | Box-shadow radial ripple, 0.9 s |

All new animations respect `prefers-reduced-motion: reduce`.

---

## 4. No Regressions

- HTML validated clean (Python `HTMLParser` — 0 unclosed tags, 0 parse errors)
- All existing features confirmed present: WebGL constellation, dark/light toggle, custom cursor, horizontal scroll, modals, skill bars, timeline, contact form, toasts, keyboard shortcuts
- Aurora uses `::after` on existing `.hero__bg-grid` — does not affect z-index stack or layout
- Scramble effect operates only on text nodes, preserving all `<em>` and `<strong>` child elements

---

## 5. Remaining Opportunities (Next Cycle)

1. **Image lazy-load audit** — The profile photo uses `loading="eager"`; project card images could benefit from `loading="lazy"` with a low-quality placeholder blur-up.
2. **Lottie or SVG path animation on the hero** — A hand-drawn SVG line connecting the CTA buttons to the neural canvas would reinforce the organic/technical duality.
3. **Contact form backend** — The form currently has client-side validation but no submission handler; wiring to a Formspree or Netlify Forms endpoint would make it functional.
4. **View Transitions API** — Chrome 111+ supports `document.startViewTransition()` for hero-level cross-section animations; would layer elegantly on the current scroll system.
5. **Open Graph image generation** — An automated OG image built from the hero design tokens would improve social sharing previews.

---

*Report generated by Portfolio UX/UI Agent — automated improvement cycle.*
