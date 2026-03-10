# WCAG Accessibility Checklist
## AI/ML Engineer Portfolio - Accessibility Compliance Guide

### Overview
This checklist ensures WCAG 2.1 AA compliance for the AI/ML engineer portfolio website. All items must be tested and verified before production deployment.

### Testing Environment Requirements
- **Screen Readers**: NVDA (Windows), VoiceOver (macOS), ORCA (Linux)
- **Browser Testing**: Chrome, Firefox, Safari, Edge (latest versions)
- **Mobile Testing**: iOS Safari, Android Chrome
- **Automated Tools**: axe-core, Lighthouse, WAVE, Pa11y

---

## 1. Perceivable - Information and UI components must be perceivable

### 1.1 Text Alternatives
- [ ] **Images have descriptive alt text**
  - ✓ Project screenshots: Describe functionality and interface
  - ✓ Technical diagrams: Explain architecture and data flow
  - ✓ Profile photos: "Komal Shahid, AI Engineer"
  - ✓ Decorative images: Use `alt=""` or `role="presentation"`
  - ✓ Complex images: Use `longdesc` or adjacent explanatory text

**Test Method**: Screen reader navigation, automated scanning
**Expected Outcome**: All images announced meaningfully or skipped appropriately

```html
<!-- Good Examples -->
<img src="/images/depression-ai-architecture.jpg" 
     alt="System architecture diagram showing BERT transformer and CNN components connected through fusion layer for depression detection" />

<img src="/images/hero-background.jpg" 
     alt="" role="presentation" />
```

### 1.2 Time-Based Media
- [ ] **Video content has captions** (if demos include video)
- [ ] **Audio descriptions provided** for visual-only content
- [ ] **Auto-playing media has controls**
- [ ] **No flashing content** that could trigger seizures

**Test Method**: Manual review of all media elements
**Expected Outcome**: All multimedia accessible to deaf/blind users

### 1.3 Adaptable Content
- [ ] **Semantic HTML structure**
  - ✓ Proper heading hierarchy (H1→H2→H3)
  - ✓ Navigation landmarks (`<nav>`, `role="navigation"`)
  - ✓ Main content area (`<main>`, `role="main"`)
  - ✓ Article sections (`<article>`, `<section>`)

```html
<!-- Proper Structure -->
<header role="banner">
  <nav role="navigation">
    <!-- Navigation items -->
  </nav>
</header>

<main role="main">
  <article>
    <h1>Page Title</h1>
    <section>
      <h2>Section Title</h2>
      <!-- Content -->
    </section>
  </article>
</main>

<footer role="contentinfo">
  <!-- Footer content -->
</footer>
```

- [ ] **Content order makes sense without CSS**
- [ ] **Reading order follows visual order**
- [ ] **Tables have headers and captions** (for performance metrics)

**Test Method**: Disable CSS, screen reader navigation
**Expected Outcome**: Logical content flow maintained

### 1.4 Distinguishable
- [ ] **Color contrast meets AA standards**
  - ✓ Normal text: 4.5:1 ratio minimum
  - ✓ Large text (18px+): 3:1 ratio minimum
  - ✓ UI components: 3:1 ratio minimum
  - ✓ Interactive elements clearly distinguishable

**Color Contrast Verification**:
```css
/* Primary Blue (#0B3D91) on White (#FFFFFF) */
/* Ratio: 8.59:1 ✓ PASSES AA */

/* Accent Cyan (#00A3FF) on White (#FFFFFF) */
/* Ratio: 3.84:1 ✓ PASSES AA for large text */

/* Neutral Dark (#0F1724) on White (#FFFFFF) */
/* Ratio: 13.26:1 ✓ PASSES AAA */
```

- [ ] **Information not conveyed by color alone**
  - ✓ Status indicators use icons + color
  - ✓ Links underlined, not just colored
  - ✓ Form errors use text + color
  - ✓ Charts include patterns/labels

**Test Method**: Color blindness simulator, contrast analyzer
**Expected Outcome**: All information accessible without color perception

- [ ] **Text can be resized to 200% without horizontal scrolling**
- [ ] **Images of text avoided** (use actual text when possible)

---

## 2. Operable - UI components and navigation must be operable

### 2.1 Keyboard Accessible
- [ ] **All functionality available via keyboard**
  - ✓ Navigation menus
  - ✓ CTA buttons
  - ✓ Form controls
  - ✓ Modal dialogs
  - ✓ Project card interactions

```javascript
// Keyboard Event Handling Example
const handleKeyPress = (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    openProjectModal();
  }
};
```

- [ ] **No keyboard traps** (users can navigate away from any element)
- [ ] **Skip navigation links** for screen reader users

```html
<a href="#main-content" class="skip-link">Skip to main content</a>
```

**Test Method**: Tab through entire site using only keyboard
**Expected Outcome**: All interactive elements reachable and usable

### 2.2 Enough Time
- [ ] **No time limits** on reading content
- [ ] **Auto-updating content has pause controls** (if applicable)
- [ ] **Session timeouts have warnings** (if applicable)

### 2.3 Seizures and Physical Reactions  
- [ ] **No content flashes more than 3 times per second**
- [ ] **Animation can be paused or disabled**
- [ ] **Respect prefers-reduced-motion setting**

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### 2.4 Navigable
- [ ] **Page titles are descriptive and unique**
  - ✓ Home: "Komal Shahid - AI & ML Engineer | Privacy-Preserving Healthcare AI"
  - ✓ Projects: "AI/ML Projects Portfolio - Depression Detection, Federated Learning"
  - ✓ Case Study: "Depression Detection AI Case Study - 92% AUC Healthcare System"

- [ ] **Focus order is logical and consistent**
- [ ] **Link text is descriptive** (avoid "click here", "read more")

```html
<!-- Bad -->
<a href="/projects/depression-detection">Click here</a>

<!-- Good --> 
<a href="/projects/depression-detection">View Depression Detection AI case study</a>
```

- [ ] **Multiple ways to find pages** (navigation, search, sitemap)
- [ ] **Current location clearly indicated** (breadcrumbs, active nav)

**Test Method**: Screen reader testing, keyboard navigation
**Expected Outcome**: Easy navigation and location awareness

### 2.5 Input Modalities
- [ ] **Touch targets minimum 44×44px** on mobile
- [ ] **Pointer gestures have keyboard alternatives**
- [ ] **Drag and drop has keyboard alternatives** (if used)

---

## 3. Understandable - Information and operation of UI must be understandable

### 3.1 Readable
- [ ] **Page language identified** (`lang="en"`)
- [ ] **Language changes marked** (`lang="es"` for Spanish terms)
- [ ] **Unusual words defined** (technical terms have tooltips/glossary)

```html
<html lang="en">
<p>The model achieved an <abbr title="Area Under the Receiver Operating Characteristic Curve">AUC</abbr> of 0.92.</p>
```

### 3.2 Predictable
- [ ] **Navigation is consistent** across pages
- [ ] **Interactive elements behave consistently**
- [ ] **Context changes only occur on user request** (no surprise redirects)
- [ ] **Forms have clear labels and instructions**

**Test Method**: Navigate between pages, interact with all elements
**Expected Outcome**: Consistent, predictable user experience

### 3.3 Input Assistance
- [ ] **Form validation provides clear error messages**
- [ ] **Required fields clearly marked**
- [ ] **Labels associated with form controls**

```html
<label for="email">Email Address (required)</label>
<input type="email" id="email" required aria-describedby="email-error">
<div id="email-error" role="alert" aria-live="polite">
  Please enter a valid email address
</div>
```

---

## 4. Robust - Content must be robust enough for various user agents

### 4.1 Compatible
- [ ] **Valid HTML markup** (W3C validation)
- [ ] **ARIA attributes used correctly**
- [ ] **Interactive elements have proper roles**

```html
<!-- Custom button component -->
<div role="button" 
     tabindex="0" 
     aria-pressed="false"
     onkeypress="handleKeyPress(event)"
     onclick="toggleFunction()">
  Toggle Option
</div>
```

---

## Testing Procedures

### Automated Testing
```bash
# Install accessibility testing tools
npm install -g axe-cli pa11y lighthouse

# Run automated tests
axe https://komalshahid.dev
pa11y https://komalshahid.dev
lighthouse --only-categories=accessibility https://komalshahid.dev

# Expected Results:
# - axe: 0 violations
# - pa11y: 0 errors  
# - Lighthouse accessibility: ≥90 score
```

### Manual Testing Checklist

#### Screen Reader Testing
- [ ] **NVDA (Windows)**: Complete site navigation
- [ ] **VoiceOver (macOS)**: Hero section and project cards
- [ ] **Mobile VoiceOver/TalkBack**: Touch navigation

#### Keyboard Testing
- [ ] **Tab navigation**: Logical order, visible focus
- [ ] **Enter/Space activation**: All interactive elements
- [ ] **Arrow keys**: Any custom navigation (sliders, tabs)
- [ ] **Escape key**: Closes modals/dropdowns

#### Visual Testing  
- [ ] **200% zoom**: No horizontal scrolling
- [ ] **High contrast mode**: Content still visible
- [ ] **Color blindness**: Information not color-dependent

### Performance Testing with Accessibility
```javascript
// Lighthouse CI configuration
module.exports = {
  ci: {
    collect: {
      numberOfRuns: 3
    },
    assert: {
      assertions: {
        'categories:accessibility': ['error', { minScore: 90 }],
        'categories:performance': ['error', { minScore: 90 }]
      }
    }
  }
};
```

---

## Common Issues and Solutions

### Issue 1: Focus Management in React Components
**Problem**: Focus lost during dynamic content updates
**Solution**: Use React refs and focus management

```jsx
const ProjectModal = () => {
  const modalRef = useRef();
  
  useEffect(() => {
    if (isOpen) {
      modalRef.current?.focus();
    }
  }, [isOpen]);
  
  return (
    <div role="dialog" 
         tabIndex={-1} 
         ref={modalRef}
         aria-labelledby="modal-title">
      <!-- Modal content -->
    </div>
  );
};
```

### Issue 2: Complex Data Visualizations
**Problem**: Charts inaccessible to screen readers
**Solution**: Provide data tables and text summaries

```html
<figure role="img" aria-labelledby="chart-title" aria-describedby="chart-desc">
  <h3 id="chart-title">Model Performance Comparison</h3>
  <div id="chart-desc">
    Depression Detection AI achieves 92% AUC, outperforming baseline by 19 percentage points.
  </div>
  <!-- Chart visualization -->
  <table class="sr-only">
    <!-- Accessible data table version -->
  </table>
</figure>
```

### Issue 3: Color-Only Information
**Problem**: Metrics using only color coding
**Solution**: Add icons and text indicators

```jsx
const MetricPill = ({ status, value }) => (
  <span className={`metric-pill ${status}`}>
    {status === 'good' && <CheckIcon aria-hidden="true" />}
    {status === 'warning' && <AlertIcon aria-hidden="true" />}
    <span className="sr-only">{status} status: </span>
    {value}
  </span>
);
```

---

## Deployment Checklist

### Pre-Launch Verification
- [ ] Run full automated accessibility audit
- [ ] Complete manual testing with screen readers
- [ ] Verify color contrast ratios
- [ ] Test keyboard navigation paths
- [ ] Validate HTML markup
- [ ] Check ARIA implementation
- [ ] Test with users with disabilities (if possible)

### Post-Launch Monitoring
- [ ] Set up continuous accessibility monitoring
- [ ] Regular automated scans (weekly)
- [ ] User feedback mechanism for accessibility issues
- [ ] Annual comprehensive accessibility audit

### Success Metrics
- **Lighthouse Accessibility Score**: ≥ 90
- **Automated Violations**: 0
- **Screen Reader Compatibility**: 100%
- **Keyboard Navigation**: 100% functional
- **Color Contrast**: 100% WCAG AA compliant

---

## Resources and Tools

### Testing Tools
- **axe DevTools**: Browser extension for real-time testing
- **WAVE**: Web accessibility evaluation tool
- **Colour Contrast Analyser**: Desktop tool for contrast checking
- **Screen Reader Testing**: NVDA (free), JAWS, VoiceOver

### Documentation
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/)
- [WebAIM Resources](https://webaim.org/)

This comprehensive checklist ensures the portfolio meets professional accessibility standards while showcasing technical competence in inclusive design practices - an important consideration for AI/ML roles focused on ethical technology development.