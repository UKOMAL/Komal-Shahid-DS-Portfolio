/**
 * Hero.jsx
 * Full-bleed hero section for AI/ML portfolio.
 * Requirements:
 *   - Role label visible on mount
 *   - Top metric pill visible < 3s on mobile (no heavy image blocking render)
 *   - Primary CTA: "View Case Study", Secondary CTA: "Run Demo"
 *   - WCAG AA contrast on all text (dark bg + white text)
 */

import React from "react";
import MetricPill from "./MetricPill";
import CTAButton from "./CTAButton";

/**
 * @typedef {Object} HeroProps
 * @property {string}   roleLabel       - e.g. "AI Engineer · ML Engineer · Data Scientist"
 * @property {string}   tagline         - Primary headline tagline
 * @property {string}   metricValue     - e.g. "0.886"
 * @property {string}   metricLabel     - e.g. "AUC-ROC"
 * @property {string}   metricDelta     - e.g. "+0.11 vs baseline"
 * @property {string}   metricTooltip   - Calculation explanation shown on hover
 * @property {string}   primaryCTAText  - e.g. "View Case Study"
 * @property {string}   primaryCTAHref  - e.g. "/projects/fraud-detection"
 * @property {string}   secondaryCTAText- e.g. "Run Demo"
 * @property {string}   secondaryCTAHref- e.g. "/projects/fraud-detection#demo"
 * @property {string}   [bgImage]       - Optional hero background image URL
 */

export default function Hero({
  roleLabel = "AI Engineer · ML Engineer · Data Scientist",
  tagline = "Building AI that detects, protects, and predicts — with measurable impact.",
  metricValue = "0.886",
  metricLabel = "AUC-ROC",
  metricDelta = "+0.11 vs baseline",
  metricTooltip = "Area under ROC curve on 800K+ real-world fraud transactions, 5-fold stratified CV",
  primaryCTAText = "View Case Study",
  primaryCTAHref = "/projects/fraud-detection",
  secondaryCTAText = "Run Demo",
  secondaryCTAHref = "/projects/fraud-detection#demo",
  bgImage,
}) {
  return (
    <section
      className="hero"
      aria-label="Portfolio hero"
      style={bgImage ? { backgroundImage: `url(${bgImage})` } : undefined}
    >
      {/* Overlay ensures WCAG AA contrast regardless of bg image */}
      <div className="hero__overlay" aria-hidden="true" />

      <div className="hero__content">
        {/* Role label — loaded immediately, no JS dependency */}
        <p className="hero__role" aria-label="Current role targets">
          {roleLabel}
        </p>

        {/* H1 tagline */}
        <h1 className="hero__tagline">{tagline}</h1>

        {/* Top metric — inline, no lazy-load, renders with initial HTML */}
        <div className="hero__metric" aria-label="Top model metric">
          <MetricPill
            label={metricLabel}
            value={metricValue}
            delta={metricDelta}
            tooltip={metricTooltip}
          />
        </div>

        {/* Dual CTAs */}
        <div className="hero__ctas" role="group" aria-label="Primary actions">
          <CTAButton
            variant="primary"
            href={primaryCTAHref}
            data-analytics-event="cta_click"
            data-analytics-label={primaryCTAText}
          >
            {primaryCTAText}
          </CTAButton>

          <CTAButton
            variant="secondary"
            href={secondaryCTAHref}
            data-analytics-event="cta_click"
            data-analytics-label={secondaryCTAText}
          >
            {secondaryCTAText}
          </CTAButton>
        </div>
      </div>

      {/* CSS defined in assets/tokens.css — see Hero section */}
      <style>{`
        .hero {
          position: relative;
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: var(--color-neutral-dark);
          background-size: cover;
          background-position: center;
          padding: var(--space-8) var(--space-4);
        }

        .hero__overlay {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            135deg,
            rgba(11, 61, 145, 0.92) 0%,
            rgba(15, 23, 36, 0.85) 100%
          );
        }

        .hero__content {
          position: relative;
          z-index: 1;
          text-align: center;
          max-width: 780px;
          margin: 0 auto;
        }

        .hero__role {
          font-size: var(--text-sm);
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--color-accent);
          margin-bottom: var(--space-3);
        }

        .hero__tagline {
          font-size: var(--text-h1);
          font-weight: 700;
          color: var(--color-neutral-light);
          line-height: 1.15;
          margin-bottom: var(--space-5);
        }

        .hero__metric {
          display: flex;
          justify-content: center;
          margin-bottom: var(--space-6);
        }

        .hero__ctas {
          display: flex;
          gap: var(--space-3);
          justify-content: center;
          flex-wrap: wrap;
        }

        @media (max-width: 640px) {
          .hero__tagline {
            font-size: 2rem;
          }
          .hero__ctas {
            flex-direction: column;
            align-items: center;
          }
        }
      `}</style>
    </section>
  );
}
