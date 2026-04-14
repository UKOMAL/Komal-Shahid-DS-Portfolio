/**
 * CaseStudyLayout.jsx
 * Full-page layout wrapper for project case studies.
 * Features:
 *   - Collapsible reproducibility section
 *   - Sticky section nav (Problem · Dataset · Approach · Results · Fairness · Demo)
 *   - MetricPill strip at the top
 *   - "View Case Study", "Run Demo", "Download Report", "GitHub" CTAs
 */

import React, { useState } from "react";
import MetricPill from "./MetricPill";
import CTAButton from "./CTAButton";
import Badge from "./Badge";

/**
 * @typedef {Object} CaseStudyMetric
 * @property {string} label
 * @property {string} value
 * @property {string} [delta]
 * @property {string} [tooltip]
 */

/**
 * @typedef {Object} CaseStudyLayoutProps
 * @property {string}            title           - Project title
 * @property {string}            subtitle        - One-line project tagline
 * @property {CaseStudyMetric[]} metrics         - Top metric pills (2–4)
 * @property {string[]}          tags            - Tech stack tags
 * @property {string}            [demoHref]      - Demo URL
 * @property {string}            [githubHref]    - GitHub URL
 * @property {string}            [reportHref]    - Download report URL
 * @property {React.ReactNode}   children        - Case study body content
 * @property {React.ReactNode}   [reproducibility] - Collapsible reproducibility content
 */

const SECTION_IDS = ["problem", "dataset", "approach", "results", "fairness", "demo", "reproducibility", "lessons"];

export default function CaseStudyLayout({
  title,
  subtitle,
  metrics = [],
  tags = [],
  demoHref,
  githubHref,
  reportHref,
  children,
  reproducibility,
}) {
  const [reproOpen, setReproOpen] = useState(false);

  return (
    <article className="case-study">
      {/* Header */}
      <header className="case-study__header">
        <div className="case-study__header-content">
          <h1 className="case-study__title">{title}</h1>
          <p className="case-study__subtitle">{subtitle}</p>

          {/* Metric pills strip */}
          {metrics.length > 0 && (
            <div className="case-study__metrics" aria-label="Key metrics">
              {metrics.map((m, i) => (
                <MetricPill
                  key={i}
                  label={m.label}
                  value={m.value}
                  delta={m.delta}
                  tooltip={m.tooltip}
                />
              ))}
            </div>
          )}

          {/* Tech tags */}
          <div className="case-study__tags">
            {tags.map((t) => (
              <Badge key={t} label={t} />
            ))}
          </div>

          {/* CTA strip */}
          <div className="case-study__ctas" role="group" aria-label="Case study actions">
            <CTAButton
              variant="primary"
              href="#problem"
              data-analytics-event="cta_click"
              data-analytics-label="View Case Study"
            >
              View Case Study ↓
            </CTAButton>
            {demoHref && (
              <CTAButton
                variant="secondary"
                href={demoHref}
                data-analytics-event="cta_click"
                data-analytics-label="Run Demo"
              >
                Run Demo ▶
              </CTAButton>
            )}
            {reportHref && (
              <CTAButton
                variant="ghost"
                href={reportHref}
                data-analytics-event="cta_click"
                data-analytics-label="Download Report"
              >
                Download Report ↓
              </CTAButton>
            )}
            {githubHref && (
              <a
                className="case-study__github"
                href={githubHref}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View project on GitHub"
                data-analytics-event="github_click"
              >
                GitHub →
              </a>
            )}
          </div>
        </div>
      </header>

      {/* Sticky section nav */}
      <nav className="case-study__section-nav" aria-label="Case study sections">
        {SECTION_IDS.map((id) => (
          <a key={id} href={`#${id}`} className="case-study__section-link">
            {id.charAt(0).toUpperCase() + id.slice(1)}
          </a>
        ))}
      </nav>

      {/* Body content — markdown/JSX sections passed as children */}
      <div className="case-study__body">{children}</div>

      {/* Collapsible Reproducibility Section */}
      {reproducibility && (
        <section id="reproducibility" className="case-study__repro">
          <button
            className="case-study__repro-toggle"
            aria-expanded={reproOpen}
            aria-controls="repro-content"
            onClick={() => setReproOpen((v) => !v)}
          >
            <span>Reproducibility &amp; Artifacts</span>
            <span className="case-study__repro-icon" aria-hidden="true">
              {reproOpen ? "▲" : "▼"}
            </span>
          </button>

          <div
            id="repro-content"
            className={`case-study__repro-content ${reproOpen ? "case-study__repro-content--open" : ""}`}
            aria-hidden={!reproOpen}
          >
            {reproducibility}
          </div>
        </section>
      )}

      <style>{`
        .case-study__header {
          background: linear-gradient(135deg, var(--color-primary) 0%, var(--color-neutral-dark) 100%);
          color: var(--color-neutral-light);
          padding: var(--space-8) var(--space-4);
          text-align: center;
        }

        .case-study__header-content {
          max-width: 820px;
          margin: 0 auto;
        }

        .case-study__title {
          font-size: var(--text-h1);
          font-weight: 700;
          margin-bottom: var(--space-3);
        }

        .case-study__subtitle {
          font-size: var(--text-h3);
          opacity: 0.85;
          margin-bottom: var(--space-5);
        }

        .case-study__metrics {
          display: flex;
          justify-content: center;
          flex-wrap: wrap;
          gap: var(--space-3);
          margin-bottom: var(--space-4);
        }

        .case-study__tags {
          display: flex;
          justify-content: center;
          flex-wrap: wrap;
          gap: var(--space-2);
          margin-bottom: var(--space-5);
        }

        .case-study__ctas {
          display: flex;
          gap: var(--space-3);
          justify-content: center;
          flex-wrap: wrap;
        }

        .case-study__github {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--color-neutral-light);
          text-decoration: none;
          align-self: center;
          opacity: 0.85;
          transition: opacity 0.2s;
        }

        .case-study__github:hover { opacity: 1; }

        .case-study__section-nav {
          position: sticky;
          top: 0;
          z-index: 100;
          background: #fff;
          border-bottom: 1px solid #e2e8f0;
          display: flex;
          gap: var(--space-4);
          padding: var(--space-2) var(--space-4);
          overflow-x: auto;
        }

        .case-study__section-link {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--color-primary);
          text-decoration: none;
          white-space: nowrap;
          padding: var(--space-1) 0;
          border-bottom: 2px solid transparent;
          transition: border-color 0.2s;
        }

        .case-study__section-link:hover,
        .case-study__section-link:focus {
          border-bottom-color: var(--color-accent);
        }

        .case-study__body {
          max-width: 820px;
          margin: 0 auto;
          padding: var(--space-8) var(--space-4);
        }

        .case-study__repro {
          max-width: 820px;
          margin: 0 auto var(--space-8);
          padding: 0 var(--space-4);
        }

        .case-study__repro-toggle {
          width: 100%;
          display: flex;
          justify-content: space-between;
          align-items: center;
          background: var(--color-primary);
          color: #fff;
          border: none;
          border-radius: 8px;
          padding: var(--space-3) var(--space-4);
          font-size: var(--text-body);
          font-weight: 700;
          cursor: pointer;
          transition: background 0.2s;
        }

        .case-study__repro-toggle:hover { background: var(--color-accent); }

        .case-study__repro-content {
          max-height: 0;
          overflow: hidden;
          opacity: 0;
          transition: max-height 0.4s ease, opacity 0.3s ease;
          padding: 0 var(--space-4);
          background: var(--color-neutral-light);
          border-radius: 0 0 8px 8px;
        }

        .case-study__repro-content--open {
          max-height: 600px;
          opacity: 1;
          padding: var(--space-4);
        }

        @media (max-width: 640px) {
          .case-study__title { font-size: 1.75rem; }
          .case-study__section-nav { gap: var(--space-3); }
        }
      `}</style>
    </article>
  );
}
