/**
 * ProjectCard.jsx
 * Card for the projects index page.
 * Behavior: hover (or focus) reveals 3 key bullets.
 * Includes: title, headline metric, tech tags, "View Case Study" + "Run Demo" links.
 * WCAG: keyboard-accessible hover reveal via focus-within.
 */

import React, { useState } from "react";
import Badge from "./Badge";
import MetricPill from "./MetricPill";
import CTAButton from "./CTAButton";

/**
 * @typedef {Object} ProjectCardProps
 * @property {string}   title           - Project title
 * @property {string}   description     - One-line problem statement
 * @property {string}   metricLabel     - e.g. "AUC-ROC"
 * @property {string}   metricValue     - e.g. "0.886"
 * @property {string}   [metricDelta]   - Optional delta string
 * @property {string}   [metricTooltip] - Metric calculation tooltip
 * @property {string[]} tags            - Tech stack tags
 * @property {string[]} bullets         - 3 hover-reveal impact bullets
 * @property {string}   caseStudyHref   - Link to case study page
 * @property {string}   [demoHref]      - Link to live demo
 * @property {string}   [githubHref]    - Link to GitHub repo
 * @property {string}   [imageSrc]      - Optional project thumbnail URL
 * @property {string}   [imageAlt]      - Alt text for thumbnail image
 */

export default function ProjectCard({
  title,
  description,
  metricLabel,
  metricValue,
  metricDelta,
  metricTooltip,
  tags = [],
  bullets = [],
  caseStudyHref,
  demoHref,
  githubHref,
  imageSrc,
  imageAlt,
}) {
  const [revealed, setRevealed] = useState(false);

  return (
    <article
      className={`project-card ${revealed ? "project-card--revealed" : ""}`}
      onMouseEnter={() => setRevealed(true)}
      onMouseLeave={() => setRevealed(false)}
      onFocus={() => setRevealed(true)}
      onBlur={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) setRevealed(false);
      }}
      aria-label={`Project: ${title}`}
    >
      {imageSrc && (
        <img
          className="project-card__image"
          src={imageSrc}
          alt={imageAlt || `${title} project thumbnail`}
          width={400}
          height={200}
          loading="lazy"
        />
      )}

      <div className="project-card__body">
        <h3 className="project-card__title">{title}</h3>
        <p className="project-card__description">{description}</p>

        <div className="project-card__metric">
          <MetricPill
            label={metricLabel}
            value={metricValue}
            delta={metricDelta}
            tooltip={metricTooltip}
          />
        </div>

        {/* Hover-reveal bullets — visible on hover/focus via CSS + state */}
        <ul
          className="project-card__bullets"
          aria-label="Key impact bullets"
          aria-hidden={!revealed}
        >
          {bullets.slice(0, 3).map((bullet, i) => (
            <li key={i} className="project-card__bullet">
              {bullet}
            </li>
          ))}
        </ul>

        {/* Tech tags */}
        <div className="project-card__tags" aria-label="Technologies used">
          {tags.map((tag) => (
            <Badge key={tag} label={tag} />
          ))}
        </div>

        {/* CTAs */}
        <div className="project-card__actions" role="group" aria-label={`Actions for ${title}`}>
          {caseStudyHref && (
            <CTAButton
              variant="primary"
              href={caseStudyHref}
              size="sm"
              data-analytics-event="cta_click"
              data-analytics-label="View Case Study"
              data-analytics-project={title}
            >
              View Case Study
            </CTAButton>
          )}
          {demoHref && (
            <CTAButton
              variant="secondary"
              href={demoHref}
              size="sm"
              data-analytics-event="cta_click"
              data-analytics-label="Run Demo"
              data-analytics-project={title}
            >
              Run Demo ▶
            </CTAButton>
          )}
          {githubHref && (
            <a
              className="project-card__github-link"
              href={githubHref}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`View ${title} on GitHub`}
              data-analytics-event="github_click"
              data-analytics-project={title}
            >
              GitHub →
            </a>
          )}
        </div>
      </div>

      <style>{`
        .project-card {
          background: #fff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          overflow: hidden;
          transition: box-shadow 0.25s ease, transform 0.25s ease;
          display: flex;
          flex-direction: column;
        }

        .project-card:hover,
        .project-card:focus-within {
          box-shadow: 0 12px 40px rgba(11, 61, 145, 0.15);
          transform: translateY(-4px);
        }

        .project-card__image {
          width: 100%;
          height: 200px;
          object-fit: cover;
        }

        .project-card__body {
          padding: var(--space-5);
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
          flex: 1;
        }

        .project-card__title {
          font-size: var(--text-h3);
          font-weight: 700;
          color: var(--color-primary);
          margin: 0;
        }

        .project-card__description {
          font-size: var(--text-body);
          color: #4a5568;
          margin: 0;
        }

        /* Hover-reveal bullets */
        .project-card__bullets {
          list-style: none;
          padding: 0;
          margin: 0;
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          max-height: 0;
          overflow: hidden;
          opacity: 0;
          transition: max-height 0.3s ease, opacity 0.3s ease;
        }

        .project-card--revealed .project-card__bullets,
        .project-card:focus-within .project-card__bullets {
          max-height: 200px;
          opacity: 1;
        }

        .project-card__bullet {
          font-size: var(--text-sm);
          color: #2d3748;
          padding-left: 1.2em;
          position: relative;
        }

        .project-card__bullet::before {
          content: "▸";
          position: absolute;
          left: 0;
          color: var(--color-accent);
        }

        .project-card__tags {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
        }

        .project-card__actions {
          display: flex;
          gap: var(--space-2);
          flex-wrap: wrap;
          align-items: center;
          margin-top: auto;
        }

        .project-card__github-link {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--color-primary);
          text-decoration: none;
          padding: 4px 0;
          border-bottom: 1px solid transparent;
          transition: border-color 0.2s;
        }

        .project-card__github-link:hover,
        .project-card__github-link:focus {
          border-bottom-color: var(--color-primary);
        }
      `}</style>
    </article>
  );
}
