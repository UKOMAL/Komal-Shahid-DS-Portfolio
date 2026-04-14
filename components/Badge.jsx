/**
 * Badge.jsx
 * Pill-shaped tag for skills, technologies, and domains.
 * Supports "primary", "accent", and "neutral" variants.
 */

import React from "react";

/**
 * @typedef {Object} BadgeProps
 * @property {string}                       label   - Badge text
 * @property {"primary"|"accent"|"neutral"} [variant] - Visual variant
 * @property {string}                       [href]  - Optional link URL (renders as <a>)
 */

export default function Badge({ label, variant = "primary", href }) {
  const className = `badge badge--${variant}`;

  if (href) {
    return (
      <a href={href} className={className} aria-label={`Filter by ${label}`}>
        {label}
        <style>{badgeStyles}</style>
      </a>
    );
  }

  return (
    <span className={className} aria-label={label}>
      {label}
      <style>{badgeStyles}</style>
    </span>
  );
}

const badgeStyles = `
  .badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: var(--text-sm);
    font-weight: 600;
    white-space: nowrap;
    text-decoration: none;
    transition: background 0.2s, color 0.2s;
    line-height: 1.4;
  }

  .badge--primary {
    background: rgba(11, 61, 145, 0.1);
    color: var(--color-primary);
    border: 1px solid rgba(11, 61, 145, 0.25);
  }

  .badge--primary:hover,
  .badge--primary:focus {
    background: var(--color-primary);
    color: #fff;
  }

  .badge--accent {
    background: rgba(0, 163, 255, 0.1);
    color: var(--color-accent);
    border: 1px solid rgba(0, 163, 255, 0.25);
  }

  .badge--accent:hover,
  .badge--accent:focus {
    background: var(--color-accent);
    color: #fff;
  }

  .badge--neutral {
    background: var(--color-neutral-light);
    color: #4a5568;
    border: 1px solid #e2e8f0;
  }

  .badge--neutral:hover,
  .badge--neutral:focus {
    background: #e2e8f0;
    color: #2d3748;
  }

  a.badge:focus-visible {
    outline: 2px solid var(--color-accent);
    outline-offset: 2px;
  }
`;
