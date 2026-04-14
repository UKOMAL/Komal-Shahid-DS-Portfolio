/**
 * MetricPill.jsx
 * Displays a metric value with optional delta and tooltip showing calculation method.
 * WCAG: tooltip accessible via aria-describedby; focus-visible outline.
 */

import React, { useState, useId } from "react";

/**
 * @typedef {Object} MetricPillProps
 * @property {string} label    - Metric name, e.g. "AUC-ROC"
 * @property {string} value    - Metric value, e.g. "0.886"
 * @property {string} [delta]  - Change indicator, e.g. "+0.11 vs baseline"
 * @property {string} [tooltip]- Calculation explanation shown on hover/focus
 * @property {"light"|"dark"}  [theme] - Color theme; "light" = white pill (for dark bg)
 */

export default function MetricPill({ label, value, delta, tooltip, theme = "light" }) {
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const tooltipId = useId();

  const isDeltaPositive = delta && delta.startsWith("+");
  const isDeltaNegative = delta && delta.startsWith("-");

  return (
    <div
      className={`metric-pill metric-pill--${theme}`}
      role="group"
      aria-label={`${label}: ${value}${delta ? `, delta ${delta}` : ""}`}
    >
      <span className="metric-pill__label">{label}</span>
      <span className="metric-pill__value">{value}</span>

      {delta && (
        <span
          className={`metric-pill__delta ${
            isDeltaPositive
              ? "metric-pill__delta--positive"
              : isDeltaNegative
              ? "metric-pill__delta--negative"
              : ""
          }`}
          aria-label={`Change: ${delta}`}
        >
          {delta}
        </span>
      )}

      {tooltip && (
        <button
          className="metric-pill__tooltip-trigger"
          aria-describedby={tooltipId}
          aria-label="Show metric calculation details"
          onMouseEnter={() => setTooltipVisible(true)}
          onMouseLeave={() => setTooltipVisible(false)}
          onFocus={() => setTooltipVisible(true)}
          onBlur={() => setTooltipVisible(false)}
          type="button"
        >
          <span aria-hidden="true">ⓘ</span>
          <span
            id={tooltipId}
            role="tooltip"
            className={`metric-pill__tooltip ${tooltipVisible ? "metric-pill__tooltip--visible" : ""}`}
          >
            {tooltip}
          </span>
        </button>
      )}

      <style>{`
        .metric-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border-radius: 9999px;
          padding: 8px 16px;
          font-size: var(--text-sm);
          font-weight: 600;
          position: relative;
          white-space: nowrap;
        }

        .metric-pill--light {
          background: rgba(255, 255, 255, 0.12);
          border: 1px solid rgba(255, 255, 255, 0.3);
          color: #fff;
        }

        .metric-pill--dark {
          background: var(--color-primary);
          border: 1px solid var(--color-primary);
          color: #fff;
        }

        .metric-pill__label {
          opacity: 0.8;
          font-size: 0.75rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .metric-pill__value {
          font-size: 1.1rem;
          font-weight: 700;
        }

        .metric-pill__delta {
          font-size: 0.75rem;
          padding: 2px 6px;
          border-radius: 4px;
          background: rgba(0,0,0,0.2);
        }

        .metric-pill__delta--positive { color: #68d391; }
        .metric-pill__delta--negative { color: #fc8181; }

        .metric-pill__tooltip-trigger {
          background: none;
          border: none;
          cursor: pointer;
          color: inherit;
          opacity: 0.7;
          padding: 0 2px;
          font-size: 0.85rem;
          position: relative;
          display: inline-flex;
          align-items: center;
        }

        .metric-pill__tooltip-trigger:focus-visible {
          outline: 2px solid var(--color-accent);
          border-radius: 4px;
        }

        .metric-pill__tooltip {
          position: absolute;
          bottom: calc(100% + 8px);
          left: 50%;
          transform: translateX(-50%);
          background: var(--color-neutral-dark);
          color: var(--color-neutral-light);
          border-radius: 6px;
          padding: 8px 12px;
          font-size: 0.75rem;
          font-weight: 400;
          max-width: 260px;
          white-space: normal;
          text-align: center;
          box-shadow: 0 4px 16px rgba(0,0,0,0.25);
          opacity: 0;
          pointer-events: none;
          transition: opacity 0.2s;
          z-index: 200;
        }

        .metric-pill__tooltip--visible {
          opacity: 1;
          pointer-events: auto;
        }

        /* Tooltip arrow */
        .metric-pill__tooltip::after {
          content: "";
          position: absolute;
          top: 100%;
          left: 50%;
          transform: translateX(-50%);
          border: 5px solid transparent;
          border-top-color: var(--color-neutral-dark);
        }
      `}</style>
    </div>
  );
}
