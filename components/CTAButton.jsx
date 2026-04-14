/**
 * CTAButton.jsx
 * Primary, secondary, and ghost CTA button variants.
 * Renders as <a> when href is provided, <button> otherwise.
 * WCAG: focus-visible outline, minimum 44×44px touch target on mobile.
 */

import React from "react";

/**
 * @typedef {Object} CTAButtonProps
 * @property {"primary"|"secondary"|"ghost"}  [variant]   - Visual style
 * @property {"md"|"sm"}                      [size]      - Size variant
 * @property {string}                         [href]      - Renders as anchor if provided
 * @property {string}                         [type]      - Button type (submit, button, reset)
 * @property {boolean}                        [disabled]
 * @property {React.ReactNode}                children
 * @property {Function}                       [onClick]
 */

export default function CTAButton({
  variant = "primary",
  size = "md",
  href,
  type = "button",
  disabled = false,
  children,
  onClick,
  ...rest
}) {
  const className = `cta-btn cta-btn--${variant} cta-btn--${size}${disabled ? " cta-btn--disabled" : ""}`;

  if (href && !disabled) {
    return (
      <>
        <a href={href} className={className} {...rest}>
          {children}
        </a>
        <style>{ctaStyles}</style>
      </>
    );
  }

  return (
    <>
      <button
        type={type}
        className={className}
        disabled={disabled}
        onClick={onClick}
        aria-disabled={disabled}
        {...rest}
      >
        {children}
      </button>
      <style>{ctaStyles}</style>
    </>
  );
}

const ctaStyles = `
  .cta-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-family: var(--font-sans);
    font-weight: 700;
    border-radius: 8px;
    border: 2px solid transparent;
    text-decoration: none;
    cursor: pointer;
    transition: background 0.2s, color 0.2s, border-color 0.2s, box-shadow 0.2s, transform 0.15s;
    white-space: nowrap;
    min-height: 44px; /* WCAG touch target */
  }

  .cta-btn:focus-visible {
    outline: 3px solid var(--color-accent);
    outline-offset: 3px;
  }

  /* Size variants */
  .cta-btn--md {
    padding: 12px 28px;
    font-size: var(--text-body);
  }

  .cta-btn--sm {
    padding: 8px 18px;
    font-size: var(--text-sm);
    min-height: 36px;
  }

  /* Primary */
  .cta-btn--primary {
    background: var(--color-accent);
    color: var(--color-neutral-dark);
    border-color: var(--color-accent);
  }

  .cta-btn--primary:hover:not(.cta-btn--disabled) {
    background: #0090e0;
    border-color: #0090e0;
    box-shadow: 0 6px 20px rgba(0, 163, 255, 0.35);
    transform: translateY(-1px);
  }

  /* Secondary */
  .cta-btn--secondary {
    background: transparent;
    color: var(--color-neutral-light);
    border-color: var(--color-neutral-light);
  }

  .cta-btn--secondary:hover:not(.cta-btn--disabled) {
    background: rgba(255,255,255,0.1);
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    transform: translateY(-1px);
  }

  /* Ghost — for use on light backgrounds */
  .cta-btn--ghost {
    background: transparent;
    color: var(--color-primary);
    border-color: var(--color-primary);
  }

  .cta-btn--ghost:hover:not(.cta-btn--disabled) {
    background: var(--color-primary);
    color: #fff;
    transform: translateY(-1px);
  }

  /* Disabled */
  .cta-btn--disabled {
    opacity: 0.45;
    cursor: not-allowed;
    pointer-events: none;
  }

  @media (max-width: 640px) {
    .cta-btn--md {
      padding: 12px 20px;
      font-size: var(--text-sm);
    }
  }
`;
