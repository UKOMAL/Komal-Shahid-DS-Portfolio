/**
 * Nav.jsx
 * Sticky navigation bar with keyboard-accessible links and mobile hamburger menu.
 * WCAG: skip-to-content link, aria-current for active route, focus-visible outlines.
 */

import React, { useState, useEffect } from "react";
import CTAButton from "./CTAButton";

/**
 * @typedef {Object} NavProps
 * @property {string}  currentRoute  - Active route, e.g. "/projects"
 * @property {boolean} [transparent] - Transparent background until scrolled
 */

const NAV_LINKS = [
  { label: "Projects", href: "/projects" },
  { label: "About", href: "/about" },
  { label: "Resume", href: "/resume" },
  { label: "Blog", href: "/blog" },
];

export default function Nav({ currentRoute = "/", transparent = false }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    if (!transparent) return;
    const handleScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [transparent]);

  const isTransparent = transparent && !scrolled && !menuOpen;

  return (
    <>
      {/* WCAG: Skip-to-main-content link */}
      <a className="nav__skip-link" href="#main-content">
        Skip to main content
      </a>

      <header
        className={`nav ${isTransparent ? "nav--transparent" : "nav--solid"}`}
        role="banner"
      >
        <div className="nav__inner">
          {/* Logo / Home link */}
          <a
            href="/"
            className="nav__logo"
            aria-label="Komal Shahid — go to home page"
          >
            <span className="nav__logo-name">Komal Shahid</span>
            <span className="nav__logo-role" aria-hidden="true">AI Engineer</span>
          </a>

          {/* Desktop navigation */}
          <nav
            className="nav__links"
            aria-label="Primary navigation"
            role="navigation"
          >
            {NAV_LINKS.map(({ label, href }) => (
              <a
                key={href}
                href={href}
                className={`nav__link ${currentRoute === href ? "nav__link--active" : ""}`}
                aria-current={currentRoute === href ? "page" : undefined}
              >
                {label}
              </a>
            ))}
          </nav>

          {/* CTA */}
          <div className="nav__cta">
            <CTAButton
              variant="primary"
              size="sm"
              href="https://calendly.com/komalshahid"
              data-analytics-event="cta_click"
              data-analytics-label="Schedule Interview"
            >
              Schedule Interview
            </CTAButton>
          </div>

          {/* Mobile hamburger */}
          <button
            className="nav__hamburger"
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={menuOpen}
            aria-controls="mobile-nav"
            onClick={() => setMenuOpen((v) => !v)}
            type="button"
          >
            <span className="nav__hamburger-bar" aria-hidden="true" />
            <span className="nav__hamburger-bar" aria-hidden="true" />
            <span className="nav__hamburger-bar" aria-hidden="true" />
          </button>
        </div>

        {/* Mobile nav drawer */}
        <nav
          id="mobile-nav"
          className={`nav__mobile ${menuOpen ? "nav__mobile--open" : ""}`}
          aria-label="Mobile navigation"
          role="navigation"
          aria-hidden={!menuOpen}
        >
          {NAV_LINKS.map(({ label, href }) => (
            <a
              key={href}
              href={href}
              className={`nav__mobile-link ${currentRoute === href ? "nav__mobile-link--active" : ""}`}
              aria-current={currentRoute === href ? "page" : undefined}
              onClick={() => setMenuOpen(false)}
              tabIndex={menuOpen ? 0 : -1}
            >
              {label}
            </a>
          ))}
          <CTAButton
            variant="primary"
            size="sm"
            href="https://calendly.com/komalshahid"
          >
            Schedule Interview
          </CTAButton>
        </nav>
      </header>

      <style>{`
        .nav__skip-link {
          position: absolute;
          left: -9999px;
          top: auto;
          width: 1px;
          height: 1px;
          overflow: hidden;
          z-index: 10000;
          background: var(--color-accent);
          color: var(--color-neutral-dark);
          font-weight: 700;
          padding: 8px 16px;
          border-radius: 0 0 8px 0;
        }

        .nav__skip-link:focus {
          position: fixed;
          left: 0;
          top: 0;
          width: auto;
          height: auto;
        }

        .nav {
          position: sticky;
          top: 0;
          z-index: 1000;
          transition: background 0.3s, box-shadow 0.3s;
        }

        .nav--solid {
          background: #fff;
          box-shadow: 0 1px 16px rgba(11, 61, 145, 0.08);
        }

        .nav--transparent {
          background: transparent;
        }

        .nav__inner {
          max-width: 1200px;
          margin: 0 auto;
          padding: 0 var(--space-4);
          height: 64px;
          display: flex;
          align-items: center;
          gap: var(--space-4);
        }

        .nav__logo {
          text-decoration: none;
          display: flex;
          flex-direction: column;
          line-height: 1.1;
          flex-shrink: 0;
        }

        .nav__logo-name {
          font-size: 1rem;
          font-weight: 700;
          color: var(--color-primary);
        }

        .nav--transparent .nav__logo-name { color: #fff; }

        .nav__logo-role {
          font-size: 0.7rem;
          font-weight: 500;
          color: var(--color-accent);
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .nav__links {
          display: flex;
          gap: var(--space-5);
          margin-left: auto;
        }

        .nav__link {
          font-size: var(--text-sm);
          font-weight: 600;
          text-decoration: none;
          color: #4a5568;
          padding-bottom: 2px;
          border-bottom: 2px solid transparent;
          transition: color 0.2s, border-color 0.2s;
        }

        .nav--transparent .nav__link { color: rgba(255,255,255,0.85); }

        .nav__link:hover,
        .nav__link:focus-visible {
          color: var(--color-primary);
          border-bottom-color: var(--color-accent);
        }

        .nav--transparent .nav__link:hover { color: #fff; }

        .nav__link--active {
          color: var(--color-primary);
          border-bottom-color: var(--color-accent);
        }

        .nav__link:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 4px; }

        .nav__cta { margin-left: var(--space-3); }

        .nav__hamburger {
          display: none;
          flex-direction: column;
          gap: 5px;
          background: none;
          border: none;
          cursor: pointer;
          padding: 8px;
          margin-left: auto;
        }

        .nav__hamburger-bar {
          display: block;
          width: 24px;
          height: 2px;
          background: var(--color-primary);
          border-radius: 2px;
          transition: transform 0.2s;
        }

        .nav__mobile {
          display: none;
          flex-direction: column;
          gap: var(--space-3);
          padding: var(--space-4);
          background: #fff;
          border-top: 1px solid #e2e8f0;
        }

        .nav__mobile--open { display: flex; }

        .nav__mobile-link {
          font-size: var(--text-body);
          font-weight: 600;
          color: #2d3748;
          text-decoration: none;
          padding: var(--space-2) 0;
        }

        .nav__mobile-link--active { color: var(--color-primary); }

        @media (max-width: 768px) {
          .nav__links { display: none; }
          .nav__cta { display: none; }
          .nav__hamburger { display: flex; }
        }
      `}</style>
    </>
  );
}
