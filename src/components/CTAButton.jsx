import React from 'react';
import { motion } from 'framer-motion';

const CTAButton = ({
  text,
  href,
  variant = 'primary', // 'primary', 'secondary', 'ghost', 'danger'
  size = 'medium', // 'small', 'medium', 'large'
  icon = null,
  iconPosition = 'left', // 'left', 'right'
  disabled = false,
  loading = false,
  analytics = '',
  onClick = null,
  className = '',
  ...props
}) => {
  // Variant styling
  const variantStyles = {
    primary: {
      base: "bg-blue-600 text-white border-2 border-blue-600 hover:bg-blue-700 hover:border-blue-700 focus:ring-blue-500",
      disabled: "bg-blue-300 border-blue-300 cursor-not-allowed"
    },
    secondary: {
      base: "bg-transparent text-blue-600 border-2 border-blue-600 hover:bg-blue-600 hover:text-white focus:ring-blue-500",
      disabled: "text-blue-300 border-blue-300 cursor-not-allowed hover:bg-transparent hover:text-blue-300"
    },
    ghost: {
      base: "bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900 focus:ring-slate-500 border-2 border-transparent",
      disabled: "text-slate-300 cursor-not-allowed hover:bg-transparent"
    },
    danger: {
      base: "bg-red-600 text-white border-2 border-red-600 hover:bg-red-700 hover:border-red-700 focus:ring-red-500",
      disabled: "bg-red-300 border-red-300 cursor-not-allowed"
    }
  };

  // Size styling
  const sizeStyles = {
    small: "px-4 py-2 text-sm font-medium",
    medium: "px-6 py-3 text-base font-semibold",
    large: "px-8 py-4 text-lg font-semibold"
  };

  const currentVariant = variantStyles[variant] || variantStyles.primary;
  const currentSize = sizeStyles[size] || sizeStyles.medium;

  const baseClasses = `
    inline-flex items-center justify-center gap-2 
    rounded-lg transition-all duration-200 
    focus:outline-none focus:ring-2 focus:ring-offset-2
    ${currentSize}
    ${disabled ? currentVariant.disabled : currentVariant.base}
    ${className}
  `;

  const buttonVariants = {
    initial: { scale: 1 },
    hover: { 
      scale: disabled ? 1 : 1.02,
      transition: { duration: 0.1 }
    },
    tap: { 
      scale: disabled ? 1 : 0.98,
      transition: { duration: 0.1 }
    }
  };

  const handleClick = (e) => {
    if (disabled || loading) {
      e.preventDefault();
      return;
    }

    // Analytics tracking
    if (analytics && typeof gtag !== 'undefined') {
      gtag('event', 'click', {
        event_category: 'CTA',
        event_label: analytics,
        value: text
      });
    }

    if (onClick) {
      onClick(e);
    }
  };

  const LoadingSpinner = () => (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle 
        className="opacity-25" 
        cx="12" 
        cy="12" 
        r="10" 
        stroke="currentColor" 
        strokeWidth="4"
      />
      <path 
        className="opacity-75" 
        fill="currentColor" 
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  // Render as anchor tag if href is provided, otherwise as button
  const Component = href ? 'a' : 'button';
  const linkProps = href ? { href, ...(href.startsWith('http') ? { target: '_blank', rel: 'noopener noreferrer' } : {}) } : {};
  const buttonProps = href ? {} : { type: 'button', disabled: disabled || loading };

  return (
    <motion.div
      variants={buttonVariants}
      initial="initial"
      whileHover="hover"
      whileTap="tap"
    >
      <Component
        className={baseClasses}
        onClick={handleClick}
        aria-label={`${text}${analytics ? ` - ${analytics}` : ''}`}
        aria-disabled={disabled || loading}
        {...linkProps}
        {...buttonProps}
        {...props}
      >
        {/* Loading State */}
        {loading ? (
          <>
            <LoadingSpinner />
            <span>Loading...</span>
          </>
        ) : (
          <>
            {/* Icon Left */}
            {icon && iconPosition === 'left' && (
              <span className="flex items-center">
                {icon}
              </span>
            )}

            {/* Button Text */}
            <span>{text}</span>

            {/* Icon Right */}
            {icon && iconPosition === 'right' && (
              <span className="flex items-center">
                {icon}
              </span>
            )}

            {/* External Link Icon */}
            {href && href.startsWith('http') && (
              <svg 
                className="w-4 h-4 ml-1" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" 
                />
              </svg>
            )}
          </>
        )}
      </Component>
    </motion.div>
  );
};

// Pre-configured CTA buttons for common use cases
export const ViewCaseStudyButton = ({ href, analytics = "view_case_study", ...props }) => (
  <CTAButton
    text="View Case Study"
    href={href}
    variant="primary"
    analytics={analytics}
    icon={
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    }
    {...props}
  />
);

export const RunDemoButton = ({ href, analytics = "run_demo", ...props }) => (
  <CTAButton
    text="Run Demo"
    href={href}
    variant="secondary"
    analytics={analytics}
    icon={
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h1m4 0h1m2-5V9a3 3 0 00-3-3H8a3 3 0 00-3 3v1M7 21h10a2 2 0 002-2V9a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    }
    {...props}
  />
);

export const DownloadButton = ({ href, analytics = "download_report", ...props }) => (
  <CTAButton
    text="Download Report"
    href={href}
    variant="ghost"
    analytics={analytics}
    icon={
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    }
    {...props}
  />
);

export const ScheduleInterviewButton = ({ href, analytics = "schedule_interview", ...props }) => (
  <CTAButton
    text="Schedule Interview"
    href={href}
    variant="primary"
    analytics={analytics}
    icon={
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    }
    {...props}
  />
);

export default CTAButton;