import React, { useState } from 'react';
import { motion } from 'framer-motion';

const MetricPill = ({ 
  metric, 
  value, 
  delta = null, 
  tooltip = null,
  variant = 'default', // 'default', 'hero', 'small', 'success', 'warning'
  className = ""
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  // Variant styling
  const variantStyles = {
    default: {
      container: "bg-cyan-50 border border-cyan-200 text-cyan-800",
      metric: "text-cyan-600",
      value: "text-cyan-900 font-semibold",
      size: "px-3 py-1.5 text-sm"
    },
    hero: {
      container: "bg-white/20 backdrop-blur-sm border border-white/30 text-white",
      metric: "text-cyan-200", 
      value: "text-white font-bold",
      size: "px-6 py-3 text-lg"
    },
    small: {
      container: "bg-slate-100 border border-slate-200 text-slate-700",
      metric: "text-slate-500",
      value: "text-slate-800 font-medium",
      size: "px-2 py-1 text-xs"
    },
    success: {
      container: "bg-green-50 border border-green-200 text-green-800",
      metric: "text-green-600",
      value: "text-green-900 font-semibold", 
      size: "px-3 py-1.5 text-sm"
    },
    warning: {
      container: "bg-orange-50 border border-orange-200 text-orange-800",
      metric: "text-orange-600",
      value: "text-orange-900 font-semibold",
      size: "px-3 py-1.5 text-sm"
    }
  };

  const currentStyle = variantStyles[variant] || variantStyles.default;

  const pillVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        type: "spring",
        stiffness: 200,
        damping: 20
      }
    },
    hover: {
      scale: 1.05,
      transition: { duration: 0.2 }
    }
  };

  const formatValue = (val) => {
    // Handle different value formats
    if (typeof val === 'number') {
      if (val < 1) return val.toFixed(2); // For AUC, R² values
      if (val >= 1000) return `${(val/1000).toFixed(1)}K`; // For large numbers
      return val.toFixed(1);
    }
    return val;
  };

  const getDeltaColor = (delta) => {
    if (!delta) return '';
    const numDelta = parseFloat(delta.toString().replace(/[%+]/g, ''));
    return numDelta > 0 ? 'text-green-600' : 'text-red-600';
  };

  const getDeltaIcon = (delta) => {
    if (!delta) return null;
    const numDelta = parseFloat(delta.toString().replace(/[%+]/g, ''));
    return numDelta > 0 ? '↗' : '↘';
  };

  return (
    <div className="relative inline-block">
      <motion.div
        variants={pillVariants}
        initial="initial"
        animate="animate"
        whileHover="hover"
        className={`
          inline-flex items-center gap-2 rounded-full
          transition-all duration-200 cursor-default
          ${currentStyle.container} ${currentStyle.size} ${className}
        `}
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        role="status"
        aria-label={`${metric}: ${value}${delta ? `, change: ${delta}` : ''}`}
      >
        {/* Metric Label */}
        <span className={`text-xs font-medium uppercase tracking-wider ${currentStyle.metric}`}>
          {metric}
        </span>

        {/* Value */}
        <span className={currentStyle.value}>
          {formatValue(value)}
        </span>

        {/* Delta (if provided) */}
        {delta && (
          <span className={`text-xs font-medium ${getDeltaColor(delta)}`}>
            <span className="mr-1">{getDeltaIcon(delta)}</span>
            {delta}
          </span>
        )}

        {/* Info Icon (if tooltip available) */}
        {tooltip && (
          <svg 
            className={`w-3 h-3 ${currentStyle.metric} opacity-60`} 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
            />
          </svg>
        )}
      </motion.div>

      {/* Tooltip */}
      {tooltip && showTooltip && (
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.9 }}
          className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50"
        >
          <div className="bg-slate-900 text-white text-xs rounded-lg py-2 px-3 max-w-xs text-center shadow-lg">
            {tooltip}
            {/* Tooltip Arrow */}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2">
              <div className="border-4 border-transparent border-t-slate-900"></div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

// Predefined metric pills for common use cases
export const AUCPill = ({ value, tooltip }) => (
  <MetricPill
    metric="AUC"
    value={value}
    tooltip={tooltip || `Area Under Curve: ${value} indicates ${value > 0.9 ? 'excellent' : value > 0.8 ? 'good' : 'moderate'} model performance`}
    variant="success"
  />
);

export const F1Pill = ({ value, tooltip }) => (
  <MetricPill
    metric="F1"
    value={value}
    tooltip={tooltip || `F1 Score: ${value} balances precision and recall`}
    variant="default"
  />
);

export const AccuracyPill = ({ value, tooltip }) => (
  <MetricPill
    metric="ACC"
    value={`${(value * 100).toFixed(1)}%`}
    tooltip={tooltip || `Model accuracy: ${(value * 100).toFixed(1)}%`}
    variant="success"
  />
);

export const R2Pill = ({ value, tooltip }) => (
  <MetricPill
    metric="R²"
    value={value}
    tooltip={tooltip || `Coefficient of Determination: ${value} explains ${(value * 100).toFixed(0)}% of variance`}
    variant="default"
  />
);

export default MetricPill;