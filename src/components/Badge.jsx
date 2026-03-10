import React from 'react';
import { motion } from 'framer-motion';

const Badge = ({
  text,
  variant = 'default', // 'default', 'tech', 'status', 'priority', 'success', 'warning', 'error'
  size = 'medium', // 'small', 'medium', 'large'
  icon = null,
  removable = false,
  onRemove = null,
  className = "",
  ...props
}) => {
  // Variant styling
  const variantStyles = {
    default: {
      container: "bg-slate-100 text-slate-700 border-slate-200",
      hover: "hover:bg-slate-200"
    },
    tech: {
      container: "bg-blue-50 text-blue-800 border-blue-200",
      hover: "hover:bg-blue-100"
    },
    status: {
      container: "bg-green-50 text-green-800 border-green-200",
      hover: "hover:bg-green-100"
    },
    priority: {
      container: "bg-purple-50 text-purple-800 border-purple-200",
      hover: "hover:bg-purple-100"
    },
    success: {
      container: "bg-green-50 text-green-800 border-green-200",
      hover: "hover:bg-green-100"
    },
    warning: {
      container: "bg-yellow-50 text-yellow-800 border-yellow-200",
      hover: "hover:bg-yellow-100"
    },
    error: {
      container: "bg-red-50 text-red-800 border-red-200",
      hover: "hover:bg-red-100"
    }
  };

  // Size styling
  const sizeStyles = {
    small: "px-2 py-1 text-xs",
    medium: "px-3 py-1.5 text-sm",
    large: "px-4 py-2 text-base"
  };

  const currentVariant = variantStyles[variant] || variantStyles.default;
  const currentSize = sizeStyles[size] || sizeStyles.medium;

  const badgeVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        type: "spring",
        stiffness: 300,
        damping: 20
      }
    },
    hover: {
      scale: 1.05,
      transition: { duration: 0.1 }
    },
    exit: {
      scale: 0.8,
      opacity: 0,
      transition: { duration: 0.2 }
    }
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    if (onRemove) {
      onRemove(text);
    }
  };

  return (
    <motion.span
      variants={badgeVariants}
      initial="initial"
      animate="animate"
      whileHover="hover"
      exit="exit"
      className={`
        inline-flex items-center gap-1.5 font-medium rounded-full border
        transition-colors duration-200 cursor-default
        ${currentVariant.container} ${currentVariant.hover} ${currentSize} ${className}
      `}
      role="status"
      aria-label={`Badge: ${text}`}
      {...props}
    >
      {/* Icon */}
      {icon && (
        <span className="flex items-center">
          {typeof icon === 'string' ? (
            <span className="text-xs">{icon}</span>
          ) : (
            icon
          )}
        </span>
      )}

      {/* Text */}
      <span className="whitespace-nowrap">{text}</span>

      {/* Remove Button */}
      {removable && (
        <button
          onClick={handleRemove}
          className="ml-1 p-0.5 rounded-full hover:bg-black/10 transition-colors duration-150"
          aria-label={`Remove ${text} badge`}
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </motion.span>
  );
};

// Predefined badge components for common use cases
export const TechBadge = ({ text, ...props }) => (
  <Badge text={text} variant="tech" {...props} />
);

export const SkillBadge = ({ text, level, ...props }) => {
  const getLevelColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'expert': return 'success';
      case 'advanced': return 'priority';
      case 'intermediate': return 'warning';
      case 'beginner': return 'default';
      default: return 'tech';
    }
  };

  return (
    <Badge 
      text={text} 
      variant={getLevelColor(level)}
      icon={level && (
        <span className="text-xs font-bold">
          {level.charAt(0).toUpperCase()}
        </span>
      )}
      {...props}
    />
  );
};

export const StatusBadge = ({ text, status, ...props }) => {
  const statusIcons = {
    active: '🟢',
    inactive: '⚪',
    completed: '✅',
    in_progress: '🟡',
    error: '🔴',
    warning: '🟠',
    success: '🟢'
  };

  const statusVariants = {
    active: 'success',
    inactive: 'default',
    completed: 'success',
    in_progress: 'warning',
    error: 'error',
    warning: 'warning',
    success: 'success'
  };

  return (
    <Badge
      text={text}
      variant={statusVariants[status] || 'default'}
      icon={statusIcons[status]}
      {...props}
    />
  );
};

export const MetricBadge = ({ label, value, unit = '', ...props }) => (
  <Badge
    text={`${label}: ${value}${unit}`}
    variant="status"
    {...props}
  />
);

export const CategoryBadge = ({ text, color, ...props }) => {
  const categoryColors = {
    ai: 'tech',
    ml: 'priority',
    data: 'success',
    web: 'warning',
    mobile: 'error',
    backend: 'default'
  };

  return (
    <Badge
      text={text}
      variant={categoryColors[color] || color || 'default'}
      {...props}
    />
  );
};

export const PriorityBadge = ({ priority, ...props }) => {
  const priorities = {
    high: { text: 'High Priority', variant: 'error', icon: '🔥' },
    medium: { text: 'Medium Priority', variant: 'warning', icon: '⚡' },
    low: { text: 'Low Priority', variant: 'default', icon: '📝' }
  };

  const config = priorities[priority?.toLowerCase()] || priorities.medium;

  return (
    <Badge
      text={config.text}
      variant={config.variant}
      icon={config.icon}
      {...props}
    />
  );
};

// Badge group component for managing multiple badges
export const BadgeGroup = ({ badges = [], className = "", onRemove, ...props }) => {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`} {...props}>
      {badges.map((badge, index) => {
        if (typeof badge === 'string') {
          return (
            <Badge
              key={index}
              text={badge}
              removable={!!onRemove}
              onRemove={onRemove}
            />
          );
        }

        return (
          <Badge
            key={index}
            removable={!!onRemove}
            onRemove={onRemove}
            {...badge}
          />
        );
      })}
    </div>
  );
};

export default Badge;