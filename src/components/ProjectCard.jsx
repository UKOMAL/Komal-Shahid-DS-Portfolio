import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Badge from './Badge';
import MetricPill from './MetricPill';
import CTAButton from './CTAButton';

const ProjectCard = ({
  title,
  description,
  image,
  metrics = [],
  tags = [],
  demoLink,
  caseStudyLink,
  githubLink,
  bullets = [],
  className = ""
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const cardVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.4 }
    },
    hover: { 
      y: -4,
      transition: { duration: 0.2 }
    }
  };

  const bulletVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: { 
      opacity: 1, 
      x: 0,
      transition: { duration: 0.3 }
    }
  };

  return (
    <motion.article
      variants={cardVariants}
      initial="initial"
      animate="animate"
      whileHover="hover"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      className={`
        project-card bg-white rounded-xl shadow-lg hover:shadow-xl
        transition-shadow duration-300 overflow-hidden group
        ${className}
      `}
      role="article"
      aria-labelledby={`project-title-${title.replace(/\s+/g, '-').toLowerCase()}`}
    >
      {/* Project Image */}
      {image && (
        <div className="relative h-48 overflow-hidden">
          <img
            src={image}
            alt={`${title} project screenshot`}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            loading="lazy"
          />
          {/* Gradient overlay for better text readability */}
          <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
        </div>
      )}

      {/* Card Content */}
      <div className="p-6">
        {/* Title */}
        <h3 
          id={`project-title-${title.replace(/\s+/g, '-').toLowerCase()}`}
          className="text-xl font-bold text-slate-900 mb-3 line-clamp-2"
        >
          {title}
        </h3>

        {/* Metrics */}
        {metrics.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {metrics.map((metric, index) => (
              <MetricPill
                key={index}
                metric={metric.label}
                value={metric.value}
                tooltip={metric.tooltip}
                variant="small"
              />
            ))}
          </div>
        )}

        {/* Description */}
        <p className="text-slate-600 leading-relaxed mb-4 line-clamp-3">
          {description}
        </p>

        {/* Hover Bullets - Show on hover */}
        <motion.div
          className="mb-4 overflow-hidden"
          initial={{ height: 0 }}
          animate={{ 
            height: isHovered && bullets.length > 0 ? 'auto' : 0
          }}
          transition={{ duration: 0.3 }}
        >
          {bullets.length > 0 && (
            <ul className="space-y-1 text-sm text-slate-600 pt-2 border-t border-slate-100">
              {bullets.slice(0, 3).map((bullet, index) => (
                <motion.li
                  key={index}
                  variants={bulletVariants}
                  initial="hidden"
                  animate={isHovered ? "visible" : "hidden"}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-start gap-2"
                >
                  <span className="text-cyan-500 mt-1">•</span>
                  <span>{bullet}</span>
                </motion.li>
              ))}
            </ul>
          )}
        </motion.div>

        {/* Technology Tags */}
        {tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {tags.map((tag, index) => (
              <Badge
                key={index}
                text={tag}
                variant="tech"
                size="small"
              />
            ))}
          </div>
        )}

        {/* Action Links */}
        <div className="flex flex-col sm:flex-row gap-3">
          {caseStudyLink && (
            <CTAButton
              text="View Case Study"
              href={caseStudyLink}
              variant="primary"
              size="small"
              analytics="project_card_case_study"
              className="flex-1"
            />
          )}
          
          {demoLink && (
            <CTAButton
              text="Run Demo"
              href={demoLink}
              variant="secondary"
              size="small"
              analytics="project_card_demo"
              className="flex-1"
              icon={
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h1m4 0h1m2-5V9a3 3 0 00-3-3H8a3 3 0 00-3 3v1M7 21h10a2 2 0 002-2V9a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              }
            />
          )}

          {githubLink && (
            <a
              href={githubLink}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-slate-400 hover:text-slate-600 transition-colors duration-200"
              aria-label={`View ${title} on GitHub`}
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
            </a>
          )}
        </div>

        {/* Performance Indicators */}
        <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
          <span>Last updated: 2 weeks ago</span>
          {metrics.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-green-400 rounded-full"></span>
              Production Ready
            </span>
          )}
        </div>
      </div>

      {/* Loading State */}
      {!image && (
        <div className="h-48 bg-slate-200 animate-pulse flex items-center justify-center">
          <span className="text-slate-400">Loading image...</span>
        </div>
      )}
    </motion.article>
  );
};

export default ProjectCard;