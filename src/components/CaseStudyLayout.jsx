import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import CTAButton, { ViewCaseStudyButton, RunDemoButton, DownloadButton } from './CTAButton';
import MetricPill, { AUCPill, F1Pill, AccuracyPill, R2Pill } from './MetricPill';
import Badge from './Badge';

// Defined outside the component: the list is static and never changes,
// so it must not be recreated on every render (which would cause the
// IntersectionObserver useEffect to re-run and reconnect on every render).
const CASE_STUDY_SECTIONS = [
  { id: 'problem', label: 'Problem', icon: '🎯' },
  { id: 'dataset', label: 'Dataset', icon: '📊' },
  { id: 'approach', label: 'Approach', icon: '🔬' },
  { id: 'results', label: 'Results', icon: '📈' },
  { id: 'ethics', label: 'Ethics', icon: '⚖️' },
  { id: 'reproducibility', label: 'Reproducible', icon: '🔄' },
  { id: 'lessons', label: 'Lessons', icon: '💡' }
];

const CaseStudyLayout = ({
  title,
  subtitle,
  heroImage,
  problem,
  dataset,
  approach,
  results,
  fairness,
  interpretability,
  reproducibility,
  demo,
  lessons,
  codeLinks,
  metrics = [],
  tags = [],
  timeline,
  teamSize,
  className = ""
}) => {
  const [activeSection, setActiveSection] = useState('problem');
  const [isReproducibilityExpanded, setIsReproducibilityExpanded] = useState(false);

  useEffect(() => {
    // Intersection Observer for active section highlighting
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.3 }
    );

    CASE_STUDY_SECTIONS.forEach(section => {
      const element = document.getElementById(section.id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, []);

  const sectionVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.6, ease: "easeOut" }
    }
  };

  return (
    <article className={`case-study-layout max-w-4xl mx-auto ${className}`}>
      {/* Hero Section */}
      <motion.header 
        className="text-center mb-12"
        initial="hidden"
        animate="visible"
        variants={sectionVariants}
      >
        {heroImage && (
          <div className="relative mb-8 rounded-xl overflow-hidden shadow-2xl">
            <img
              src={heroImage}
              alt={`${title} project overview`}
              className="w-full h-64 md:h-96 object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
          </div>
        )}
        
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-4">
          {title}
        </h1>
        
        {subtitle && (
          <p className="text-xl text-slate-600 mb-6 max-w-2xl mx-auto">
            {subtitle}
          </p>
        )}

        {/* Key Metrics */}
        {metrics.length > 0 && (
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            {metrics.map((metric, index) => (
              <MetricPill
                key={index}
                metric={metric.label}
                value={metric.value}
                tooltip={metric.tooltip}
                variant="success"
              />
            ))}
          </div>
        )}

        {/* Technology Tags */}
        {tags.length > 0 && (
          <div className="flex flex-wrap justify-center gap-2 mb-8">
            {tags.map((tag, index) => (
              <Badge key={index} text={tag} variant="tech" />
            ))}
          </div>
        )}

        {/* Quick Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          {demo?.link && (
            <RunDemoButton href={demo.link} />
          )}
          {codeLinks?.github && (
            <CTAButton
              text="View Code"
              href={codeLinks.github}
              variant="secondary"
              analytics="view_code"
              icon={
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              }
            />
          )}
          {reproducibility?.notebook && (
            <DownloadButton href={reproducibility.notebook} />
          )}
        </div>
      </motion.header>

      {/* Sticky Navigation */}
      <nav className="sticky top-20 z-30 bg-white/95 backdrop-blur-sm border border-slate-200 rounded-lg p-4 mb-8 shadow-sm">
        <div className="flex flex-wrap justify-center gap-2">
          {CASE_STUDY_SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className={`
                flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium
                transition-colors duration-200
                ${activeSection === section.id 
                  ? 'bg-blue-100 text-blue-800' 
                  : 'text-slate-600 hover:bg-slate-100'
                }
              `}
            >
              <span>{section.icon}</span>
              <span className="hidden sm:inline">{section.label}</span>
            </a>
          ))}
        </div>
      </nav>

      {/* Content Sections */}
      <div className="space-y-16">
        {/* Problem Section */}
        <motion.section 
          id="problem"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            🎯 Problem Statement
          </h2>
          <div className="bg-blue-50 border-l-4 border-blue-400 p-6 rounded-r-lg">
            {problem}
          </div>
        </motion.section>

        {/* Dataset Section */}
        <motion.section 
          id="dataset"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            📊 Dataset & Data Engineering
          </h2>
          {dataset}
        </motion.section>

        {/* Approach Section */}
        <motion.section 
          id="approach"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            🔬 Methodology & Approach
          </h2>
          {approach}
        </motion.section>

        {/* Results Section */}
        <motion.section 
          id="results"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            📈 Results & Performance
          </h2>
          {results}
        </motion.section>

        {/* Ethics Section */}
        <motion.section 
          id="ethics"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            ⚖️ Fairness & Interpretability
          </h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">Fairness Analysis</h3>
              {fairness}
            </div>
            <div>
              <h3 className="text-xl font-semibold mb-4">Model Interpretability</h3>
              {interpretability}
            </div>
          </div>
        </motion.section>

        {/* Reproducibility Section - Collapsible */}
        <motion.section 
          id="reproducibility"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <div 
            className="cursor-pointer bg-slate-50 p-6 rounded-lg border border-slate-200 hover:bg-slate-100 transition-colors duration-200"
            onClick={() => setIsReproducibilityExpanded(!isReproducibilityExpanded)}
          >
            <h2 className="text-3xl font-bold text-slate-900 mb-2 flex items-center justify-between gap-3">
              <span className="flex items-center gap-3">
                🔄 Reproducible Artifacts
              </span>
              <motion.svg 
                className="w-6 h-6"
                animate={{ rotate: isReproducibilityExpanded ? 180 : 0 }}
                transition={{ duration: 0.2 }}
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </motion.svg>
            </h2>
            <p className="text-slate-600 mb-0">
              Click to view reproducibility checklist and artifacts
            </p>
          </div>

          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ 
              height: isReproducibilityExpanded ? 'auto' : 0,
              opacity: isReproducibilityExpanded ? 1 : 0
            }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="pt-6">
              {reproducibility}
            </div>
          </motion.div>
        </motion.section>

        {/* Lessons Learned */}
        <motion.section 
          id="lessons"
          variants={sectionVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="prose prose-lg max-w-none"
        >
          <h2 className="text-3xl font-bold text-slate-900 mb-6 flex items-center gap-3">
            💡 Key Insights & Lessons Learned
          </h2>
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-r-lg">
            {lessons}
          </div>
        </motion.section>
      </div>

      {/* Footer Actions */}
      <motion.footer 
        className="mt-16 pt-8 border-t border-slate-200 text-center space-y-6"
        variants={sectionVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true }}
      >
        <h3 className="text-2xl font-bold text-slate-900">Ready to Discuss This Project?</h3>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <CTAButton
            text="Schedule Technical Interview"
            href="/contact"
            variant="primary"
            size="large"
            analytics="case_study_footer_interview"
          />
          <CTAButton
            text="View All Projects"
            href="/projects"
            variant="secondary"
            size="large"
            analytics="case_study_footer_projects"
          />
        </div>

        {/* Project Metadata */}
        <div className="flex flex-wrap justify-center gap-6 text-sm text-slate-500 mt-8">
          {timeline && (
            <span className="flex items-center gap-1">
              📅 {timeline}
            </span>
          )}
          {teamSize && (
            <span className="flex items-center gap-1">
              👥 Team of {teamSize}
            </span>
          )}
          <span className="flex items-center gap-1">
            🔄 Last updated: 2 weeks ago
          </span>
        </div>
      </motion.footer>
    </article>
  );
};

export default CaseStudyLayout;