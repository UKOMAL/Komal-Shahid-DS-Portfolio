import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import CTAButton from './CTAButton';
import MetricPill from './MetricPill';

const Hero = ({ 
  roleLabel = "AI & ML Engineer",
  topMetric = { value: "0.92", label: "Model AUC", tooltip: "Average across 3 major projects" },
  primaryCTA = { text: "View Case Study", href: "/projects", analytics: "hero_primary_cta" },
  secondaryCTA = { text: "Run Demo", href: "/demo", analytics: "hero_secondary_cta" },
  tagline = "Building privacy-preserving AI that protects 800K+ transactions while advancing healthcare innovation.",
  backgroundImage = "/images/hero-background.jpg"
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [currentTaglineIndex, setCurrentTaglineIndex] = useState(0);

  // Tagline options for A/B testing
  const taglineOptions = [
    tagline, // Default from props
    "Delivering 92% AUC depression detection and 89% F1-score federated learning across distributed healthcare networks.",
    "Transforming complex healthcare and financial challenges into ethical AI solutions with measurable business impact."
  ];

  useEffect(() => {
    // Simulate loading and trigger animations
    const timer = setTimeout(() => setIsLoaded(true), 100);
    return () => clearTimeout(timer);
  }, []);

  const heroVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.6,
        ease: "easeOut",
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { duration: 0.4 }
    }
  };

  return (
    <section 
      className="hero relative min-h-[600px] md:min-h-[500px] flex items-center justify-center overflow-hidden bg-gradient-to-br from-blue-900 via-blue-800 to-slate-900"
      role="banner"
      aria-label="Hero section"
    >
      {/* Background Image with Overlay */}
      <div 
        className="absolute inset-0 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${backgroundImage})` }}
        aria-hidden="true"
      />
      <div className="absolute inset-0 bg-gradient-to-r from-blue-900/80 to-slate-900/80" />
      
      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 text-center max-w-4xl">
        <motion.div
          variants={heroVariants}
          initial="hidden"
          animate={isLoaded ? "visible" : "hidden"}
          className="space-y-6"
        >
          {/* Name */}
          <motion.h1 
            variants={itemVariants}
            className="text-4xl md:text-6xl font-bold text-white mb-2"
          >
            Komal Shahid
          </motion.h1>

          {/* Role Label */}
          <motion.h2 
            variants={itemVariants}
            className="text-xl md:text-2xl font-semibold text-cyan-300 mb-4"
          >
            {roleLabel}
          </motion.h2>

          {/* Top Metric */}
          <motion.div 
            variants={itemVariants}
            className="flex justify-center mb-6"
          >
            <MetricPill 
              metric={topMetric.label}
              value={topMetric.value}
              tooltip={topMetric.tooltip}
              variant="hero"
            />
          </motion.div>

          {/* Tagline */}
          <motion.p 
            variants={itemVariants}
            className="text-lg md:text-xl text-slate-200 max-w-3xl mx-auto leading-relaxed mb-8"
          >
            {taglineOptions[currentTaglineIndex]}
          </motion.p>

          {/* CTA Buttons */}
          <motion.div 
            variants={itemVariants}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
          >
            <CTAButton
              text={primaryCTA.text}
              href={primaryCTA.href}
              variant="primary"
              size="large" 
              analytics={primaryCTA.analytics}
              className="w-full sm:w-auto"
            />
            <CTAButton
              text={secondaryCTA.text}
              href={secondaryCTA.href}
              variant="secondary"
              size="large"
              analytics={secondaryCTA.analytics}
              className="w-full sm:w-auto"
            />
          </motion.div>

          {/* Trust Indicators */}
          <motion.div 
            variants={itemVariants}
            className="flex flex-wrap justify-center items-center gap-6 mt-8 text-slate-300"
          >
            <span className="flex items-center gap-2 text-sm">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
              </svg>
              MS Data Science, Bellevue University
            </span>
            <span className="flex items-center gap-2 text-sm">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 12a2 2 0 100-4 2 2 0 000 4z"/>
                <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd"/>
              </svg>
              Available Q2 2026
            </span>
          </motion.div>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div 
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5, duration: 0.5 }}
      >
        <div className="w-6 h-10 border-2 border-slate-300 rounded-full flex justify-center">
          <motion.div 
            className="w-1 h-3 bg-slate-300 rounded-full mt-2"
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
        </div>
      </motion.div>
    </section>
  );
};

export default Hero;