# Analytics & Conversion Tracking Plan
## AI/ML Engineer Portfolio - Data-Driven Optimization Strategy

### Overview
This comprehensive analytics plan focuses on measuring recruiter engagement, technical competency demonstration, and conversion optimization for AI/ML engineering roles.

---

## Primary Conversion Goals & KPIs

### 1. Primary Conversions (High Intent)
| Goal | Definition | Success Rate Target | Tracking Method |
|------|------------|-------------------|-----------------|
| **View Case Study** | Click from hero/project card to detailed case study | 15-20% of visitors | `gtag('event', 'view_case_study')` |
| **Run Demo** | Interact with live project demonstration | 8-12% of visitors | `gtag('event', 'run_demo')` |
| **Request Interview** | Contact form submission or calendar booking | 2-5% of visitors | `gtag('event', 'schedule_interview')` |

### 2. Secondary Conversions (Medium Intent)
| Goal | Definition | Success Rate Target | Tracking Method |
|------|------------|-------------------|-----------------|
| **Download Resume** | PDF resume download | 10-15% of visitors | `gtag('event', 'download_resume')` |
| **View GitHub** | Click to GitHub repository | 20-25% of visitors | `gtag('event', 'external_link', {link_type: 'github'})` |
| **LinkedIn Profile** | Visit LinkedIn profile | 5-10% of visitors | `gtag('event', 'external_link', {link_type: 'linkedin'})` |

### 3. Engagement Metrics (Indicative)
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Time on Page** | >45 seconds average | Google Analytics |
| **Hero Scroll-Through** | >70% users scroll past hero | Scroll depth tracking |
| **Case Study Completion** | >60% read to end | Scroll depth + time tracking |
| **Project Card Hover** | >40% hover engagement | Custom event tracking |

---

## Analytics Implementation

### Google Analytics 4 Configuration

#### Custom Events Setup
```javascript
// Enhanced measurement events for portfolio
gtag('config', 'GA_MEASUREMENT_ID', {
  // Portfolio-specific configuration
  custom_map: {
    'custom_parameter_1': 'project_type',
    'custom_parameter_2': 'visitor_type', 
    'custom_parameter_3': 'technical_level'
  },
  
  // Conversion tracking
  send_page_view: true,
  enhanced_measurement: {
    scrolls: true,
    outbound_clicks: true,
    site_search: true,
    video_engagement: true,
    file_downloads: true
  }
});

// Custom dimensions for AI/ML portfolio
gtag('event', 'page_view', {
  'project_type': 'healthcare_ai', // healthcare_ai, fintech_ai, computer_vision
  'visitor_type': 'recruiter', // recruiter, technical_peer, student
  'page_depth': 1, // 1=homepage, 2=project_index, 3=case_study
  'user_intent': 'evaluation' // evaluation, learning, collaboration
});
```

#### Conversion Events Tracking
```javascript
// Hero Section Tracking
const trackHeroCTA = (ctaType, projectName) => {
  gtag('event', 'hero_cta_click', {
    'event_category': 'Hero Engagement',
    'event_label': ctaType, // 'view_case_study' or 'run_demo'
    'project_name': projectName,
    'cta_position': 'hero_primary',
    'value': ctaType === 'view_case_study' ? 10 : 8 // Weighted importance
  });
};

// Project Card Interaction
const trackProjectCard = (action, projectName, cardPosition) => {
  gtag('event', 'project_interaction', {
    'event_category': 'Project Showcase',
    'event_label': action, // hover, click, demo_click
    'project_name': projectName,
    'card_position': cardPosition, // 1, 2, 3
    'interaction_type': action
  });
};

// Technical Demo Engagement  
const trackDemoUsage = (demoType, engagementLevel, timeSpent) => {
  gtag('event', 'demo_engagement', {
    'event_category': 'Technical Demonstration',
    'event_label': demoType, // depression_ai, fraud_detection, etc.
    'engagement_level': engagementLevel, // low, medium, high
    'time_spent': Math.round(timeSpent), // seconds
    'demo_completion': engagementLevel === 'high'
  });
};
```

### Heat Mapping & User Behavior

#### Microsoft Clarity Implementation
```html
<!-- Microsoft Clarity for heat maps and session recordings -->
<script type="text/javascript">
(function(c,l,a,r,i,t,y){
    c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
    t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
    y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
})(window, document, "clarity", "script", "CLARITY_PROJECT_ID");

// Custom clarity tags for portfolio analysis
clarity("set", "page_type", "portfolio_home");
clarity("set", "visitor_role", "recruiter"); // Based on user agent/behavior
clarity("set", "technical_content", "ai_ml_engineering");
</script>
```

#### Custom Scroll Depth Tracking
```javascript
// Advanced scroll tracking for case studies
let scrollMilestones = [25, 50, 75, 90, 100];
let scrollTracked = [];

const trackScrollDepth = () => {
  const scrollPercent = Math.round(
    (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
  );
  
  scrollMilestones.forEach(milestone => {
    if (scrollPercent >= milestone && !scrollTracked.includes(milestone)) {
      scrollTracked.push(milestone);
      
      gtag('event', 'scroll_depth', {
        'event_category': 'User Engagement',
        'event_label': `${milestone}%`,
        'page_type': document.title.includes('Case Study') ? 'case_study' : 'general',
        'content_depth': milestone
      });
    }
  });
};

window.addEventListener('scroll', throttle(trackScrollDepth, 500));
```

---

## A/B Testing Strategy

### Test 1: Hero CTA Optimization
**Hypothesis**: Technical demos convert better than case study views for recruiting

**Variants**:
- **A (Control)**: "View Case Study" primary, "Run Demo" secondary
- **B (Test)**: "Run Demo" primary, "View Case Study" secondary

**Measurement**:
```javascript
// A/B test tracking
const abTestVariant = Math.random() < 0.5 ? 'A' : 'B';

gtag('event', 'ab_test_view', {
  'test_name': 'hero_cta_order',
  'variant': abTestVariant,
  'page': 'homepage'
});

// Track conversion by variant
gtag('event', 'conversion', {
  'test_variant': abTestVariant,
  'conversion_type': 'primary_cta',
  'value': 1
});
```

**Success Criteria**: 
- Primary metric: Click-through rate to demos/case studies
- Secondary: Time spent on target pages
- Duration: 2 weeks, minimum 1000 visitors

### Test 2: Metric-First vs Story-First Positioning
**Hypothesis**: Leading with quantified results appeals more to technical recruiters

**Variants**:
- **A (Metric-First)**: "Model AUC 0.92" → Tagline → CTAs
- **B (Story-First)**: Tagline → "Model AUC 0.92" → CTAs

**Measurement**: Hero engagement rate, case study completion

### Test 3: Technical Depth vs Business Impact
**Hypothesis**: Balancing technical depth with business value increases recruiter interest

**Variants**:
- **A (Technical)**: Emphasize algorithms, architectures, performance metrics
- **B (Business)**: Emphasize ROI, problem-solving, real-world impact

**Pages**: Project case study introductions
**Success Criteria**: Interview requests, time on case study pages

---

## Conversion Funnel Analysis

### Funnel Stages & Drop-off Points
```
1. AWARENESS (100%)
   ↓ Landing page visit
   
2. INTEREST (70-80%)
   ↓ Scroll past hero
   
3. CONSIDERATION (40-50%)  
   ↓ View project details
   
4. EVALUATION (20-30%)
   ↓ Read full case study
   
5. INTENT (10-15%)
   ↓ Interact with demo
   
6. ACTION (2-5%)
   ↓ Contact/interview request
```

### Funnel Optimization Tracking
```javascript
// Stage progression tracking
const trackFunnelStage = (stage, additionalData = {}) => {
  gtag('event', 'funnel_progression', {
    'event_category': 'Conversion Funnel',
    'funnel_stage': stage, // awareness, interest, consideration, etc.
    'stage_order': getFunnelStageOrder(stage),
    'session_depth': getSessionPageViews(),
    ...additionalData
  });
};

// Exit intent detection
document.addEventListener('mouseout', (e) => {
  if (!e.toElement && !e.relatedTarget) {
    gtag('event', 'exit_intent', {
      'page_section': getCurrentSection(),
      'time_on_page': getTimeOnPage(),
      'scroll_depth': getScrollDepth()
    });
  }
});
```

---

## Attribution & Source Analysis

### UTM Parameter Strategy
```
Campaign Sources:
- LinkedIn posts: utm_source=linkedin&utm_medium=social&utm_campaign=portfolio_launch
- GitHub profile: utm_source=github&utm_medium=referral&utm_campaign=github_profile
- Resume applications: utm_source=resume&utm_medium=application&utm_campaign=job_applications
- Direct outreach: utm_source=outreach&utm_medium=email&utm_campaign=recruiter_direct

Content Attribution:
- Project focus: utm_content=depression_detection | federated_learning | fraud_detection
- Role targeting: utm_content=ai_engineer | ml_engineer | data_scientist
```

### Channel Performance Tracking
```javascript
// Enhanced source attribution
const trackTrafficSource = () => {
  const urlParams = new URLSearchParams(window.location.search);
  const referrer = document.referrer;
  
  let sourceData = {
    'traffic_source': 'direct',
    'source_category': 'direct',
    'campaign': urlParams.get('utm_campaign') || 'none'
  };
  
  // Classify traffic sources
  if (referrer.includes('linkedin.com')) {
    sourceData = { ...sourceData, traffic_source: 'linkedin', source_category: 'social' };
  } else if (referrer.includes('github.com')) {
    sourceData = { ...sourceData, traffic_source: 'github', source_category: 'professional' };
  } else if (urlParams.get('utm_source')) {
    sourceData = { 
      ...sourceData, 
      traffic_source: urlParams.get('utm_source'),
      source_category: 'campaign'
    };
  }
  
  gtag('event', 'traffic_attribution', sourceData);
};
```

---

## Performance Monitoring

### Core Web Vitals for Recruitment
```javascript
// Performance impact on conversions
const trackPerformanceImpact = () => {
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (entry.entryType === 'largest-contentful-paint') {
        gtag('event', 'performance_metric', {
          'metric_name': 'LCP',
          'metric_value': Math.round(entry.startTime),
          'performance_impact': entry.startTime < 2500 ? 'good' : 'needs_improvement'
        });
      }
    }
  }).observe({ entryTypes: ['largest-contentful-paint'] });
};

// Hero load time critical for first impression
const trackHeroLoadTime = () => {
  window.addEventListener('load', () => {
    const heroLoadTime = performance.now();
    gtag('event', 'hero_load_complete', {
      'load_time': Math.round(heroLoadTime),
      'performance_grade': heroLoadTime < 3000 ? 'excellent' : 'needs_optimization'
    });
  });
};
```

---

## Reporting Dashboard Configuration

### Google Analytics 4 Custom Reports

#### 1. Recruiter Engagement Report
**Metrics**: Page views, session duration, conversion rate by traffic source
**Dimensions**: Source/Medium, Campaign, Page Title
**Filters**: Include only potential recruiter traffic (LinkedIn, job boards, direct)

#### 2. Technical Content Performance
**Metrics**: Time on page, scroll depth, demo engagement
**Dimensions**: Project type, technical complexity, content format
**Segments**: High-intent visitors (>3 page views)

#### 3. Conversion Attribution Analysis
**Metrics**: Goal completions by source
**Dimensions**: Source/Medium, Landing page, Conversion path
**Attribution Model**: Data-driven (40-day lookback)

### Real-Time Monitoring Alerts
```javascript
// Set up alerts for key metrics
const setupMonitoringAlerts = () => {
  // Low conversion rate alert
  if (dailyConversionRate < 0.02) {
    sendAlert('Low conversion rate detected', 'daily_performance');
  }
  
  // High bounce rate from key sources
  if (linkedinBounceRate > 0.7) {
    sendAlert('High LinkedIn bounce rate', 'traffic_quality');
  }
  
  // Demo engagement drop
  if (demoCompletionRate < 0.6) {
    sendAlert('Demo engagement declining', 'technical_content');
  }
};
```

---

## Privacy & Compliance

### GDPR/CCPA Compliance
```javascript
// Privacy-compliant analytics
const initializeAnalytics = () => {
  // Check consent status
  const hasConsent = getCookieConsent();
  
  if (hasConsent) {
    gtag('config', 'GA_MEASUREMENT_ID', {
      'anonymize_ip': true,
      'allow_google_signals': false, // Disable advertising features
      'allow_ad_personalization_signals': false
    });
  } else {
    // Basic analytics without personal data
    gtag('config', 'GA_MEASUREMENT_ID', {
      'anonymize_ip': true,
      'storage': 'none',
      'functionality_storage': 'denied',
      'security_storage': 'denied'
    });
  }
};
```

---

## Success Measurement Timeline

### Week 1-2: Baseline Establishment
- Deploy tracking infrastructure
- Collect baseline metrics
- Validate data accuracy

### Week 3-4: Initial Optimization
- Implement first A/B tests
- Monitor conversion funnel
- Adjust based on early data

### Week 5-8: Performance Tuning
- Optimize high-traffic pages
- Refine conversion paths
- Test messaging variants

### Month 2+: Scale & Iterate
- Expand successful tests
- Implement advanced attribution
- Seasonal adjustment strategies

### Key Success Indicators (90 days)
- **Conversion Rate**: >3% visitor-to-contact rate
- **Engagement Quality**: >60 seconds average session
- **Technical Credibility**: >40% case study completion rate
- **Source Diversity**: <50% dependence on any single channel

This analytics plan provides comprehensive tracking of recruiter behavior while demonstrating data-driven optimization skills essential for senior AI/ML engineering roles.