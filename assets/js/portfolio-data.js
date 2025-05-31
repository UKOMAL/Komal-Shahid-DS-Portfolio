/**
 * Portfolio Data
 * Data exported from Builder.io for Komal Shahid's portfolio
 */

// Portfolio data
const portfolioData = {
  activeTab: "projects",
  hoverCard: null,
  featuredProjects: [
    {
      title: "Federated Healthcare AI",
      desc: "Privacy-preserving machine learning for healthcare data across multiple institutions without sharing sensitive information.",
      image: "assets/images/federated_healthcare_graph.png",
      tech: ["Python", "Federated Learning", "Privacy", "Healthcare Analytics"],
      link: "https://github.com/UKOMAL/Federated-Healthcare-AI",
      demoLink: "assets/files/federated_healthcare_whitepaper.pdf",
      featured: true,
      bgColor: "linear-gradient(45deg, #ff96c7, #bfc5fe)"
    },
    {
      title: "Depression Detection System",
      desc: "AI-powered system for detecting indicators of depression from written text, designed to help identify at-risk individuals for early intervention.",
      image: "assets/images/depression_detection_graph.png",
      tech: ["NLP", "Python", "Machine Learning", "Mental Health"],
      link: "https://github.com/UKOMAL/Depression-Detection-System",
      demoLink: "assets/files/depression_detection_demo.html",
      featured: true,
      bgColor: "linear-gradient(45deg, #96d4ff, #fec5fe)"
    },
    {
      title: "Colorful Canvas: AI Art Studio",
      desc: "AI-powered toolkit for creating 3D visual illusions and effects from 2D images using depth mapping and neural networks.",
      image: "assets/images/projects/colorful-canvas.jpg",
      tech: ["Computer Vision", "Python", "PyTorch", "3D Effects"],
      link: "https://github.com/UKOMAL/Colorful-Canvas",
      demoLink: "assets/files/colorful_canvas_demo.html",
      featured: true,
      bgColor: "linear-gradient(45deg, #96a8ff, #c5feff)"
    },
    {
      title: "Network Visualization",
      desc: "Interactive network visualization tool that analyzes complex relationships in large datasets with graph theory algorithms.",
      image: "assets/images/network_visualization_improved.png",
      tech: ["Network Analysis", "D3.js", "Python", "Graph Theory"],
      link: "https://github.com/UKOMAL/Network-Visualization",
      demoLink: "assets/files/network_viz_demo.html",
      featured: true,
      bgColor: "linear-gradient(45deg, #96a8ff, #ffc5fe)"
    }
  ],
  projects: [
    {
      title: "Model Convergence Analysis",
      desc: "In-depth analysis of machine learning model convergence patterns with automated reporting of convergence metrics.",
      image: "assets/images/model_convergence.png",
      tech: ["Data Analysis", "ML Optimization", "Python", "TensorFlow"],
      link: "https://github.com/UKOMAL/Model-Convergence-Analysis",
      demoLink: "assets/files/convergence_analysis_report.pdf",
      bgColor: "linear-gradient(45deg, #a896ff, #c5feff)"
    },
    {
      title: "Performance Analysis",
      desc: "Comprehensive benchmark tool that analyzes machine learning model performance across different institutions and datasets.",
      image: "assets/images/institution_performance.png",
      tech: ["Performance Analysis", "Python", "Data Visualization", "Scikit-learn"],
      link: "https://github.com/UKOMAL/Performance-Analysis",
      demoLink: "assets/files/performance_metrics.html",
      bgColor: "linear-gradient(45deg, #96ffcb, #fefec5)"
    },
    {
      title: "Communication Efficiency",
      desc: "Framework for optimizing communication protocols in distributed systems to reduce bandwidth usage by up to 60%.",
      image: "assets/images/communication_efficiency.png",
      tech: ["Distributed Systems", "ML", "Python", "Network Optimization"],
      link: "https://github.com/UKOMAL/Communication-Efficiency",
      demoLink: "assets/files/communication_protocols.pdf",
      bgColor: "linear-gradient(45deg, #ff9696, #c5e8fe)"
    },
    {
      title: "Privacy Budget Analysis",
      desc: "Statistical framework for analyzing and optimizing privacy budgets in differential privacy implementations with visual reporting.",
      image: "assets/images/privacy_budget_tradeoff.png",
      tech: ["Privacy", "Differential Privacy", "Python", "Statistical Analysis"],
      link: "https://github.com/UKOMAL/Privacy-Budget-Analysis",
      demoLink: "assets/files/privacy_budget_report.html",
      bgColor: "linear-gradient(45deg, #96ffd0, #c5fedb)"
    }
  ],
  skills: [
    {
      category: "Programming",
      items: ["Python", "R", "SQL", "Excel"],
    },
    {
      category: "Machine Learning",
      items: [
        "Scikit-learn",
        "Statistical Modeling",
        "Time Series Analysis",
        "Classification",
      ],
    },
    {
      category: "Data Analysis",
      items: ["Pandas", "NumPy", "Data Visualization", "EDA"],
    },
    {
      category: "Tools",
      items: ["Jupyter", "Git", "Tableau", "Power BI"],
    },
  ],
  profileImages: [
    "https://github.com/UKOMAL.png",
    "assets/images/profile.jpg",
    "assets/images/performance_heatmap.png",
    "assets/images/model_comparison.png",
    "assets/images/convergence_final.png",
    "assets/images/client_contribution.png"
  ]
};

/**
 * Portfolio project data
 * Contains all the information about projects for displaying in modals
 */
const portfolioProjects = [
  {
    id: 'federated-healthcare-ai',
    title: 'Federated Healthcare AI',
    description: 'Privacy-preserving machine learning system that enables multiple healthcare institutions to collaboratively train AI models without sharing sensitive patient data. This federated learning approach ensures patient privacy while allowing institutions to benefit from larger, more diverse training datasets.',
    image: 'assets/images/projects/federated-learning.jpg',
    category: 'AI & Machine Learning',
    features: [
      'Decentralized model training across multiple institutions',
      'Differential privacy guarantees to prevent re-identification',
      'Secure aggregation protocol for model updates',
      'Adaptive learning rates based on client data quality',
      'Cross-silo federation with heterogeneous data structures',
      'Real-time performance monitoring dashboard'
    ],
    technologies: [
      'PyTorch',
      'TensorFlow Federated',
      'NVIDIA FLARE',
      'PySyft',
      'OpenMined',
      'Docker',
      'Kubernetes',
      'Python',
      'Flask'
    ],
    challenges: 'Addressing the statistical heterogeneity between institutions was a significant challenge. I developed a novel approach using personalized federated learning, where base models are shared but final layers are customized for each institution. This improved model performance by 23% compared to standard federated averaging techniques while maintaining privacy guarantees.',
    github: 'https://github.com/ukomal'
  },
  {
    id: 'depression-detection',
    title: 'Depression Detection System',
    description: 'An AI-powered system for detecting indicators of depression from written text, designed to help mental health professionals identify at-risk individuals for early intervention. The system analyzes linguistic patterns, sentiment, and content to provide insights about potential depression indicators.',
    image: 'assets/images/projects/depression-detection.jpg',
    category: 'Natural Language Processing',
    features: [
      'Advanced NLP for linguistic pattern recognition',
      'Multi-modal sentiment analysis (text, emojis, punctuation)',
      'Temporal analysis to detect changes over time',
      'Explainable AI features to highlight concerning content',
      'Privacy-focused design with user consent controls',
      'Integration with telehealth platforms'
    ],
    technologies: [
      'BERT',
      'Hugging Face',
      'spaCy',
      'PyTorch',
      'React',
      'Node.js',
      'MongoDB',
      'Docker'
    ],
    challenges: 'Balancing accuracy with ethical considerations was the primary challenge. I created a multi-stage classification system with human-in-the-loop validation to minimize false positives while maintaining high sensitivity. The final system achieved 89% accuracy on clinical validation sets while incorporating explicit ethical guidelines and privacy protections into the model architecture.',
    github: 'https://github.com/ukomal',
    demo: '#'
  },
  {
    id: 'network-visualization',
    title: 'Network Visualization Tool',
    description: 'An interactive network visualization tool that helps researchers and analysts understand complex relationships within large datasets. Built with D3.js and Python, this tool provides powerful insights through intuitive visualizations of network graphs, community clusters, and influence patterns.',
    image: 'assets/images/projects/network-viz.jpg',
    category: 'Data Visualization',
    features: [
      'Interactive force-directed graph layout',
      'Community detection algorithms',
      'Centrality measures and node importance',
      'Temporal network evolution visualization',
      'Advanced filtering and search capabilities',
      'Export to various formats (SVG, PNG, JSON)'
    ],
    technologies: [
      'D3.js',
      'Python',
      'NetworkX',
      'React',
      'Flask',
      'Neo4j',
      'WebGL'
    ],
    challenges: 'Rendering large-scale networks with thousands of nodes while maintaining performance was the main challenge. I implemented a multi-level architecture with WebGL for rendering and a custom quadtree-based spatial indexing system to optimize interaction. This approach enabled smooth exploration of networks with up to 100,000 nodes on standard hardware.',
    github: 'https://github.com/ukomal',
    demo: '#'
  },
  {
    id: 'healthcare-data-analysis',
    title: 'Healthcare Data Analysis',
    description: 'A comprehensive analysis of healthcare datasets to reveal insights for improved patient care and operational efficiency. This project focused on integrating multiple data sources to discover patterns in patient outcomes, treatment effectiveness, and hospital resource utilization.',
    image: 'assets/images/projects/healthcare-data.jpg',
    category: 'Data Analysis',
    features: [
      'Predictive modeling for patient readmission risk',
      'Treatment outcome analysis across demographics',
      'Resource utilization optimization',
      'Anomaly detection for billing errors',
      'Interactive dashboards for clinical decision support',
      'HIPAA-compliant data processing pipeline'
    ],
    technologies: [
      'Python',
      'Pandas',
      'scikit-learn',
      'R',
      'Tableau',
      'SQL',
      'FHIR',
      'Snowflake'
    ],
    challenges: 'Working with fragmented, inconsistent healthcare data was the greatest challenge. I developed a custom ETL pipeline with advanced entity resolution techniques to reconcile patient records across different systems. This approach improved data quality by 67% and enabled the discovery of previously hidden patterns in patient treatment journeys.',
    github: 'https://github.com/ukomal',
    demo: '#'
  },
  {
    id: 'predictive-maintenance',
    title: 'Predictive Maintenance AI',
    description: 'An AI system that predicts equipment failures before they happen using sensor data and machine learning. This proactive approach helps manufacturing and industrial clients reduce downtime, optimize maintenance schedules, and extend equipment lifespan.',
    image: 'assets/images/projects/predictive-maintenance.jpg',
    category: 'AI & Machine Learning',
    features: [
      'Real-time sensor data analysis',
      'Anomaly detection for early warning signs',
      'Remaining useful life (RUL) prediction',
      'Maintenance optimization scheduling',
      'Integration with IoT platforms and SCADA systems',
      'Digital twin modeling for equipment simulation'
    ],
    technologies: [
      'Python',
      'TensorFlow',
      'LSTM Networks',
      'Azure IoT',
      'Apache Kafka',
      'TimeScaleDB',
      'Docker',
      'Kubernetes'
    ],
    challenges: 'The main challenge was developing models that could generalize across different types of equipment with limited failure examples. I implemented a transfer learning approach with domain adaptation that allowed models trained on common failures to be fine-tuned for specific equipment with minimal additional data. This reduced the required training data by 80% while maintaining 93% prediction accuracy.',
    github: 'https://github.com/ukomal'
  },
  {
    id: 'interactive-dashboard',
    title: 'Interactive Data Dashboard',
    description: 'A dynamic dashboard for visualizing complex datasets with interactive filtering and exploration tools. This customizable visualization platform enables users to discover patterns, track KPIs, and share insights through an intuitive interface.',
    image: 'assets/images/projects/data-viz-dashboard.jpg',
    category: 'Visualization',
    features: [
      'Real-time data visualization with interactive controls',
      'Custom chart creation and layout options',
      'Advanced filtering and drill-down capabilities',
      'Shareable dashboards with user permissions',
      'Automated reporting and alerts',
      'Integration with multiple data sources'
    ],
    technologies: [
      'D3.js',
      'React',
      'Node.js',
      'GraphQL',
      'PostgreSQL',
      'Docker',
      'AWS',
      'Recharts'
    ],
    challenges: 'Creating a dashboard that remained responsive with large datasets was the primary challenge. I implemented a combination of server-side aggregation, WebSocket streaming for live updates, and client-side data virtualization. This architecture supports interactive exploration of datasets with millions of records while maintaining sub-second response times.',
    github: 'https://github.com/ukomal',
    demo: '#'
  },
  {
    id: 'colorful-canvas',
    title: 'Colorful Canvas: AI Art Studio',
    description: 'An AI-powered toolkit for creating stunning 3D visual illusions and effects from 2D images. This project uses depth estimation, neural networks, and advanced image processing to transform ordinary photos into immersive 3D experiences with various visual effects.',
    image: 'assets/images/projects/colorful-canvas.jpg',
    category: 'Computer Vision & Creative AI',
    features: [
      'Shadow Box Effect: Creates realistic display case illusions with depth-based 3D enhancement',
      'Screen Pop Effect: Makes objects appear to come out of the screen with chromatic aberration',
      'Depth mapping using state-of-the-art neural networks',
      'Glass reflection simulation and lighting effects',
      'Automatic 3D transformation with minimal user input',
      'Comprehensive visualization of the transformation process'
    ],
    technologies: [
      'Python',
      'PyTorch',
      'OpenCV',
      'Transformers',
      'Depth Estimation Models',
      'NumPy',
      'Matplotlib',
      'PIL'
    ],
    challenges: 'The main challenge was creating realistic 3D illusions from single 2D images without stereo pairs or specialized hardware. I developed a depth-aware processing pipeline that combines neural network-based depth estimation with custom post-processing techniques to create convincing 3D effects. The final system can transform any photo into various 3D visual illusions with minimal artifacts while maintaining the original image quality.',
    github: 'https://github.com/ukomal',
    demo: '#'
  }
];

// Portfolio Project Data
window.portfolioData = {
  // Featured Project
  "network-visualization": {
    title: "Network Visualization Tool",
    category: "Data Visualization",
    image: "assets/images/projects/network-viz.jpg",
    description: "An interactive network visualization tool that helps analyze complex relationships in large datasets. Built with D3.js and Python, this tool provides powerful insights through intuitive visualizations. The system uses force-directed graph algorithms to automatically arrange nodes based on their relationships, with interactive capabilities that allow users to explore connections and filter data in real-time.",
    technologies: [
      "D3.js", 
      "JavaScript", 
      "Python", 
      "NetworkX", 
      "Flask API"
    ],
    features: [
      "Interactive 3D graph visualization",
      "Real-time data filtering and search",
      "Community detection algorithms",
      "Centrality measurements",
      "Custom node styling and grouping"
    ],
    github: "https://github.com/ukomal/network-viz",
    demo: "#"
  },
  
  // Project 1
  "federated-healthcare-ai": {
    title: "Federated Healthcare AI",
    category: "AI & Machine Learning",
    image: "assets/images/projects/federated-learning.jpg",
    description: "A privacy-preserving machine learning system that enables healthcare institutions to collaboratively train AI models without sharing sensitive patient data. This federated learning approach maintains data privacy while achieving model performance comparable to centralized approaches.",
    technologies: [
      "PyTorch", 
      "TensorFlow Federated", 
      "Python", 
      "Docker", 
      "Kubernetes"
    ],
    features: [
      "Privacy-preserving federated learning",
      "Secure aggregation protocols",
      "Differential privacy implementation",
      "Cross-silo architecture for multiple institutions",
      "Model performance monitoring dashboard"
    ],
    github: "https://github.com/ukomal/federated-healthcare",
    demo: null
  },
  
  // Project 2
  "depression-detection": {
    title: "Depression Detection System",
    category: "Natural Language Processing",
    image: "assets/images/projects/depression-detection.jpg",
    description: "An AI-powered system that analyzes text data to identify potential indicators of depression. The model was trained on a diverse dataset of social media posts, clinical notes, and self-reported surveys to identify linguistic patterns associated with depression.",
    technologies: [
      "BERT", 
      "Hugging Face Transformers", 
      "Python", 
      "Flask", 
      "React"
    ],
    features: [
      "Multi-modal depression detection",
      "Privacy-focused analysis pipeline",
      "Confidence scoring with explanation",
      "Integration with clinical assessment tools",
      "API for third-party applications"
    ],
    github: "https://github.com/ukomal/depression-detection",
    demo: "#"
  },
  
  // Project 4
  "healthcare-data-analysis": {
    title: "Healthcare Data Analysis",
    category: "Data Analysis",
    image: "assets/images/projects/healthcare-data.jpg",
    description: "A comprehensive analysis of healthcare datasets to reveal insights for improved patient care and operational efficiency. This project involved cleaning and integrating data from multiple sources, applying statistical methods, and creating interactive dashboards for healthcare administrators.",
    technologies: [
      "Python", 
      "Pandas", 
      "NumPy", 
      "Scikit-learn", 
      "Tableau"
    ],
    features: [
      "Predictive analytics for patient readmission",
      "Cost optimization analysis",
      "Treatment effectiveness comparison",
      "Resource allocation optimization",
      "Interactive visualizations for stakeholders"
    ],
    github: "https://github.com/ukomal/healthcare-data-analysis",
    demo: null
  },
  
  // Project 5
  "predictive-maintenance": {
    title: "Predictive Maintenance AI",
    category: "AI & Machine Learning",
    image: "assets/images/projects/predictive-maintenance.jpg",
    description: "An AI system that predicts equipment failures before they happen using sensor data and machine learning. Deployed in manufacturing settings, this solution has reduced downtime by detecting potential issues early and enabling proactive maintenance scheduling.",
    technologies: [
      "TensorFlow", 
      "Keras", 
      "Python", 
      "Time Series Analysis", 
      "IoT Sensors"
    ],
    features: [
      "Real-time anomaly detection",
      "Remaining useful life prediction",
      "Automated maintenance scheduling",
      "Sensor data visualization dashboard",
      "Integration with maintenance management systems"
    ],
    github: "https://github.com/ukomal/predictive-maintenance",
    demo: "#"
  },
  
  // Project 6
  "interactive-dashboard": {
    title: "Interactive Data Dashboard",
    category: "Visualization",
    image: "assets/images/projects/data-viz-dashboard.jpg",
    description: "A dynamic dashboard for visualizing complex datasets with interactive filtering and exploration tools. The dashboard provides multiple visualization types and customization options, allowing users to gain insights from data through an intuitive interface.",
    technologies: [
      "D3.js", 
      "React", 
      "Node.js", 
      "MongoDB", 
      "Express"
    ],
    features: [
      "Real-time data updates",
      "Custom visualization creation",
      "Advanced filtering and search",
      "Data export functionality",
      "Responsive design for all devices"
    ],
    github: "https://github.com/ukomal/interactive-dashboard",
    demo: "#"
  },
  
  // Project 7
  "colorful-canvas": {
    title: "Colorful Canvas: AI Art Studio",
    category: "Computer Vision & Creative AI",
    image: "assets/images/projects/colorful-canvas.jpg",
    description: "An AI-powered toolkit for creating stunning 3D visual illusions and effects from 2D images. This project uses depth estimation, neural networks, and advanced image processing to transform ordinary photos into immersive 3D experiences with various visual effects.",
    technologies: [
      "Python",
      "PyTorch",
      "OpenCV",
      "Transformers",
      "Depth Estimation Models",
      "NumPy",
      "Matplotlib",
      "PIL"
    ],
    features: [
      "Shadow Box Effect: Creates realistic display case illusions with depth-based 3D enhancement",
      "Screen Pop Effect: Makes objects appear to come out of the screen with chromatic aberration",
      "Depth mapping using state-of-the-art neural networks",
      "Glass reflection simulation and lighting effects",
      "Automatic 3D transformation with minimal user input",
      "Comprehensive visualization of the transformation process"
    ],
    github: "https://github.com/ukomal/colorful-canvas",
    demo: "#"
  }
}; 