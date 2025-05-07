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