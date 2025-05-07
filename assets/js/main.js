/**
 * Main JavaScript file for Komal Shahid's portfolio
 * Handles tab navigation, project rendering, and animations
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize state
  let state = {
    activeTab: "featured",
    hoverCard: null
  };

  // DOM Elements
  const navTabs = document.querySelectorAll('.nav-tab');
  const contentSections = document.querySelectorAll('.content-section');
  const projectsContainer = document.getElementById('projects-container');
  const featuredProjectsContainer = document.getElementById('featured-projects-container');
  const skillsContainer = document.getElementById('skills-container');
  const cubeContainer = document.querySelector('.cube-container');
  const flowerCursor = document.querySelector('.flower-cursor');
  const parallaxTitle = document.querySelector('.parallax-title');

  // Initialize the page
  initializePage();

  /**
   * Initialize the page with content and event listeners
   */
  function initializePage() {
    initializeCube();
    renderFeaturedProjects();
    renderProjects();
    renderSkills();
    setupEventListeners();
    initializeScrollAnimations();
    initializeParallax();
    setupFlowerCursor();
    setupParallaxEffects();

    // Show the active tab on page load
    setActiveTab("featured");
  }

  /**
   * Initialize scroll animations
   */
  function initializeScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, {
      threshold: 0.1
    });

    // Observe all sections
    document.querySelectorAll('section').forEach(section => {
      observer.observe(section);
    });

    // Smooth scroll for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });
        }
      });
    });
  }

  /**
   * Initialize parallax effect
   */
  function initializeParallax() {
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const parallaxElements = document.querySelectorAll('.parallax');
      
      parallaxElements.forEach(element => {
        const speed = element.dataset.speed || 0.5;
        element.style.transform = `translateY(${scrolled * speed}px)`;
      });
    });
  }

  /**
   * Render featured projects to the featured section
   */
  function renderFeaturedProjects() {
    if (!featuredProjectsContainer) return;
    
    featuredProjectsContainer.innerHTML = '';
    
    // Render featured projects
    portfolioData.featuredProjects.forEach((project, index) => {
      const projectCard = document.createElement('div');
      projectCard.className = 'featured-project-card';
      
      projectCard.innerHTML = `
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: ${project.bgColor}; opacity: 0.12; animation: gradient 15s ease infinite;"></div>
        <div class="project-content">
          <div class="project-bg" style="background: ${project.bgColor}"></div>
          <div class="project-info">
            <h3>${project.title}</h3>
            <p>${project.desc}</p>
            <div class="project-tags">
              ${project.tech.map(tech => `<span class="project-tag">${tech}</span>`).join('')}
            </div>
            <div class="project-links">
              <a href="${project.link}" target="_blank" class="project-link">GitHub <i class="fas fa-external-link-alt"></i></a>
              ${project.demoLink ? `<a href="${project.demoLink}" target="_blank" class="project-link">Demo <i class="fas fa-play-circle"></i></a>` : ''}
            </div>
          </div>
          <div class="project-image-container">
            <img src="${project.image}" alt="${project.title}" class="project-image">
          </div>
        </div>
      `;

      // Add event listeners for hover effects
      projectCard.addEventListener('mouseenter', () => {
        state.hoverCard = index;
        projectCard.style.transform = 'translateY(-10px)';
        const bg = projectCard.querySelector('.project-bg');
        if (bg) bg.style.opacity = '0.5';
      });
      
      projectCard.addEventListener('mouseleave', () => {
        state.hoverCard = null;
        projectCard.style.transform = 'translateY(0)';
        const bg = projectCard.querySelector('.project-bg');
        if (bg) bg.style.opacity = '0.2';
      });

      // Remove click event on the entire card
      // Add individual click events to the links instead
      projectCard.removeEventListener('click', () => {});

      featuredProjectsContainer.appendChild(projectCard);
    });
  }

  /**
   * Render regular projects to the projects tab
   */
  function renderProjects() {
    if (!projectsContainer) return;
    
    projectsContainer.innerHTML = '';
    
    // Only regular projects for the projects tab
    // Don't duplicate the featured projects
    portfolioData.projects.forEach((project, index) => {
      const projectCard = document.createElement('div');
      projectCard.className = 'project-card';
      projectCard.style.animation = `fadeInProject 0.5s forwards ${index * 0.1}s`;
      
      projectCard.innerHTML = `
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: ${project.bgColor}; opacity: 0.12; animation: gradient 15s ease infinite;"></div>
        <div class="project-content">
          <div class="project-bg" style="background: ${project.bgColor}"></div>
          <div class="project-info">
            <h3>${project.title}</h3>
            <p>${project.desc}</p>
            <div class="project-tags">
              ${project.tech.map(tech => `<span class="project-tag">${tech}</span>`).join('')}
            </div>
            <div class="project-links">
              <a href="${project.link}" target="_blank" class="project-link">GitHub <i class="fas fa-external-link-alt"></i></a>
              ${project.demoLink ? `<a href="${project.demoLink}" target="_blank" class="project-link">Demo <i class="fas fa-play-circle"></i></a>` : ''}
            </div>
          </div>
          <div class="project-image-container">
            <img src="${project.image}" alt="${project.title}" class="project-image">
          </div>
        </div>
      `;

      // Add event listeners for hover effects
      projectCard.addEventListener('mouseenter', () => {
        state.hoverCard = index;
        projectCard.style.transform = 'translateY(-10px)';
        const bg = projectCard.querySelector('.project-bg');
        if (bg) bg.style.opacity = '0.5';
      });
      
      projectCard.addEventListener('mouseleave', () => {
        state.hoverCard = null;
        projectCard.style.transform = 'translateY(0)';
        const bg = projectCard.querySelector('.project-bg');
        if (bg) bg.style.opacity = '0.2';
      });

      // Add click event to open the dedicated info/demo page
      projectCard.addEventListener('click', () => {
        // Example: /projects/project2-federated-healthcare-ai/demo/index.html
        let projectSlug = '';
        if (project.title.toLowerCase().includes('federated')) {
          projectSlug = 'project2-federated-healthcare-ai';
        } else if (project.title.toLowerCase().includes('depression')) {
          projectSlug = 'project1-depression-detection';
        } // Add more mappings as needed
        if (projectSlug) {
          window.open(`/projects/${projectSlug}/demo/index.html`, '_blank');
        }
      });

      projectsContainer.appendChild(projectCard);
    });
  }

  /**
   * Render skills to the DOM
   */
  function renderSkills() {
    if (!skillsContainer) return;
    
    skillsContainer.innerHTML = '';

    portfolioData.skills.forEach((skillGroup, index) => {
      const skillCard = document.createElement('div');
      skillCard.className = 'skill-card';
      skillCard.style.animation = `fadeInProject 0.5s forwards ${index * 0.1}s`;
      
      skillCard.innerHTML = `
        <h4>${skillGroup.category}</h4>
        <div class="skill-items">
          ${skillGroup.items.map(item => `<span class="skill-item">${item}</span>`).join('')}
        </div>
      `;

      skillsContainer.appendChild(skillCard);
    });
  }

  /**
   * Setup flower cursor animation
   */
  function setupFlowerCursor() {
    const cursor = document.querySelector('.flower-cursor');
    
    document.addEventListener('mousemove', (e) => {
      cursor.style.left = e.clientX - 30 + 'px';
      cursor.style.top = e.clientY - 30 + 'px';
    });
    
    document.addEventListener('mousedown', () => {
      cursor.classList.add('visible');
      setTimeout(() => {
        cursor.classList.remove('visible');
      }, 1000);
    });
    
    // Show cursor on interactive elements
    const interactiveElements = document.querySelectorAll('a, button, .project-card, .featured-project-card');
    
    interactiveElements.forEach(element => {
      element.addEventListener('mouseenter', () => {
        cursor.classList.add('visible');
      });
      
      element.addEventListener('mouseleave', () => {
        cursor.classList.remove('visible');
      });
    });
  }

  /**
   * Setup parallax effects
   */
  function setupParallaxEffects() {
    window.addEventListener('scroll', () => {
      // Parallax for the title
      if (parallaxTitle) {
        const scrollPosition = window.scrollY;
        parallaxTitle.style.transform = `translateY(${scrollPosition * 0.2}px)`;
        parallaxTitle.style.opacity = Math.max(1 - scrollPosition * 0.002, 0);
      }
      
      // Check if featured projects are in view
      const featuredProjects = document.querySelectorAll('.featured-project-card');
      featuredProjects.forEach((project, index) => {
        const rect = project.getBoundingClientRect();
        const isInView = rect.top < window.innerHeight * 0.8 && rect.bottom > 0;
        
        if (isInView && !project.classList.contains('visible')) {
          setTimeout(() => {
            project.classList.add('visible');
          }, index * 200);
        }
      });
    });
  }

  /**
   * Set up event listeners
   */
  function setupEventListeners() {
    // Tab navigation
    navTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const tabName = tab.getAttribute('data-tab');
        setActiveTab(tabName);
        
        // Scroll to top when changing tabs
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    });
  }

  /**
   * Set active tab
   * @param {string} tabName - The name of the tab to activate
   */
  function setActiveTab(tabName) {
    state.activeTab = tabName;

    // Update active tab UI
    navTabs.forEach(tab => {
      if (tab.getAttribute('data-tab') === tabName) {
        tab.classList.add('active');
      } else {
        tab.classList.remove('active');
      }
    });

    // Show/hide content sections
    contentSections.forEach(section => {
      if (section.id === tabName) {
        section.style.display = 'block';
        
        // Trigger animations for the newly visible section
        setTimeout(() => {
          section.classList.add('visible');
          
          // Animate cards
          const cards = section.querySelectorAll('.project-card, .skill-card, .about-card');
          cards.forEach((card, index) => {
            setTimeout(() => {
              card.style.opacity = '1';
              card.style.transform = 'translateY(0)';
            }, index * 100);
          });
        }, 50);
      } else {
        section.style.display = 'none';
        section.classList.remove('visible');
      }
    });
  }

  // Initialize 3D cube
  function initializeCube() {
    const container = document.querySelector('.cube-container');
    if (!container) return;

    const faces = [
      { image: portfolioData.profileImages[0], rotation: 'rotateY(0deg)' },
      { image: portfolioData.profileImages[1], rotation: 'rotateY(90deg)' },
      { image: portfolioData.profileImages[2], rotation: 'rotateY(180deg)' },
      { image: portfolioData.profileImages[3], rotation: 'rotateY(270deg)' },
      { image: portfolioData.profileImages[4], rotation: 'rotateX(90deg)' },
      { image: portfolioData.profileImages[5], rotation: 'rotateX(-90deg)' }
    ];

    faces.forEach(face => {
      const faceElement = document.createElement('div');
      faceElement.className = 'cube-face';
      faceElement.style.transform = face.rotation;
      faceElement.style.backgroundImage = `url(${face.image})`;
      container.appendChild(faceElement);
    });

    // Add continuous rotation
    let rotation = 0;
    setInterval(() => {
      rotation += 0.5;
      container.style.transform = `rotateY(${rotation}deg)`;
    }, 50);
  }

  // 3D Cube rendering
  function renderCube() {
    const container = document.getElementById('cube-container');
    if (!container) return;
    container.innerHTML = '';
    const faces = [
      { rotation: 'rotateY(0deg) translateZ(100px)' },
      { rotation: 'rotateY(90deg) translateZ(100px)' },
      { rotation: 'rotateY(180deg) translateZ(100px)' },
      { rotation: 'rotateY(-90deg) translateZ(100px)' },
      { rotation: 'rotateX(90deg) translateZ(100px)' },
      { rotation: 'rotateX(-90deg) translateZ(100px)' }
    ];
    faces.forEach(face => {
      const faceDiv = document.createElement('div');
      faceDiv.className = 'cube-face';
      faceDiv.style.transform = face.rotation;
      const img = document.createElement('img');
      img.src = 'assets/images/my_illustrated_photo.jpeg';
      img.alt = 'Komal Shahid Portrait';
      faceDiv.appendChild(img);
      container.appendChild(faceDiv);
    });
    let angle = 0;
    setInterval(() => {
      angle += 0.5;
      container.style.transform = `rotateX(${angle}deg) rotateY(${angle}deg)`;
    }, 40);
  }

  // Vanta.js for featured projects wavy background
  if (window.VANTA) {
    VANTA.WAVES({
      el: '#vanta-bg',
      color: 0xff96c7,
      shininess: 60,
      waveHeight: 28,
      waveSpeed: 0.8,
      zoom: 1.1,
      backgroundColor: 0xffb86c
    });
  }

  // Lottie for flower animation in About Me
  if (window.lottie) {
    lottie.loadAnimation({
      container: document.getElementById('about-flowers-lottie'),
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/lottie/flowers.json' // Replace with your Lottie flower file
    });
    // Lottie for featured project thumbnail
    lottie.loadAnimation({
      container: document.getElementById('featured-lottie'),
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/lottie/ai-graph.json' // Replace with your Lottie AI/graph animation
    });
  }

  // Modal for project details
  function openProjectModal(projectId) {
    const modal = document.getElementById('project-modal');
    if (!modal) return;
    let content = '';
    if (projectId === 'federated-healthcare-ai') {
      content = `
        <div class="project-modal-content">
          <button class="project-modal-close" onclick="closeProjectModal()">&times;</button>
          <h2>Federated Healthcare AI</h2>
          <div style="width:100px;height:100px;margin:0 auto 1rem auto;" id="modal-lottie-thumb"></div>
          <p>Privacy-preserving machine learning for healthcare data across multiple institutions. This project implements a federated learning framework that enables multiple healthcare institutions to collaboratively train AI models without sharing raw patient data. The system ensures privacy and compliance while leveraging distributed data for improved model performance.</p>
          <ul>
            <li>Privacy-preserving federated learning protocol</li>
            <li>Support for multiple healthcare data modalities</li>
            <li>Differential privacy and secure aggregation</li>
            <li>Interactive simulation and visualization</li>
          </ul>
          <div class="project-links">
            <a href="https://github.com/UKOMAL/Federated-Healthcare-AI" target="_blank" class="project-link">GitHub</a>
            <a href="/projects/project2-federated-healthcare-ai/demo/index.html" target="_blank" class="project-link">Live Demo</a>
          </div>
        </div>
      `;
    }
    modal.innerHTML = content;
    modal.style.display = 'flex';
    // Lottie for modal
    if (window.lottie && document.getElementById('modal-lottie-thumb')) {
      lottie.loadAnimation({
        container: document.getElementById('modal-lottie-thumb'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        path: 'assets/lottie/ai-graph.json' // Same as featured project
      });
    }
  }
  function closeProjectModal() {
    const modal = document.getElementById('project-modal');
    if (modal) modal.style.display = 'none';
  }

  document.addEventListener('DOMContentLoaded', function() {
    renderCube();
  });
}); 