/**
 * Main JavaScript file for Komal Shahid's portfolio
 * Handles tab navigation, project rendering, and animations
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize functionality
  initNavigation();
  initPortfolioFilters();
  initFlowerCursor();
  initScrollAnimations();
  initCube();
  
  // Handle project modals
  document.body.addEventListener('click', (e) => {
    if (e.target.classList.contains('project-modal-close') || 
        e.target.closest('.project-modal-close')) {
      closeProjectModal();
    }
  });
  
  // Add escape key handler for modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeProjectModal();
    }
  });
  
  // Initialize 3D cube for hero section
  init3DCube();
  
  // Setup mobile navigation
  setupMobileNav();
  
  // Initialize parallax scrolling
  initParallax();
  
  // Add scroll event for header
  window.addEventListener('scroll', () => {
    const header = document.querySelector('.main-header');
    if (window.scrollY > 50) {
      header.classList.add('scrolled');
    } else {
      header.classList.remove('scrolled');
    }
  });
  
  // Portfolio filters
  const filterBtns = document.querySelectorAll('.filter-btn');
  const portfolioItems = document.querySelectorAll('.portfolio-item');
  
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active class from all buttons
      filterBtns.forEach(b => b.classList.remove('active'));
      
      // Add active class to clicked button
      btn.classList.add('active');
      
      // Get filter value
      const filterValue = btn.getAttribute('data-filter');
      
      // Filter items
      portfolioItems.forEach(item => {
        if (filterValue === 'all' || item.getAttribute('data-category') === filterValue) {
          item.style.display = 'block';
          setTimeout(() => {
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
          }, 50);
        } else {
          item.style.opacity = '0';
          item.style.transform = 'translateY(20px)';
          setTimeout(() => {
            item.style.display = 'none';
          }, 300);
        }
      });
    });
  });
});

/**
 * Initialize navigation functionality
 */
function initNavigation() {
  // Handle smooth scrolling for navigation links
  document.querySelectorAll('.nav-tab').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = link.getAttribute('href');
      
      if (target && target !== '#') {
        const targetElement = document.querySelector(target);
        if (targetElement) {
          window.scrollTo({
            top: targetElement.offsetTop - 80, // Offset for header
            behavior: 'smooth'
          });
        }
      }
      
      // Set active tab
      document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
      });
      link.classList.add('active');
    });
  });
  
  // Mobile menu toggle
  const mobileMenuButton = document.querySelector('.mobile-menu-toggle');
  const mobileMenu = document.querySelector('.main-nav ul');
  
  if (mobileMenuButton && mobileMenu) {
    mobileMenuButton.addEventListener('click', () => {
      mobileMenu.classList.toggle('active');
    });
  }
  
  // Close mobile menu when clicking outside
  document.addEventListener('click', (e) => {
    if (mobileMenu && mobileMenu.classList.contains('active') && 
        !e.target.closest('.main-nav') && 
        !e.target.closest('.mobile-menu-toggle')) {
      mobileMenu.classList.remove('active');
    }
  });
  
  // Handle scroll-based navigation active state
  window.addEventListener('scroll', () => {
    const scrollPosition = window.scrollY;
    
    // Get all sections and find which one is currently in view
    const sections = document.querySelectorAll('section[id]');
    sections.forEach(section => {
      const sectionTop = section.offsetTop - 100;
      const sectionHeight = section.offsetHeight;
      
      if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
        // Set active tab based on visible section
        const id = section.getAttribute('id');
        document.querySelectorAll('.nav-tab').forEach(tab => {
          tab.classList.remove('active');
          if (tab.getAttribute('href') === `#${id}` || 
              (id === 'featured-project' && tab.getAttribute('data-tab') === 'featured')) {
            tab.classList.add('active');
          }
        });
      }
    });
  });
}

/**
 * Initialize portfolio filters
 */
function initPortfolioFilters() {
  const filterButtons = document.querySelectorAll('.filter-btn');
  const portfolioItems = document.querySelectorAll('.portfolio-item');
  
  if (!filterButtons.length || !portfolioItems.length) return;
  
  filterButtons.forEach(button => {
    button.addEventListener('click', () => {
      // Update active button
      filterButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      
      // Filter items
      const filterValue = button.getAttribute('data-filter');
      
      portfolioItems.forEach(item => {
        if (filterValue === 'all' || item.getAttribute('data-category') === filterValue) {
          item.style.display = 'block';
          setTimeout(() => {
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
          }, 50);
        } else {
          item.style.opacity = '0';
          item.style.transform = 'translateY(20px)';
          setTimeout(() => {
            item.style.display = 'none';
          }, 500);
        }
      });
    });
  });
}

/**
 * Initialize flower cursor animation
 */
function initFlowerCursor() {
  const cursor = document.querySelector('.flower-cursor');
  
  document.addEventListener('mousemove', (e) => {
    const x = e.clientX;
    const y = e.clientY;
    
    cursor.style.left = `${x}px`;
    cursor.style.top = `${y}px`;
    
    // Add subtle rotation based on mouse movement
    const rotateX = (y / window.innerHeight - 0.5) * 20;
    const rotateY = (x / window.innerWidth - 0.5) * 20;
    
    cursor.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
  });
  
  // Scale effect on click
  document.addEventListener('mousedown', () => {
    cursor.style.transform = `scale(1.5)`;
  });
  
  document.addEventListener('mouseup', () => {
    cursor.style.transform = `scale(1)`;
  });
}

/**
 * Initialize scroll animations
 */
function initScrollAnimations() {
  // Create intersection observer for animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
        
        // If it's the portfolio section, animate items sequentially
        if (entry.target.classList.contains('portfolio-section')) {
          const items = entry.target.querySelectorAll('.portfolio-item');
          items.forEach((item, index) => {
            setTimeout(() => {
              item.classList.add('in-view');
            }, 150 * index);
          });
        }
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: '0px 0px -10% 0px'
  });
  
  // Observe sections for animation
  document.querySelectorAll('section').forEach(section => {
    observer.observe(section);
  });
  
  // Scroll indicator opacity based on scroll position
  const scrollIndicator = document.querySelector('.scroll-indicator');
  if (scrollIndicator) {
    window.addEventListener('scroll', () => {
      const scrollPosition = window.scrollY;
      const opacity = 1 - Math.min(scrollPosition / 200, 1);
      scrollIndicator.style.opacity = opacity;
    });
  }
}

/**
 * Initialize the 3D cube
 */
function initCube() {
  const cubeContainer = document.getElementById('cube-container');
  if (!cubeContainer) return;
  
  // Create cube wrapper
  const cubeWrapper = document.createElement('div');
  cubeWrapper.className = 'cube-wrapper';
  cubeContainer.appendChild(cubeWrapper);
  
  // Create cube
  const cube = document.createElement('div');
  cube.className = 'cube';
  cubeWrapper.appendChild(cube);
  
  // Create cube faces with GitHub profile image and other content
  const faces = [
    { transform: 'translateZ(175px)', content: '<img src="https://github.com/ukomal.png" alt="Komal Shahid GitHub Profile">' },
    { transform: 'rotateY(90deg) translateZ(175px)', content: '<div class="cube-text">AI Engineer</div>' },
    { transform: 'rotateY(180deg) translateZ(175px)', content: '<div class="cube-icon"><i class="fas fa-brain"></i></div>' },
    { transform: 'rotateY(-90deg) translateZ(175px)', content: '<div class="cube-text">ML Expert</div>' },
    { transform: 'rotateX(90deg) translateZ(175px)', content: '<div class="cube-icon"><i class="fas fa-code"></i></div>' },
    { transform: 'rotateX(-90deg) translateZ(175px)', content: '<div class="cube-text">Data Scientist</div>' }
  ];
  
  // Add faces to cube
  faces.forEach(face => {
    const cubeFace = document.createElement('div');
    cubeFace.className = 'cube-face';
    cubeFace.style.transform = face.transform;
    cubeFace.innerHTML = face.content;
    cube.appendChild(cubeFace);
  });
}

/**
 * Open project modal with detailed information
 */
function openProjectModal(projectId) {
  console.log("Opening modal for project:", projectId);
  
  // Find project data
  const project = portfolioProjects.find(p => p.id === projectId);
  if (!project) {
    console.error("Project not found:", projectId);
    return;
  }
  
  // Get modal
  const modal = document.getElementById('project-modal');
  if (!modal) {
    console.error("Modal element not found!");
    return;
  }
  
  console.log("Modal element found:", modal);
  console.log("Project data:", project);
  
  // Create modal content
  const modalContent = `
    <div class="project-modal-content">
      <div class="project-modal-image">
        <img src="${project.image || 'assets/images/projects/placeholder.jpg'}" alt="${project.title}">
      </div>
      <div class="project-modal-info">
        <h2 class="project-modal-title">${project.title}</h2>
        <div class="project-modal-category">${project.category || 'Project'}</div>
        <p class="project-modal-description">${project.description}</p>
        
        <div class="project-modal-features">
          <h3>Key Features</h3>
          <ul>
            ${project.features.map(feature => `<li>${feature}</li>`).join('')}
          </ul>
        </div>
        
        <div class="project-modal-tech">
          <h3>Technologies</h3>
          <div class="tech-tags">
            ${project.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
          </div>
        </div>
        
        <div class="project-modal-challenges">
          <h3>Challenges & Solutions</h3>
          <p>${project.challenges}</p>
        </div>
        
        <div class="project-modal-actions">
          ${project.github ? `<a href="${project.github}" target="_blank" class="project-modal-btn btn-primary">
            <i class="fab fa-github"></i> GitHub
          </a>` : ''}
          
          ${project.demo ? `<button class="project-modal-btn btn-primary open-demo-btn" data-project-id="${projectId}">
            <i class="fas fa-play-circle"></i> Live Demo
          </button>` : ''}
          
          <button class="project-modal-btn btn-secondary project-modal-close">
            Close
          </button>
        </div>
      </div>
      <button class="project-modal-close" aria-label="Close">&times;</button>
    </div>
  `;
  
  // Set modal content and display
  modal.innerHTML = modalContent;
  modal.style.display = 'flex';
  
  // Force reflow to ensure transition works
  modal.offsetWidth;
  
  // Add active class for CSS transition
  modal.classList.add('active');
  
  // Add event listeners
  const closeButtons = modal.querySelectorAll('.project-modal-close');
  closeButtons.forEach(btn => {
    btn.addEventListener('click', closeProjectModal);
  });
  
  // Add demo button listener
  const demoBtn = modal.querySelector('.open-demo-btn');
  if (demoBtn) {
    demoBtn.addEventListener('click', () => {
      openProjectDemo(projectId);
    });
  }
  
  // Close on click outside
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      closeProjectModal();
    }
  });
  
  // Add escape key listener
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeProjectModal();
    }
  });
}

/**
 * Close project modal
 */
function closeProjectModal() {
  const modal = document.getElementById('project-modal');
  if (!modal) return;
  
  modal.classList.remove('active');
  
  // Wait for animation to finish before hiding
  setTimeout(() => {
    modal.style.display = 'none';
  }, 300);
}

/**
 * Open project demo modal
 */
function openProjectDemo(projectId) {
  // Get demo modal
  const demoModal = document.getElementById('project-demo-modal');
  const demoForm = document.getElementById('demo-form');
  const demoResult = document.getElementById('demo-result');
  
  // Clear previous results
  demoResult.innerHTML = '';
  
  // Set project ID for the form
  demoForm.setAttribute('data-project-id', projectId);
  
  // Configure the demo modal based on project type
  if (projectId === 'depression-detection') {
    document.querySelector('#demo-form label').textContent = 'Enter text to analyze:';
    document.querySelector('#demo-input').placeholder = 'e.g., I feel tired and hopeless lately...';
    document.querySelector('#demo-form button').textContent = 'Analyze Text';
    document.querySelector('.modal h2').textContent = 'Depression Detection Demo';
  } else if (projectId === 'federated-healthcare-ai') {
    document.querySelector('#demo-form label').textContent = 'Number of training rounds:';
    document.querySelector('#demo-input').placeholder = '10';
    document.querySelector('#demo-input').type = 'number';
    document.querySelector('#demo-input').min = '5';
    document.querySelector('#demo-input').max = '20';
    document.querySelector('#demo-input').value = '10';
    document.querySelector('#demo-form button').textContent = 'Run Simulation';
    document.querySelector('.modal h2').textContent = 'Federated Healthcare AI Demo';
  } else if (projectId === 'network-visualization') {
    document.querySelector('#demo-form label').textContent = 'Select dataset:';
    document.querySelector('#demo-input').outerHTML = `
      <select id="demo-input" name="demo-input">
        <option value="social">Social Network</option>
        <option value="healthcare">Healthcare Network</option>
        <option value="research">Research Collaboration</option>
      </select>
    `;
    document.querySelector('#demo-form button').textContent = 'Visualize Network';
    document.querySelector('.modal h2').textContent = 'Network Visualization Demo';
  }
  
  // Show modal
  demoModal.style.display = 'flex';
  
  // Add event listeners
  document.querySelector('.modal-close').addEventListener('click', () => {
    demoModal.style.display = 'none';
  });
  
  // Initialize the demo functionality
  if (window.projectDemos && typeof window.projectDemos.initialize === 'function') {
    window.projectDemos.initialize();
  }
}

/**
 * Initialize 3D cube
 */
function init3DCube() {
  const container = document.getElementById('cube-container');
  if (!container) return;
  
  // Set up scene
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
  
  const renderer = new THREE.WebGLRenderer({ alpha: true });
  renderer.setSize(container.offsetWidth, container.offsetHeight);
  container.appendChild(renderer.domElement);
  
  // Create cube
  const geometry = new THREE.BoxGeometry(3, 3, 3);
  
  // Create materials with GitHub profile image
  const materials = [
    new THREE.MeshBasicMaterial({ color: 0xff6b6b }),
    new THREE.MeshBasicMaterial({ color: 0x7971ea }),
    new THREE.MeshBasicMaterial({ color: 0x4ecdc4 }),
    new THREE.MeshBasicMaterial({ color: 0xffbe0b }),
    new THREE.MeshBasicMaterial({ color: 0xff6b6b }),
    new THREE.MeshBasicMaterial({ color: 0x7971ea })
  ];
  
  // Front face - load GitHub profile image if available
  const loader = new THREE.TextureLoader();
  loader.load(
    'https://github.com/ukomal.png',
    function(texture) {
      materials[0] = new THREE.MeshBasicMaterial({ map: texture });
      cube.material[0] = materials[0];
    },
    undefined,
    function(err) {
      console.error('Error loading GitHub profile image');
    }
  );
  
  const cube = new THREE.Mesh(geometry, materials);
  scene.add(cube);
  
  camera.position.z = 5;
  
  // Animation function
  function animate() {
    requestAnimationFrame(animate);
    
    // Rotate the cube
    cube.rotation.x += 0.005;
    cube.rotation.y += 0.01;
    
    renderer.render(scene, camera);
  }
  
  animate();
  
  // Handle window resize
  window.addEventListener('resize', () => {
    camera.aspect = container.offsetWidth / container.offsetHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.offsetWidth, container.offsetHeight);
  });
}

// Setup mobile navigation
function setupMobileNav() {
  const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
  const mainNav = document.querySelector('.main-nav');
  
  if (mobileMenuToggle && mainNav) {
    mobileMenuToggle.addEventListener('click', () => {
      mainNav.classList.toggle('active');
      
      // Toggle icon
      const icon = mobileMenuToggle.querySelector('i');
      if (icon) {
        if (icon.classList.contains('fa-bars')) {
          icon.classList.remove('fa-bars');
          icon.classList.add('fa-times');
        } else {
          icon.classList.remove('fa-times');
          icon.classList.add('fa-bars');
        }
      }
    });
  }
}

// Initialize parallax scrolling
function initParallax() {
  // Parallax elements
  const elements = [
    { selector: '.hero-wave-bg', speed: 0.1 },
    { selector: '.color-streak', speed: 0.05 },
    { selector: '.filter', speed: 0.02 },
    { selector: '.geometry-shape', speed: 0.08 },
    { selector: '.featured-wave-background', speed: 0.03 }
  ];
  
  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    
    elements.forEach(element => {
      const items = document.querySelectorAll(element.selector);
      
      items.forEach(item => {
        const yPos = scrollY * element.speed;
        item.style.transform = `translateY(${yPos}px)`;
      });
    });
  });
} 