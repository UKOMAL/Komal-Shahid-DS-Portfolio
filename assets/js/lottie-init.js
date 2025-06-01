/**
 * Lottie Animations Initialization
 * This file handles all Lottie animations for the portfolio
 */

document.addEventListener('DOMContentLoaded', function() {
  // Initialize the Lottie animations
  initLottieAnimations();
});

function initLottieAnimations() {
  // Lottie cursor
  const cursorLottie = document.getElementById('lottie-cursor');
  if (cursorLottie) {
    const cursorAnimation = lottie.loadAnimation({
      container: cursorLottie,
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/animations/cursor-animation.json'
    });
    
    // Track mouse movement
    document.addEventListener('mousemove', function(e) {
      cursorLottie.style.left = (e.clientX - 24) + 'px';
      cursorLottie.style.top = (e.clientY - 24) + 'px';
    });
  }
  
  // Project animations
  const projectAnimations = [
    {
      id: 'lottie-project-1',
      path: 'assets/animations/federated-learning.json',
      bg: 'linear-gradient(135deg, #ff96c7, #bfc5fe)'
    },
    {
      id: 'lottie-project-2',
      path: 'assets/animations/depression-detection.json',
      bg: 'linear-gradient(135deg, #a896ff, #ff9696)'
    },
    {
      id: 'lottie-project-3',
      path: 'assets/animations/network-visualization.json',
      bg: 'linear-gradient(135deg, #96a8ff, #ffc5fe)'
    },
    {
      id: 'lottie-project-4',
      path: 'assets/animations/healthcare-analysis.json',
      bg: 'linear-gradient(135deg, #96ffcb, #fefec5)'
    }
  ];
  
  // Load project animations
  projectAnimations.forEach(animation => {
    const container = document.getElementById(animation.id);
    if (container) {
      // Add background gradient
      container.style.background = animation.bg;
      
      // If animation path is available, load Lottie, otherwise use fallback
      if (isLottieAvailable(animation.path)) {
        lottie.loadAnimation({
          container: container,
          renderer: 'svg',
          loop: true,
          autoplay: true,
          path: animation.path
        });
      } else {
        // Fallback to CSS animation
        container.classList.add('fallback-animation');
        
        // Create abstract shapes for fallback
        const shapes = ['circle', 'square', 'triangle', 'pentagon'];
        for (let i = 0; i < 4; i++) {
          const shape = document.createElement('div');
          shape.className = `shape shape-${shapes[i % shapes.length]}`;
          shape.style.animationDelay = `${i * 0.2}s`;
          container.appendChild(shape);
        }
      }
      
      // Add hover interaction
      container.addEventListener('mouseenter', function() {
        container.style.transform = 'scale(1.05)';
      });
      
      container.addEventListener('mouseleave', function() {
        container.style.transform = 'scale(1)';
      });
    }
  });
  
  // Hero animation
  const heroLottie = document.getElementById('lottie-hero');
  if (heroLottie) {
    lottie.loadAnimation({
      container: heroLottie,
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/animations/hero-animation.json'
    });
  }
  
  // About section animation
  const aboutLottie = document.getElementById('lottie-about');
  if (aboutLottie) {
    lottie.loadAnimation({
      container: aboutLottie,
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/animations/abstract-lines.json'
    });
  }
  
  // Contact section animation
  const contactLottie = document.getElementById('lottie-contact');
  if (contactLottie) {
    lottie.loadAnimation({
      container: contactLottie,
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/animations/contact-bg.json'
    });
  }
  
  // Footer animation
  const footerLottie = document.getElementById('lottie-footer');
  if (footerLottie) {
    lottie.loadAnimation({
      container: footerLottie,
      renderer: 'svg',
      loop: true,
      autoplay: true,
      path: 'assets/animations/footer-waves.json'
    });
  }
}

// Check if a Lottie animation file is available
function isLottieAvailable(path) {
  // In a real app, we would check if the file exists
  // For demo purposes, assume it exists if path is not empty
  return !!path;
} 