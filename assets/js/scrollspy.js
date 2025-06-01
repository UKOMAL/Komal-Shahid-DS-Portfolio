// Simple ScrollSpy for nav highlighting

document.addEventListener('DOMContentLoaded', function() {
  const sections = ['hero', 'expertise', 'projects', 'about', 'contact'];
  const navLinks = document.querySelectorAll('.nav-link');

  function onScroll() {
    let scrollPos = window.scrollY || window.pageYOffset;
    let found = false;
    for (let i = sections.length - 1; i >= 0; i--) {
      const section = document.getElementById(sections[i]);
      if (section && scrollPos + 100 >= section.offsetTop) {
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector('.nav-link[href*="' + sections[i] + '"]');
        if (activeLink) activeLink.classList.add('active');
        found = true;
        break;
      }
    }
    if (!found) navLinks.forEach(link => link.classList.remove('active'));
  }

  window.addEventListener('scroll', onScroll);
  onScroll();
}); 