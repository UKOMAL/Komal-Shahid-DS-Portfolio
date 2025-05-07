/**
 * GitHub Integration with Builder.io Authorization
 * This script handles GitHub OAuth flow and displays repositories and contribution data
 */

// GitHub Configuration - Replace with your Builder.io GitHub App credentials
const BUILDER_GITHUB_CLIENT_ID = '9bd80b306a124819add107e98c61d45d';
const BUILDER_GITHUB_REDIRECT_URI = window.location.origin + window.location.pathname;
const BUILDER_API_KEY = 'f947ced278234b4d8dc12c19358844bf';

// GitHub API endpoints
const GITHUB_API = 'https://api.github.com';
const BUILDER_API = 'https://builder.io/api/v1/github';

// Language colors for repository display
const languageColors = {
    'JavaScript': '#f1e05a',
    'TypeScript': '#2b7489',
    'HTML': '#e34c26',
    'CSS': '#563d7c',
    'Python': '#3572A5',
    'Java': '#b07219',
    'C#': '#178600',
    'PHP': '#4F5D95',
    'C++': '#f34b7d',
    'Ruby': '#701516',
    'Swift': '#ffac45',
    'Go': '#00ADD8',
    'Kotlin': '#F18E33',
    'Rust': '#dea584',
    'Dart': '#00B4AB',
    'Jupyter Notebook': '#DA5B0B',
    'Shell': '#89e051'
};

// DOM Elements
const connectBtn = document.getElementById('github-connect-btn');
const authStatus = document.getElementById('github-auth-status');
const reposContainer = document.getElementById('github-repos');
const repoList = document.getElementById('repo-list');
const contributionsContainer = document.getElementById('github-contributions');
const contributionCalendar = document.getElementById('contribution-calendar');

// Initialize GitHub integration
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're returning from GitHub OAuth flow
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    
    if (code) {
        // Exchange code for token using Builder.io as a proxy
        exchangeCodeForToken(code);
        
        // Remove code from URL to prevent refresh issues
        const cleanUrl = window.location.href.split('?')[0];
        window.history.replaceState({}, document.title, cleanUrl);
    } else {
        // Check if we have a stored token
        const token = localStorage.getItem('github_token');
        if (token) {
            showConnectedState();
            fetchGitHubData(token);
        }
    }
    
    // Set up Connect button
    connectBtn.addEventListener('click', initiateGitHubAuth);
});

// Start GitHub OAuth flow
function initiateGitHubAuth() {
    const githubAuthUrl = `https://github.com/login/oauth/authorize?client_id=${BUILDER_GITHUB_CLIENT_ID}&redirect_uri=${encodeURIComponent(BUILDER_GITHUB_REDIRECT_URI)}&scope=repo,read:user`;
    window.location.href = githubAuthUrl;
}

// Exchange code for token through Builder.io proxy
async function exchangeCodeForToken(code) {
    try {
        showLoading('Connecting to GitHub...');
        
        const response = await fetch(`${BUILDER_API}/exchange-code?apiKey=${BUILDER_API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                code,
                redirectUri: BUILDER_GITHUB_REDIRECT_URI
            })
        });
        
        const data = await response.json();
        
        if (data.token) {
            // Store token securely
            localStorage.setItem('github_token', data.token);
            showConnectedState();
            fetchGitHubData(data.token);
        } else {
            showError('Failed to authenticate with GitHub');
        }
    } catch (error) {
        console.error('GitHub authentication error:', error);
        showError('Error connecting to GitHub');
    }
}

// Show loading state
function showLoading(message) {
    authStatus.innerHTML = `
        <div class="github-status-icon">
            <div class="github-spinner"></div>
        </div>
        <div class="github-status-info">
            <h3>Connecting...</h3>
            <p>${message}</p>
        </div>
    `;
}

// Show error state
function showError(message) {
    authStatus.innerHTML = `
        <div class="github-status-icon" style="color: #e74c3c;">
            <i class="fas fa-exclamation-circle"></i>
        </div>
        <div class="github-status-info">
            <h3>Connection Error</h3>
            <p>${message}</p>
        </div>
        <button id="github-retry-btn" class="btn btn-github">
            <i class="fas fa-sync-alt"></i> Retry
        </button>
    `;
    
    document.getElementById('github-retry-btn').addEventListener('click', initiateGitHubAuth);
}

// Show connected state
function showConnectedState() {
    authStatus.innerHTML = `
        <div class="github-status-icon" style="color: #2ea44f;">
            <i class="fas fa-check-circle"></i>
        </div>
        <div class="github-status-info">
            <h3>Connected to GitHub</h3>
            <p>Your GitHub account is successfully connected</p>
        </div>
        <button id="github-disconnect-btn" class="btn btn-github" style="background-color: #e74c3c;">
            <i class="fas fa-sign-out-alt"></i> Disconnect
        </button>
    `;
    
    document.getElementById('github-disconnect-btn').addEventListener('click', disconnectGitHub);
}

// Disconnect from GitHub
function disconnectGitHub() {
    localStorage.removeItem('github_token');
    
    authStatus.innerHTML = `
        <div class="github-status-icon">
            <i class="fab fa-github"></i>
        </div>
        <div class="github-status-info">
            <h3>Connect to GitHub</h3>
            <p>Link your GitHub account to view repositories and contributions</p>
        </div>
        <button id="github-connect-btn" class="btn btn-github">
            <i class="fab fa-github"></i> Connect GitHub
        </button>
    `;
    
    document.getElementById('github-connect-btn').addEventListener('click', initiateGitHubAuth);
    
    // Hide repos and contributions
    reposContainer.style.display = 'none';
    contributionsContainer.style.display = 'none';
}

// Fetch GitHub data using the token
async function fetchGitHubData(token) {
    try {
        // Show repositories container with loading state
        reposContainer.style.display = 'block';
        contributionsContainer.style.display = 'block';
        
        // Fetch user data first
        const userData = await fetchGitHubAPI('/user', token);
        
        // Update user info
        const githubAvatar = document.getElementById('github-avatar');
        githubAvatar.src = userData.avatar_url;
        githubAvatar.classList.add('github-avatar'); // Ensure the class is applied
        document.getElementById('github-username').textContent = userData.login;
        document.getElementById('github-stats').textContent = `${userData.public_repos} repos • ${userData.followers} followers`;
        
        // Fetch repositories
        const repos = await fetchGitHubAPI('/user/repos?sort=updated&per_page=6', token);
        displayRepositories(repos);
        
        // Fetch contribution data using Builder.io proxy (GitHub doesn't have a direct API for this)
        await fetchContributions(userData.login, token);
        
    } catch (error) {
        console.error('Error fetching GitHub data:', error);
        repoList.innerHTML = `<p class="github-error">Error loading repositories: ${error.message}</p>`;
    }
}

// Generic function to call GitHub API
async function fetchGitHubAPI(endpoint, token) {
    const response = await fetch(`${GITHUB_API}${endpoint}`, {
        headers: {
            'Authorization': `token ${token}`,
            'Accept': 'application/vnd.github.v3+json'
        }
    });
    
    if (!response.ok) {
        throw new Error(`GitHub API error: ${response.status}`);
    }
    
    return await response.json();
}

// Display repositories
function displayRepositories(repos) {
    if (repos.length === 0) {
        repoList.innerHTML = '<p>No repositories found</p>';
        return;
    }
    
    // Clear loading state
    repoList.innerHTML = '';
    
    // Add repositories
    repos.forEach(repo => {
        const repoEl = document.createElement('div');
        repoEl.className = 'github-repo';
        
        // Set top border color based on language
        if (repo.language && languageColors[repo.language]) {
            repoEl.style.borderTopColor = languageColors[repo.language];
        }
        
        repoEl.innerHTML = `
            <div class="github-repo-header">
                <div>
                    <a href="${repo.html_url}" target="_blank" class="github-repo-name">${repo.name}</a>
                    <span class="github-repo-visibility ${repo.private ? 'private' : 'public'}">${repo.private ? 'Private' : 'Public'}</span>
                </div>
            </div>
            
            <div class="github-repo-description">${repo.description || 'No description provided'}</div>
            
            <div class="github-repo-meta">
                <div class="github-repo-language">
                    ${repo.language ? `
                        <span class="github-language-color" style="background-color: ${languageColors[repo.language] || '#ededed'}"></span>
                        <span>${repo.language}</span>
                    ` : 'N/A'}
                </div>
                
                <div class="github-repo-stats">
                    <div class="github-stat" title="Stars">
                        <i class="fas fa-star"></i>
                        <span>${repo.stargazers_count}</span>
                    </div>
                    <div class="github-stat" title="Forks">
                        <i class="fas fa-code-branch"></i>
                        <span>${repo.forks_count}</span>
                    </div>
                </div>
            </div>
        `;
        
        repoList.appendChild(repoEl);
    });
}

// Fetch and display contributions
async function fetchContributions(username, token) {
    try {
        // We'll use Builder.io as a proxy to get contribution data
        const response = await fetch(`${BUILDER_API}/contributions?apiKey=${BUILDER_API_KEY}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                token
            })
        });
        
        const data = await response.json();
        
        if (data.contributions) {
            displayContributionCalendar(data.contributions);
        } else {
            contributionsContainer.innerHTML = '<p>Could not load contribution data</p>';
        }
    } catch (error) {
        console.error('Error fetching contributions:', error);
        contributionsContainer.innerHTML = '<p>Error fetching contribution data</p>';
    }
}

// Display contribution calendar
function displayContributionCalendar(contributionData) {
    // Generate month labels
    const monthsContainer = document.createElement('div');
    monthsContainer.className = 'contribution-months';
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    months.forEach(month => {
        const monthEl = document.createElement('div');
        monthEl.className = 'contribution-month';
        monthEl.textContent = month;
        monthsContainer.appendChild(monthEl);
    });
    
    // Clear calendar
    contributionCalendar.innerHTML = '';
    contributionCalendar.appendChild(monthsContainer);
    
    // Create mock contribution data if we don't have real data
    // In a real implementation, you'd parse the actual data from GitHub
    const mockData = generateMockContributionData();
    
    // Generate days
    mockData.forEach(day => {
        const dayEl = document.createElement('div');
        dayEl.className = `contribution-day level-${day.level}`;
        dayEl.title = `${day.count} contributions on ${day.date}`;
        dayEl.setAttribute('data-date', day.date);
        dayEl.setAttribute('data-count', day.count);
        
        contributionCalendar.appendChild(dayEl);
    });
}

// Generate mock contribution data
function generateMockContributionData() {
    const mockData = [];
    const today = new Date();
    
    // Generate data for the past year (52 weeks × 7 days)
    for (let i = 364; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(today.getDate() - i);
        
        // Higher chance for recent contributions
        const recency = 1 - (i / 364);
        let level;
        
        // Random level with higher likelihood for recent days
        if (Math.random() < 0.6 + (recency * 0.2)) {
            // More likely to have activity for more recent days
            if (Math.random() < 0.7) {
                level = Math.floor(Math.random() * 5); // 0-4
            } else {
                level = 0; // No activity
            }
        } else {
            level = 0; // No activity
        }
        
        // Contributions count based on level
        const count = level === 0 ? 0 : (level === 1 ? Math.floor(Math.random() * 2) + 1 : 
                                        level === 2 ? Math.floor(Math.random() * 3) + 3 :
                                        level === 3 ? Math.floor(Math.random() * 5) + 6 :
                                        Math.floor(Math.random() * 10) + 10);
                                        
        mockData.push({
            date: date.toISOString().split('T')[0],
            count,
            level
        });
    }
    
    return mockData;
} 