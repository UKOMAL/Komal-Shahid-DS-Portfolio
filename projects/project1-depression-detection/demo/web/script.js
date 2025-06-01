/**
 * Depression Detection System - Interactive Demo
 * This is a simulated demo that mimics the behavior of the actual system
 */

document.addEventListener('DOMContentLoaded', function() {
  // DOM Elements
  const analyzeBtn = document.getElementById('analyzeBtn');
  const inputText = document.getElementById('inputText');
  const loadingSection = document.getElementById('loading');
  const resultSection = document.getElementById('resultSection');
  const resultBadge = document.getElementById('resultBadge');
  const resultSummary = document.getElementById('resultSummary');
  const minimumBar = document.getElementById('minimumBar');
  const minimumScore = document.getElementById('minimumScore');
  const mildBar = document.getElementById('mildBar');
  const mildScore = document.getElementById('mildScore');
  const moderateBar = document.getElementById('moderateBar');
  const moderateScore = document.getElementById('moderateScore');
  const severeBar = document.getElementById('severeBar');
  const severeScore = document.getElementById('severeScore');
  const insightsList = document.getElementById('insightsList');
  
  // Sample depression keywords and patterns
  const depressionKeywords = {
    severe: ['suicide', 'die', 'death', 'hopeless', 'worthless', 'can\'t go on', 'better off without me', 'unbearable'],
    moderate: ['exhausted', 'always tired', 'no interest', 'nothing matters', 'struggle', 'difficult', 'can\'t sleep', 'no energy'],
    mild: ['sad', 'unhappy', 'down', 'blue', 'stressed', 'worried', 'anxious', 'upset'],
    minimum: ['good', 'happy', 'great', 'excited', 'looking forward', 'enjoyed', 'positive', 'hope']
  };
  
  // Add robot sparkle effect on hover
  document.addEventListener('mousemove', createSparkleEffect);
  
  // Analyze button click handler
  analyzeBtn.addEventListener('click', function() {
    const text = inputText.value.trim();
    
    if (text.length < 10) {
      alert('Please enter a longer text for analysis (at least 10 characters).');
      return;
    }
    
    // Show loading spinner
    loadingSection.style.display = 'block';
    resultSection.style.display = 'none';
    
    // Simulate AI processing delay
    setTimeout(() => analyzeText(text), 1500);
  });
  
  // Text analysis function
  function analyzeText(text) {
    // Convert to lowercase for easier matching
    const lowercaseText = text.toLowerCase();
    
    // Count keyword matches for each category
    const keywordMatches = {
      severe: 0,
      moderate: 0,
      mild: 0,
      minimum: 0
    };
    
    // Count keyword occurrences in text
    Object.keys(depressionKeywords).forEach(category => {
      depressionKeywords[category].forEach(keyword => {
        if (lowercaseText.includes(keyword)) {
          keywordMatches[category]++;
        }
      });
    });
    
    // Simple linguistic features
    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    const sentenceCount = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0).length;
    const avgWordsPerSentence = sentenceCount > 0 ? wordCount / sentenceCount : 0;
    
    const negationCount = (lowercaseText.match(/\b(not|no|never|don't|can't|won't|couldn't|shouldn't)\b/g) || []).length;
    const firstPersonCount = (lowercaseText.match(/\b(i|me|my|mine|myself)\b/g) || []).length;
    const firstPersonRatio = wordCount > 0 ? firstPersonCount / wordCount : 0;
    
    // Calculate raw confidence scores
    // This is a simplified version of what a real ML model would do
    const negationFactor = negationCount * 0.05;
    const firstPersonFactor = firstPersonRatio > 0.1 ? firstPersonRatio * 5 : 0;
    
    // Base scores 
    let scores = {
      minimum: 0.2 + (keywordMatches.minimum * 0.15),
      mild: 0.1 + (keywordMatches.mild * 0.15),
      moderate: 0.05 + (keywordMatches.moderate * 0.15),
      severe: 0.02 + (keywordMatches.severe * 0.2)
    };
    
    // Apply linguistic adjustments
    if (negationCount > 0 && keywordMatches.minimum > 0) {
      // Negation of positive words (e.g., "not happy") reduces minimum score
      scores.minimum -= negationFactor;
      scores.mild += negationFactor * 0.5;
      scores.moderate += negationFactor * 0.3;
    }
    
    // First person pronouns are common in depression narratives
    scores.mild += firstPersonFactor * 0.1;
    scores.moderate += firstPersonFactor * 0.15;
    scores.severe += firstPersonFactor * 0.1;
    
    // Longer texts with higher average words per sentence can indicate rumination
    if (wordCount > 50 && avgWordsPerSentence > 15) {
      scores.moderate += 0.1;
      scores.severe += 0.05;
    }
    
    // Caps on scores
    Object.keys(scores).forEach(key => {
      scores[key] = Math.max(0, Math.min(1, scores[key]));
    });
    
    // Normalize scores to sum to 1
    const totalScore = Object.values(scores).reduce((sum, score) => sum + score, 0);
    if (totalScore > 0) {
      Object.keys(scores).forEach(key => {
        scores[key] = scores[key] / totalScore;
      });
    }
    
    // Determine predominant category
    let predictedCategory = 'minimum';
    let highestScore = scores.minimum;
    
    Object.keys(scores).forEach(category => {
      if (scores[category] > highestScore) {
        highestScore = scores[category];
        predictedCategory = category;
      }
    });
    
    // Generate insights based on text analysis
    const insights = generateInsights(text, keywordMatches, negationCount, firstPersonRatio, wordCount, avgWordsPerSentence);
    
    // Display results
    displayResults(scores, predictedCategory, insights);
  }
  
  // Generate insights from text analysis
  function generateInsights(text, keywordMatches, negationCount, firstPersonRatio, wordCount, avgWordsPerSentence) {
    const insights = [];
    
    // Linguistic structure insights
    if (wordCount < 30) {
      insights.push('Short text length may limit analysis accuracy.');
    }
    
    if (firstPersonRatio > 0.15) {
      insights.push('High frequency of first-person pronouns detected, which can be associated with self-focused rumination.');
    }
    
    if (avgWordsPerSentence > 20) {
      insights.push('Longer than average sentences may indicate complex thought patterns.');
    }
    
    // Emotional content insights
    if (keywordMatches.severe > 0) {
      insights.push('Text contains expressions that suggest significant emotional distress.');
    }
    
    if (negationCount > 3) {
      insights.push('Multiple negations detected, potentially indicating negative outlook.');
    }
    
    // Look for specific patterns
    if (keywordMatches.minimum > 0 && negationCount > 1) {
      insights.push('Negation of positive emotions detected, which may indicate contrast between expected and actual feelings.');
    }
    
    // Ensure we have at least 2 insights
    if (insights.length < 2) {
      insights.push('Text analysis indicates normal emotional expression patterns.');
      insights.push('Consider providing more detailed text for more comprehensive analysis.');
    }
    
    return insights;
  }
  
  // Display the analysis results
  function displayResults(scores, predictedCategory, insights) {
    // Hide loading, show results
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    // Update confidence bars and scores
    updateConfidenceBar(minimumBar, minimumScore, scores.minimum);
    updateConfidenceBar(mildBar, mildScore, scores.mild);
    updateConfidenceBar(moderateBar, moderateScore, scores.moderate);
    updateConfidenceBar(severeBar, severeScore, scores.severe);
    
    // Set badge and summary based on predicted category
    resultBadge.textContent = capitalizeFirstLetter(predictedCategory);
    resultBadge.className = 'result-badge ' + predictedCategory;
    
    // Set result summary
    resultSummary.textContent = getCategorySummary(predictedCategory);
    
    // Display insights
    insightsList.innerHTML = '';
    insights.forEach(insight => {
      const insightItem = document.createElement('div');
      insightItem.className = 'insight-item';
      insightItem.innerHTML = `<span class="insight-icon">→</span> ${insight}`;
      insightsList.appendChild(insightItem);
    });
  }
  
  // Update confidence bar visualization
  function updateConfidenceBar(barElement, scoreElement, score) {
    const percentage = Math.round(score * 100);
    barElement.style.width = percentage + '%';
    scoreElement.textContent = percentage + '%';
  }
  
  // Get summary text for the depression severity category
  function getCategorySummary(category) {
    switch(category) {
      case 'minimum':
        return 'The text indicates minimal or no signs of depression. The language used suggests positive or neutral emotional states.';
      case 'mild':
        return 'The text shows some indicators of mild depressive symptoms. There are signs of sadness or stress, but they appear manageable.';
      case 'moderate':
        return 'The text contains several indicators of moderate depressive symptoms. The language suggests persistent negative feelings and potential difficulties in daily functioning.';
      case 'severe':
        return 'The text contains strong indicators of severe depressive symptoms. The language suggests significant emotional distress and potentially concerning thought patterns.';
      default:
        return 'Analysis inconclusive. Please provide more text for a more accurate assessment.';
    }
  }
  
  // Create sparkle effect on mouse movement
  function createSparkleEffect(e) {
    if (Math.random() > 0.98) { // Only create sparkles occasionally
      const sparkle = document.createElement('div');
      sparkle.className = 'sparkle';
      sparkle.style.left = e.pageX + 'px';
      sparkle.style.top = e.pageY + 'px';
      document.body.appendChild(sparkle);
      
      // Remove sparkle after animation completes
      setTimeout(() => {
        if (sparkle.parentElement) {
          document.body.removeChild(sparkle);
        }
      }, 800);
    }
  }
  
  // Helper function to capitalize first letter
  function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }
}); 