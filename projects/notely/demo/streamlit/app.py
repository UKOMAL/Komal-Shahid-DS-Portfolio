import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Notely - Smart Note Management",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .note-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📝 Notely - Smart Note Management</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Note Organization & Semantic Search</p>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a feature to explore:", [
    "🏠 Overview",
    "✍️ Note Creation",
    "🔍 Semantic Search", 
    "🏷️ AI Categorization",
    "📊 Analytics Dashboard",
    "👥 Collaboration"
])

# Sample data
@st.cache_data
def load_sample_notes():
    categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
    priorities = ["High", "Medium", "Low"]
    
    sample_notes = [
        {
            "title": "Machine Learning Project Ideas",
            "content": "Explore federated learning applications in healthcare, computer vision for autonomous vehicles, and NLP for sentiment analysis in social media",
            "category": "Research",
            "priority": "High",
            "tags": ["ML", "AI", "Healthcare", "Computer Vision"],
            "created_date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "ai_confidence": 0.92
        },
        {
            "title": "Weekly Team Meeting Notes",
            "content": "Discussed quarterly goals, project timelines, and resource allocation. Need to follow up on budget approval for new tools",
            "category": "Meetings",
            "priority": "Medium",
            "tags": ["Team", "Goals", "Budget", "Follow-up"],
            "created_date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "ai_confidence": 0.89
        },
        {
            "title": "Python Data Analysis Tips",
            "content": "Use pandas for data manipulation, matplotlib/seaborn for visualization, and scikit-learn for machine learning. Remember to validate data quality first",
            "category": "Learning",
            "priority": "Medium",
            "tags": ["Python", "Data Science", "Programming", "Tips"],
            "created_date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "ai_confidence": 0.95
        },
        {
            "title": "Vacation Planning",
            "content": "Research destinations for summer vacation. Consider budget, weather, and activities. Maybe somewhere with good hiking trails and cultural attractions",
            "category": "Personal",
            "priority": "Low",
            "tags": ["Vacation", "Travel", "Planning", "Summer"],
            "created_date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "ai_confidence": 0.87
        },
        {
            "title": "Startup Idea: AI Note Assistant",
            "content": "Build an intelligent note-taking app that automatically categorizes notes, suggests tags, and helps with organization using NLP and machine learning",
            "category": "Ideas",
            "priority": "High",
            "tags": ["Startup", "AI", "App", "NLP", "Innovation"],
            "created_date": datetime.now() - timedelta(days=random.randint(1, 30)),
            "ai_confidence": 0.94
        }
    ]
    return sample_notes

notes_data = load_sample_notes()

# Overview Page
if page == "🏠 Overview":
    st.markdown("## 🌟 Welcome to Notely")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>📝</h3><h2>150+</h2><p>Total Notes</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🏷️</h3><h2>12</h2><p>Categories</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎯</h3><h2>95%</h2><p>AI Accuracy</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>⚡</h3><h2>0.3s</h2><p>Search Speed</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Key Features")
        st.markdown("""
        - **🤖 AI-Powered Categorization**: Automatically organizes your notes
        - **🔍 Semantic Search**: Find notes by meaning, not just keywords
        - **🏷️ Smart Tagging**: AI suggests relevant tags
        - **👥 Real-time Collaboration**: Share and edit notes with your team
        - **📊 Analytics Dashboard**: Track your note-taking patterns
        - **🔒 Privacy-First**: Your data stays secure and private
        """)
    
    with col2:
        st.markdown("### 📈 Usage Statistics")
        
        # Create sample usage data
        days = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        usage_data = pd.DataFrame({
            'Date': days,
            'Notes Created': np.random.poisson(3, len(days)),
            'Searches': np.random.poisson(8, len(days))
        })
        
        fig = px.line(usage_data, x='Date', y=['Notes Created', 'Searches'], 
                     title="Daily Activity", height=300)
        st.plotly_chart(fig, use_container_width=True)

# Note Creation Page
elif page == "✍️ Note Creation":
    st.markdown("## ✍️ Create a New Note")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        title = st.text_input("📝 Note Title", placeholder="Enter your note title...")
        content = st.text_area("📄 Note Content", height=200, 
                              placeholder="Start writing your note here...")
        
        if st.button("🚀 Create Note", type="primary"):
            if title and content:
                # Simulate AI categorization
                categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
                predicted_category = random.choice(categories)
                confidence = round(random.uniform(0.8, 0.98), 2)
                
                # Simulate tag extraction
                sample_tags = ["AI", "Project", "Important", "Review", "Follow-up", "Research"]
                suggested_tags = random.sample(sample_tags, random.randint(2, 4))
                
                st.success(f"✅ Note created successfully!")
                st.markdown(f"**🤖 AI Predicted Category:** {predicted_category} (Confidence: {confidence})")
                st.markdown(f"**🏷️ Suggested Tags:** {', '.join(suggested_tags)}")
                
                # Show the created note
                st.markdown("---")
                st.markdown("### 📋 Your Note Preview")
                st.markdown(f"""
                <div class="note-card">
                    <h4>{title}</h4>
                    <p>{content[:200]}{'...' if len(content) > 200 else ''}</p>
                    <small>Category: {predicted_category} | Tags: {', '.join(suggested_tags)}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Please enter both title and content!")
    
    with col2:
        st.markdown("### 💡 AI Assistant")
        st.info("""
        **How AI helps:**
        - 🎯 Automatically categorizes your notes
        - 🏷️ Suggests relevant tags
        - 📝 Improves over time with your patterns
        - 🔍 Makes notes easily searchable
        """)
        
        st.markdown("### 📊 Category Distribution")
        categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
        values = [25, 15, 20, 10, 18, 12]
        
        fig = px.pie(values=values, names=categories, title="Your Notes by Category")
        st.plotly_chart(fig, use_container_width=True)

# Semantic Search Page
elif page == "🔍 Semantic Search":
    st.markdown("## 🔍 Semantic Search Demo")
    
    search_query = st.text_input("🔎 Search your notes", 
                                placeholder="Try: 'machine learning projects' or 'meeting follow-ups'")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        search_type = st.radio("Search Type:", ["🧠 Semantic", "📝 Keyword"])
        max_results = st.slider("Max Results:", 1, 10, 5)
    
    if search_query:
        st.markdown("### 🎯 Search Results")
        
        # Simulate semantic search results
        if search_type == "🧠 Semantic":
            st.info("🧠 Using AI-powered semantic search to find notes by meaning...")
            # Filter and rank notes based on content similarity (simulated)
            relevant_notes = [note for note in notes_data if any(word.lower() in note['content'].lower() 
                            for word in search_query.split())]
        else:
            st.info("📝 Using traditional keyword search...")
            relevant_notes = [note for note in notes_data if search_query.lower() in note['title'].lower() 
                            or search_query.lower() in note['content'].lower()]
        
        if relevant_notes:
            for i, note in enumerate(relevant_notes[:max_results]):
                similarity_score = round(random.uniform(0.7, 0.95), 2)
                
                with st.expander(f"📝 {note['title']} (Relevance: {similarity_score})"):
                    st.markdown(f"**Category:** {note['category']}")
                    st.markdown(f"**Content:** {note['content']}")
                    st.markdown(f"**Tags:** {', '.join(note['tags'])}")
                    st.markdown(f"**Created:** {note['created_date'].strftime('%Y-%m-%d')}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.button(f"📖 Open", key=f"open_{i}")
                    with col2:
                        st.button(f"✏️ Edit", key=f"edit_{i}")
                    with col3:
                        st.button(f"🔗 Share", key=f"share_{i}")
        else:
            st.warning("No matching notes found. Try different keywords!")

# AI Categorization Page
elif page == "🏷️ AI Categorization":
    st.markdown("## 🤖 AI Categorization System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Categorization Performance")
        
        categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
        accuracy_scores = [0.94, 0.91, 0.96, 0.88, 0.93, 0.89]
        
        fig = px.bar(x=categories, y=accuracy_scores, 
                    title="AI Accuracy by Category",
                    labels={'x': 'Category', 'y': 'Accuracy Score'},
                    color=accuracy_scores,
                    color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Overall Metrics")
        st.metric("Overall Accuracy", "92.8%", "↑ 2.3%")
        st.metric("Processing Speed", "0.3s", "↓ 0.1s")
        st.metric("Confidence Score", "89.5%", "↑ 1.2%")
    
    with col2:
        st.markdown("### 🔬 Test the AI Categorizer")
        
        test_text = st.text_area("Enter text to categorize:", 
                                height=150,
                                placeholder="Paste any text here to see how our AI categorizes it...")
        
        if st.button("🚀 Analyze Text"):
            if test_text:
                # Simulate AI prediction
                categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
                predicted_category = random.choice(categories)
                confidence = round(random.uniform(0.75, 0.98), 2)
                
                st.success(f"**Predicted Category:** {predicted_category}")
                st.info(f"**Confidence:** {confidence * 100:.1f}%")
                
                # Simulate confidence scores for all categories
                st.markdown("#### 📊 Confidence Breakdown")
                all_scores = {cat: round(random.uniform(0.1, 0.9), 2) for cat in categories}
                all_scores[predicted_category] = confidence
                
                for cat, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                    st.progress(score, text=f"{cat}: {score:.2f}")
        
        st.markdown("### 🧠 How it Works")
        st.markdown("""
        Our AI uses advanced NLP techniques:
        - **BERT Embeddings** for semantic understanding
        - **Transfer Learning** from domain-specific data
        - **Multi-label Classification** for complex notes
        - **Active Learning** to improve over time
        """)

# Analytics Dashboard
elif page == "📊 Analytics Dashboard":
    st.markdown("## 📊 Analytics Dashboard")
    
    # Generate sample analytics data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Notes", "156", "↑ 12")
        st.metric("This Week", "23", "↑ 5")
    with col2:
        st.metric("Categories Used", "8", "→ 0")
        st.metric("Avg. Note Length", "247 words", "↑ 15")
    with col3:
        st.metric("Search Queries", "89", "↑ 23")
        st.metric("Collaboration Sessions", "12", "↑ 3")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Note Creation Trend")
        
        # Create time series data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        note_counts = np.random.poisson(5, len(dates))
        
        fig = px.line(x=dates, y=note_counts, title="Daily Note Creation")
        fig.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏷️ Tag Cloud")
        tags = ["AI", "Machine Learning", "Python", "Data Science", "Research", "Work", "Ideas", 
               "Projects", "Learning", "Meeting", "Follow-up", "Important", "Review"]
        tag_freq = {tag: random.randint(5, 50) for tag in tags}
        
        # Create a simple tag display instead of wordcloud for Streamlit compatibility
        st.markdown("**Most Used Tags:**")
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        for tag, freq in sorted_tags[:10]:
            st.markdown(f"• **{tag}** ({freq} times)")
    
    with col2:
        st.markdown("### 📊 Category Distribution")
        
        categories = ["Work", "Personal", "Research", "Ideas", "Meetings", "Learning"]
        values = [35, 25, 28, 15, 30, 22]
        
        fig = px.pie(values=values, names=categories, 
                    title="Notes by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ⏰ Activity Heatmap")
        
        # Create activity heatmap data
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        activity_data = np.random.poisson(2, (7, 24))
        
        fig = px.imshow(activity_data, 
                       x=hours, y=days,
                       title="Note Creation by Time",
                       color_continuous_scale='Blues')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# Collaboration Page
elif page == "👥 Collaboration":
    st.markdown("## 👥 Collaboration Features")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🤝 Shared Notebooks")
        
        shared_notebooks = [
            {"name": "Team Project Planning", "members": 5, "notes": 23, "last_activity": "2 hours ago"},
            {"name": "Research Collaboration", "members": 3, "notes": 47, "last_activity": "1 day ago"},
            {"name": "Meeting Notes Archive", "members": 8, "notes": 156, "last_activity": "3 hours ago"}
        ]
        
        for notebook in shared_notebooks:
            with st.expander(f"📓 {notebook['name']}"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Members", notebook['members'])
                with col_b:
                    st.metric("Notes", notebook['notes'])
                with col_c:
                    st.metric("Last Activity", notebook['last_activity'])
                
                st.markdown("**Recent Activity:**")
                st.markdown("• Sarah added 'Budget Review Notes'")
                st.markdown("• Mike edited 'Project Timeline'")
                st.markdown("• Alex shared 'Research Findings'")
        
        st.markdown("### 💬 Real-time Comments")
        
        st.markdown("""
        <div class="note-card">
            <h4>Machine Learning Project Timeline</h4>
            <p>Phase 1: Data collection and preprocessing (Week 1-2)<br>
            Phase 2: Model development and training (Week 3-5)<br>
            Phase 3: Testing and validation (Week 6-7)</p>
            
            <div style="border-top: 1px solid #ddd; margin-top: 1rem; padding-top: 1rem;">
                <strong>💬 Comments:</strong><br>
                <small><strong>Sarah:</strong> Should we allocate more time for data preprocessing?</small><br>
                <small><strong>Mike:</strong> Agreed, let's extend Phase 1 by one week.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Collaboration Stats")
        
        st.metric("Active Collaborations", "3")
        st.metric("Shared Notes", "89")
        st.metric("Comments This Week", "24")
        
        st.markdown("### 👥 Team Members")
        
        team_members = [
            {"name": "Sarah Chen", "role": "Project Lead", "activity": "Active"},
            {"name": "Mike Johnson", "role": "Developer", "activity": "Active"},
            {"name": "Alex Rivera", "role": "Researcher", "activity": "Away"},
            {"name": "Emma Wilson", "role": "Designer", "activity": "Active"}
        ]
        
        for member in team_members:
            status_color = "🟢" if member['activity'] == "Active" else "🟡"
            st.markdown(f"{status_color} **{member['name']}**")
            st.markdown(f"   *{member['role']}*")
        
        st.markdown("### 🔔 Recent Notifications")
        
        notifications = [
            "Sarah commented on 'Project Timeline'",
            "New note shared in 'Research Collaboration'",
            "Mike mentioned you in 'Budget Review'",
            "Weekly summary is ready"
        ]
        
        for notification in notifications:
            st.markdown(f"• {notification}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>📝 <strong>Notely</strong> - Making note-taking intelligent and collaborative</p>
    <p>Built with ❤️ using Streamlit | AI-Powered | Privacy-First</p>
</div>
""", unsafe_allow_html=True) 