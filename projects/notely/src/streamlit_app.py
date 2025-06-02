"""
Notely Streamlit App
-------------------
A Streamlit-based frontend for the Notely smart note-taking assistant.
This provides a web-based interface for interacting with the Notely system.

Author: Komal Shahid
Course: DSC680 - Bellevue University
Date: June 1, 2025
"""

import streamlit as st
import pandas as pd
import datetime
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import notely
sys.path.append(str(Path(__file__).parent.absolute()))
from notely import NotelyApp

# Initialize the app
@st.cache_resource
def get_notely_app():
    app = NotelyApp()
    app.initialize_nlp()
    app.initialize_knowledge_graph()
    return app

# Set up the Streamlit page
st.set_page_config(
    page_title="Notely - Smart Note Assistant",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a header
st.title("📝 Notely: Smart Note-Taking Assistant")
st.markdown("Transform your notes with AI-powered organization and insights")

# Initialize the app
notely_app = get_notely_app()

# Sidebar for options
st.sidebar.title("Notely Options")
user_id = st.sidebar.text_input("User ID", value="user123")
note_category = st.sidebar.selectbox(
    "Note Category",
    options=["Personal", "Work", "Study", "Project", "Meeting", "Other"]
)

# Create tabs for different functions
tabs = st.tabs(["Create Note", "View Notes", "Search", "Knowledge Graph"])

# Tab 1: Create Note
with tabs[0]:
    st.header("Create a New Note")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        note_title = st.text_input("Note Title", placeholder="Enter a title for your note")
        note_text = st.text_area("Note Content", height=300, placeholder="Type your note here...")
        
        col_submit, col_clear = st.columns([1, 3])
        with col_submit:
            submit_button = st.button("Process Note", type="primary", use_container_width=True)
        with col_clear:
            clear_button = st.button("Clear", use_container_width=False)
    
    with col2:
        st.markdown("### How It Works")
        st.markdown("Notely analyzes your notes using AI to:")
        st.markdown("- 🏷️ Categorize content")
        st.markdown("- 🔍 Extract key insights")
        st.markdown("- ✅ Identify action items")
        st.markdown("- 🔄 Connect related concepts")
        st.markdown("- 📊 Analyze sentiment")
    
    # Process note when button is clicked
    if submit_button and note_text:
        with st.spinner("Processing your note..."):
            # Process the note
            processed_note = notely_app.process_note(note_text, user_id)
            
            # Display results
            st.success("Note processed successfully!")
            
            # Show processed data
            st.markdown("### Note Analysis")
            
            col_meta, col_insights = st.columns(2)
            
            with col_meta:
                st.markdown("#### Metadata")
                st.markdown(f"**Words:** {processed_note['word_count']}")
                st.markdown(f"**Categories:** {', '.join(processed_note['categories'])}")
                st.markdown(f"**Sentiment:** {processed_note['sentiment'].capitalize()}")
                
            with col_insights:
                st.markdown("#### Key Insights")
                st.markdown(f"**Summary:** {processed_note['summary']}")
                
                if processed_note['action_items']:
                    st.markdown("**Action Items:**")
                    for item in processed_note['action_items']:
                        st.markdown(f"- {item}")
                else:
                    st.markdown("**Action Items:** None detected")
    
    if clear_button:
        st.experimental_rerun()

# Tab 2: View Notes
with tabs[1]:
    st.header("Your Notes")
    
    # Example notes data (in a real app, this would come from a database)
    example_notes = [
        {
            "title": "Team Meeting Notes",
            "date": "2025-06-01",
            "category": "Work",
            "summary": "Discussed Q4 roadmap and budget approvals.",
            "action_items": ["Prepare slides", "Follow up with John"]
        },
        {
            "title": "Research Ideas",
            "date": "2025-05-28",
            "category": "Study",
            "summary": "Collected ideas for the upcoming research paper on ML.",
            "action_items": ["Review literature", "Draft outline"]
        },
        {
            "title": "Project Kickoff",
            "date": "2025-05-25",
            "category": "Project",
            "summary": "Initial planning for the new ML project.",
            "action_items": ["Set up repository", "Schedule team meeting"]
        }
    ]
    
    # Convert to DataFrame for display
    df = pd.DataFrame(example_notes)
    
    # Show data
    st.dataframe(df, use_container_width=True)
    
    # Note details
    st.markdown("### Note Details")
    selected_note = st.selectbox("Select a note to view details", options=[note["title"] for note in example_notes])
    
    # Find selected note
    note_details = next((note for note in example_notes if note["title"] == selected_note), None)
    
    if note_details:
        st.markdown(f"**Title:** {note_details['title']}")
        st.markdown(f"**Date:** {note_details['date']}")
        st.markdown(f"**Category:** {note_details['category']}")
        st.markdown(f"**Summary:** {note_details['summary']}")
        
        st.markdown("**Action Items:**")
        for item in note_details['action_items']:
            st.markdown(f"- {item}")

# Tab 3: Search
with tabs[2]:
    st.header("Semantic Search")
    
    search_query = st.text_input("Search your notes", placeholder="Enter your search query...")
    search_button = st.button("Search", type="primary")
    
    if search_button and search_query:
        st.markdown(f"Searching for: **{search_query}**")
        
        # In a real app, this would perform actual semantic search
        st.info("This is a placeholder for the semantic search functionality.")
        
        # Show example results
        st.markdown("### Search Results")
        st.markdown("**Team Meeting Notes** (87% relevance)")
        st.markdown("...discussed the **budget approvals** for Q4 roadmap...")
        
        st.markdown("**Project Kickoff** (65% relevance)")
        st.markdown("...planning for the **new ML project** which requires...")

# Tab 4: Knowledge Graph
with tabs[3]:
    st.header("Knowledge Graph")
    
    st.info("This tab would display the knowledge graph visualization, connecting concepts across your notes.")
    
    # Placeholder for knowledge graph visualization
    st.markdown("### Concept Connections")
    st.image("https://miro.medium.com/max/1400/1*iyybIojBeXHDOLRVFJHZLg.png", caption="Example Knowledge Graph Visualization")

# Footer
st.markdown("---")
st.markdown("Notely - Smart Note-Taking Assistant © 2025") 