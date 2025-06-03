"""
Notely - Smart Note Templates
Course: DSC 670 - Data Science Capstone
Date: Spring 2025
Author: Komal Shahid

A Streamlit application that transforms plain text notes into professionally formatted templates.
"""

import streamlit as st
import re
import os
import datetime
from typing import Dict, List, Tuple, Any, Optional, Set
import io
import base64
from tempfile import NamedTemporaryFile
import subprocess
from streamlit_pdf_viewer import pdf_viewer  # type: ignore

# App configuration
st.set_page_config(
    page_title="Notely - Smart Note Templates",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
        div.block-container {
            padding: 0 !important;
            max-width: 100% !important;
            margin: 0 auto !important;
        }
        
        .stSelectbox, .stTextArea, .stTextInput, .stButton {
            width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        
        div[data-testid="column"] {
            padding: 1rem !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        div[data-testid="stHorizontalBlock"] {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
        }
        
        .stButton > button {
            width: 100% !important;
            margin: 0 !important;
            box-sizing: border-box !important;
        }
        
        .stTextArea > div > div > textarea {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        .stSelectbox > div > div {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        .streamlit-pdf-viewer {
            width: 100% !important;
            padding: 0 !important;
            margin: 0 auto !important;
            display: flex !important;
            justify-content: center !important;
        }
        
        .stMarkdown {
            width: 100% !important;
            box-sizing: border-box !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            width: 100% !important;
            text-align: left !important;
            padding: 0 !important;
            margin-bottom: 1rem !important;
        }
        
        .main .block-container {
            padding: 1rem !important;
            max-width: 1400px !important;
            margin: 0 auto !important;
        }
        
        .preview-container {
            width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            background-color: #2c3e50 !important;
            border-radius: 8px !important;
        }
        
        .preview-header {
            background-color: #2c3e50;
            color: white;
            padding: 0 !important;
            text-align: left !important;
            width: 100% !important;
            box-sizing: border-box !important;
            margin: 0 !important;
        }
        
        .preview-header h2 {
            margin: 0 !important;
            padding: 10px 0 !important;
            font-size: 1.8em !important;
        }
        
        .gallery-header, .preview-header {
            text-align: left !important;
            padding-left: 0 !important;
        }
        
        .main-header {
            background: linear-gradient(90deg, #5f2c82, #49a09d);
            padding: 2rem 1rem;
            margin: 0 0 2rem 0;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
            overflow: visible;
        }
        
        .main-header h1 {
            font-size: 4rem !important;
            font-weight: 800 !important;
            margin: 0 auto !important;
            letter-spacing: 4px;
            color: white;
            text-transform: uppercase;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            width: auto !important;
            display: inline-block;
            line-height: 1.1;
        }
        
        .main-header p {
            font-size: 1.3rem;
            margin: 0.8rem auto 0 auto;
            color: rgba(255,255,255,0.9);
            max-width: 80%;
        }
        
        div[data-baseweb="tab-list"] {
            display: flex !important;
            background-color: #34495e !important;
            border-radius: 0 !important;
            padding: 15px 0 !important;
            justify-content: center !important;
            margin-bottom: 30px !important;
            border: 3px solid white !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 999 !important;
        }
        
        button[data-baseweb="tab"] {
            font-size: 24px !important;
            font-weight: normal !important;
            padding: 10px 40px !important;
            background-color: #3498db !important;
            color: white !important;
            margin: 0 10px !important;
            border-radius: 5px !important;
            border: 2px solid white !important;
        }
        
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #2ecc71 !important;
            transform: scale(1.05) !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 20px !important;
        }
        
        button[data-baseweb="tab"] p {
            color: white !important;
            font-weight: normal !important;
            font-size: 24px !important;
        }
        
        [role="tablist"] {
            visibility: visible !important;
            display: flex !important;
        }
    </style>
    """, unsafe_allow_html=True)

class NoteTemplateEngine:
    """The core engine that processes notes and generates templates"""
    
    def __init__(self):
        self.templates = {
            "Cornell Notes": self._cornell_template,
            "Business Report": self._business_template,
            "Project Plan": self._project_template,
            "Meeting Notes": self._meeting_template,
            "Research Paper": self._research_template,
            "Creative Brief": self._creative_template
        }
        
        self.template_descriptions = {
            "Cornell Notes": "Perfect for academic study and lecture notes with organized sections for questions, notes, and summaries.",
            "Business Report": "Professional business documentation with executive summary, metrics, and actionable insights.",
            "Project Plan": "Structured project management with deliverables, timelines, and milestone tracking.",
            "Meeting Notes": "Organized meeting documentation with attendees, discussions, and action items.",
            "Research Paper": "Academic research format with methodology, analysis, and structured findings.",
            "Creative Brief": "Creative project planning with concept development and deliverable specifications."
        }
    
    def analyze_content(self, text: str) -> Dict:
        """Analyzes the text and extracts useful patterns and metrics"""
        if not text.strip():
            return {}
        
        # Basic text analysis
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Find headers and bullet points
        headers = []
        bullet_points = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.endswith(':') or (len(line) < 50 and line.isupper()):
                headers.append(line)
            elif line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                bullet_points.append(line)
        
        # Calculate readability stats
        avg_words_per_sentence = len(words) / max(len(sentences), 1)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len(paragraphs),
            "header_count": len(headers),
            "bullet_points": len(bullet_points),
            "avg_words_per_sentence": round(avg_words_per_sentence, 1),
            "readability": self._calculate_readability(words, sentences),
            "reading_time_mins": max(1, len(words) // 200),
            "headers": headers[:5],
            "key_topics": self._extract_topics(text)
        }
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> str:
        """Simple readability estimator based on word/sentence length"""
        if len(words) < 10:
            return "Too short to analyze"
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        if avg_word_length < 4 and avg_sentence_length < 15:
            return "Easy"
        elif avg_word_length < 5 and avg_sentence_length < 20:
            return "Moderate"
        else:
            return "Complex"
    
    def _extract_topics(self, text: str) -> List[Tuple[str, int]]:
        """Finds key topics by word frequency (excluding common words)"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq: Dict[str, int] = {}
        
        # Filter out common words
        stop_words: Set[str] = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 
                   'been', 'were', 'said', 'each', 'which', 'their', 'time', 
                   'would', 'there', 'could', 'other', 'about'}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def detect_best_template(self, text: str) -> str:
        """Determines which template best fits the content"""
        text_lower = text.lower()
        
        # Define keywords for different template types
        academic_keywords = ['research', 'study', 'analysis', 'hypothesis', 
                           'methodology', 'conclusion', 'literature']
        
        business_keywords = ['meeting', 'agenda', 'action items', 'deadline', 
                           'budget', 'revenue', 'strategy', 'quarterly']
        
        project_keywords = ['task', 'milestone', 'timeline', 'deliverable', 
                          'phase', 'schedule', 'sprint', 'backlog']
        
        # Count keyword matches
        scores = {
            "Cornell Notes": sum(1 for word in academic_keywords if word in text_lower),
            "Business Report": sum(1 for word in business_keywords if word in text_lower),
            "Project Plan": sum(1 for word in project_keywords if word in text_lower)
        }
        
        # Return the best match
        best_template = max(scores, key=lambda k: scores[k])
        return best_template if scores[best_template] > 0 else "Cornell Notes"
    
    def _cornell_template(self, content: str, analysis: Dict) -> str:
        """Creates a Cornell Notes format template"""
        return f"""
# Cornell Notes

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Topic:** {analysis.get('headers', ['General Notes'])[0] if analysis.get('headers') else 'General Notes'}

## Questions/Keywords
- Important terms
- Review questions
- Key concepts

## Notes
{content}

## Summary
**Key Points:**
- Word Count: {analysis.get('word_count', 0)}
- Reading Time: ~{analysis.get('reading_time_mins', 1)} minutes
- Complexity: {analysis.get('readability', 'Moderate')}

**Main Topics:** {', '.join([topic[0] for topic in analysis.get('key_topics', [])[:3]])}
"""
    
    def _business_template(self, content: str, analysis: Dict) -> str:
        """Creates a business report template"""
        return f"""
# Business Report

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Document Type:** Business Analysis

## Executive Summary
{content[:200]}{'...' if len(content) > 200 else ''}

## Main Content
{content}

## Key Metrics
- **Total Words:** {analysis.get('word_count', 0)}
- **Reading Time:** {analysis.get('reading_time_mins', 1)} minutes
- **Complexity Level:** {analysis.get('readability', 'Moderate')}
- **Structure:** {analysis.get('paragraph_count', 0)} paragraphs, {analysis.get('sentence_count', 0)} sentences

## Action Items
- Review and validate content
- Share with stakeholders
- Schedule follow-up
"""
    
    def _project_template(self, content: str, analysis: Dict) -> str:
        """Creates a project plan template"""
        return f"""
# Project Plan

**Created:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Status:** Draft

## Project Overview
{content}

## Key Deliverables
{chr(10).join(f"- {header}" for header in analysis.get('headers', ['Define scope', 'Create timeline', 'Assign resources'])[:5])}

## Timeline & Metrics
- **Content Length:** {analysis.get('word_count', 0)} words
- **Estimated Review Time:** {analysis.get('reading_time_mins', 1)} minutes
- **Document Sections:** {analysis.get('paragraph_count', 0)}
- **Action Points:** {analysis.get('bullet_points', 0)}

## Next Steps
1. Review project requirements
2. Validate timeline and resources
3. Begin implementation
"""
    
    def _meeting_template(self, content: str, analysis: Dict) -> str:
        """Creates a meeting notes template"""
        return f"""
# Meeting Notes

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Attendees:** [To be filled]

## Meeting Content
{content}

## Key Discussion Points
{chr(10).join(f"- {topic[0].title()}" for topic in analysis.get('key_topics', [])[:5])}

## Action Items
- Follow up on discussed topics
- Share meeting notes with team
- Schedule next meeting

## Meeting Statistics
- **Content Volume:** {analysis.get('word_count', 0)} words
- **Discussion Complexity:** {analysis.get('readability', 'Moderate')}
- **Topics Covered:** {len(analysis.get('key_topics', []))}
"""
    
    def _research_template(self, content: str, analysis: Dict) -> str:
        """Creates an academic research template"""
        return f"""
# Research Document

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Type:** Academic Research

## Abstract
{content[:300]}{'...' if len(content) > 300 else ''}

## Introduction
{content}

## Research Metrics
- **Total Word Count:** {analysis.get('word_count', 0)}
- **Paragraph Structure:** {analysis.get('paragraph_count', 0)} sections
- **Readability Level:** {analysis.get('readability', 'Moderate')}
- **Key Research Terms:** {', '.join([topic[0] for topic in analysis.get('key_topics', [])[:5]])}

## Methodology
Research methodology and approach to be defined based on content analysis.

## Conclusion
Further analysis required. Document contains {analysis.get('sentence_count', 0)} sentences with an average of {analysis.get('avg_words_per_sentence', 0)} words per sentence.
"""
    
    def _creative_template(self, content: str, analysis: Dict) -> str:
        """Creates a creative brief template"""
        return f"""
# Creative Brief

**Project Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}
**Brief Type:** Creative Project

## Project Overview
{content}

## Creative Direction
**Key Themes:** {', '.join([topic[0] for topic in analysis.get('key_topics', [])[:3]])}
**Content Complexity:** {analysis.get('readability', 'Moderate')}
**Estimated Development Time:** {analysis.get('reading_time_mins', 1)} hours

## Project Specifications
- **Word Count:** {analysis.get('word_count', 0)}
- **Structure:** {analysis.get('paragraph_count', 0)} main sections
- **Key Elements:** {analysis.get('bullet_points', 0)} specific points

## Deliverables
1. Creative concept development
2. Content refinement
3. Final presentation
"""
    
    def transform_content(self, content: str, template_type: str) -> Tuple[str, Dict]:
        """Main function to transform content using the selected template"""
        analysis = self.analyze_content(content)
        
        if template_type == "Auto-detect":
            template_type = self.detect_best_template(content)
        
        template_func = self.templates.get(template_type, self._cornell_template)
        transformed = template_func(content, analysis)
        
        return transformed, analysis


class FileProcessor:
    """Handles file uploads and text extraction"""
    
    @staticmethod
    def extract_text(uploaded_file) -> str:
        """Extract text from various file formats"""
        try:
            file_type = uploaded_file.type
            file_name = uploaded_file.name.lower()
            
            # Plain text files
            if file_type == "text/plain" or any(file_name.endswith(ext) for ext in 
                                             ['.txt', '.md', '.py', '.js', '.html', '.css']):
                return str(uploaded_file.read(), "utf-8")
            
            # PDF files
            elif file_type == "application/pdf" or file_name.endswith('.pdf'):
                try:
                    import PyPDF2  # type: ignore
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n"
                    return text_content
                except ImportError:
                    return "PDF processing requires PyPDF2. Please install it or copy your text manually."
                except Exception as e:
                    return f"Error reading PDF: {str(e)}. Please try copying your text manually."
            
            # Word documents
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
                try:
                    import docx  # type: ignore
                    doc = docx.Document(io.BytesIO(uploaded_file.read()))
                    text_content = ""
                    for paragraph in doc.paragraphs:
                        text_content += paragraph.text + "\n"
                    return text_content
                except ImportError:
                    return "Word processing requires python-docx. Please install it or copy your text manually."
                except Exception as e:
                    return f"Error reading Word document: {str(e)}. Please try copying your text manually."
            
            else:
                return f"Unsupported file type: {file_type}. Please use TXT, PDF, or Word documents."
                
        except Exception as e:
            return f"Error processing file: {str(e)}. Please try copying your text manually."


class TemplateGallery:
    """Manages template showcase and examples"""
    
    @staticmethod
    def get_sample_content():
        """Sample content for each template"""
        return {
            "Cornell Notes": "Today's lecture on machine learning covered supervised vs unsupervised learning algorithms. Key concepts include data preprocessing, model training, and evaluation metrics. Neural networks use layers of interconnected nodes to process information.",
            "Research Paper": "This study examines the impact of AI on productivity in knowledge work. Recent advances in artificial intelligence have transformed how we approach complex problem-solving tasks.",
            "Creative Brief": "Design a modern visual identity for sustainable energy startup. Target audience: 25-45 year old professionals. Key elements: modern minimalist aesthetic, bold color palette, tech-forward imagery.",
            "Business Report": "Q4 results show strong growth in key metrics. Revenue increased 15% to $2.5M. User growth reached 25,000 new users. Customer satisfaction maintained at 4.8/5 stars.",
            "Project Plan": "Development of new mobile application. Key deliverables include UI/UX design by Feb 15, backend development by March 1, and testing phase by March 15.",
            "Meeting Notes": "Sprint review results discussed. Next quarter planning initiated. Resource allocation for upcoming projects reviewed. Action items: update timeline, schedule client meeting."
        }
    
    @staticmethod
    def html_to_pdf(html_file_path: str, output_pdf_path: str) -> bool:
        """Convert HTML file to PDF using wkhtmltopdf or other available methods"""
        try:
            # First, ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_pdf_path)), exist_ok=True)
            
            # Try different PDF conversion methods in order of preference
            
            # Method 1: wkhtmltopdf command line tool
            try:
                cmd = ['wkhtmltopdf', '--enable-local-file-access', html_file_path, output_pdf_path]
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if os.path.exists(output_pdf_path) and os.path.getsize(output_pdf_path) > 0:
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            # Method 2: pdfkit (Python wrapper for wkhtmltopdf)
            try:
                import pdfkit  # type: ignore
                options = {
                    'enable-local-file-access': None
                }
                pdfkit.from_file(html_file_path, output_pdf_path, options=options)
                if os.path.exists(output_pdf_path) and os.path.getsize(output_pdf_path) > 0:
                    return True
            except ImportError:
                pass
            except Exception:
                pass
            
            # Method 3: weasyprint as another alternative
            try:
                from weasyprint import HTML  # type: ignore
                HTML(html_file_path).write_pdf(output_pdf_path)
                if os.path.exists(output_pdf_path) and os.path.getsize(output_pdf_path) > 0:
                    return True
            except ImportError:
                pass
            except Exception:
                pass
            
            # If we get here, all methods failed
            return False
            
        except Exception:
            return False
    
    @staticmethod
    def display_pdf(file_path: str) -> bool:
        """Display a PDF file in the Streamlit app using streamlit-pdf-viewer component"""
        try:
            # Check if file exists and has content
            if not os.path.exists(file_path):
                st.error(f"PDF file not found: {file_path}")
                return False
                
            if os.path.getsize(file_path) == 0:
                st.error(f"PDF file is empty: {file_path}")
                return False
            
            # Read the PDF file
            with open(file_path, "rb") as f:
                pdf_data = f.read()
            
            # Use the streamlit-pdf-viewer component to display the PDF at full width
            pdf_viewer(
                input=pdf_data,
                width=1000,  # Increased width to better fill container boundary
                height=800,  # Tall enough for comfortable viewing
                key=f"pdf-{os.path.basename(file_path)}"  # Unique key for each PDF
            )
            
            # Add download button for PDF
            st.download_button(
                label=f"Download {file_path.split('/')[-1]}",
                data=pdf_data,
                file_name=f"{file_path.split('/')[-1].split('.')[0]}_template.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            return True
        except Exception as e:
            st.error(f"Error displaying PDF: {str(e)}")
            return False
    
    @staticmethod
    def display_html(file_path: str) -> bool:
        """Display HTML file content directly in the app"""
        try:
            with open(file_path, "r") as f:
                html_content = f.read()
            
            # Display HTML content
            st.components.v1.html(html_content, height=600, scrolling=True)
            return True
        except Exception as e:
            st.error(f"Error displaying HTML: {str(e)}")
            return False
    
    @staticmethod
    def prepare_template_preview(template_name: str) -> Tuple[str, bool]:
        """Prepare and return the path to the template preview"""
        # Map template names to file names
        html_mapping = {
            "Cornell Notes": "ai_cornell.html",
            "Business Report": "project_notes.html",
            "Project Plan": "climate_infographic.html",
            "Meeting Notes": "fashion_infographic.html",
            "Research Paper": "history_academic.html",
            "Creative Brief": "nutrition_stepbystep.html"
        }
        
        # Use the absolute path to showcase directory
        showcase_dir = "/Users/komalshahid/Desktop/Bellevue University/DSC670/term_project/showcase"
        
        # If template name is in our mapping
        if template_name in html_mapping:
            html_file = html_mapping[template_name]
            
            # Define PDF output path
            pdf_file = f"{template_name.lower().replace(' ', '_')}.pdf"
            pdf_path = os.path.join(showcase_dir, pdf_file)
            
            # Check if PDF exists
            if os.path.exists(pdf_path):
                return pdf_path, True
            else:
                return "", False
        
        # Template name not in mapping, return empty string and False
        return "", False


def main():
    """Main application function"""
    load_css()
    
    # Header with title and subtitle
    st.markdown("""
    <div class="main-header">
        <h1>NOTELY</h1>
        <p>Transform your notes into professional templates</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple tab navigation
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "⚡ Transform", "📋 Templates"])
    
    with tab1:
        home_page()
    
    with tab2:
        transform_page()
    
    with tab3:
        templates_page()

def home_page():
    """Display the home page content"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome to Notely!")
        st.markdown("""
        Transform your plain text notes into professionally formatted templates with just a few clicks.
        
        ### ✨ Key Features
        - **Smart Analysis**: Automatically detects the best template for your content
        - **Multiple Templates**: Choose from 6 professionally designed formats
        - **File Support**: Upload TXT, PDF, or Word documents
        - **Instant Preview**: See your transformed content immediately
        - **Easy Export**: Download in Markdown or PDF format
        
        ### 🚀 How It Works
        1. **Upload or Paste** your content in the Transform tab
        2. **Choose a Template** or let our AI auto-detect the best one
        3. **Review & Download** your professionally formatted document
        """)
    
    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Templates Available", "6")
        st.metric("File Formats Supported", "3")
        st.metric("Processing Time", "< 5 sec")
        
        st.markdown("---")
        
        # Target audience
        st.markdown("### 🎯 Best For")
        st.markdown("""
        - Students & Researchers
        - Business Professionals  
        - Project Managers
        - Creative Teams
        """)

def transform_page():
    """Display the transform page content"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Input Content")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a file",
            type=['txt', 'pdf', 'docx', 'md'],
            help="Drag & drop or click to upload"
        )
        
        # Handle file upload
        default_content = ""
        if uploaded_file is not None:
            with st.spinner("Extracting text..."):
                extracted_text = FileProcessor.extract_text(uploaded_file)
                if extracted_text and not extracted_text.startswith("Error"):
                    default_content = extracted_text
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                else:
                    st.error("❌ Could not process file")
        
        # Check if template was selected from gallery
        if 'example_content' in st.session_state and not default_content:
            default_content = st.session_state.example_content
            # Clear the session state after using it
            del st.session_state.example_content
        
        # Text input
        content = st.text_area(
            "Or paste your content here:",
            height=200,
            value=default_content,
            placeholder="Enter your notes, meeting minutes, research content, or any text you'd like to format..."
        )
        
        if content:
            words = len(content.split())
            chars = len(content)
            st.caption(f"📈 {words} words • {chars} characters")
        
        # Template selection
        st.markdown("### 🎨 Choose Template")
        template_options = [
            "Auto-detect",
            "Cornell Notes",
            "Business Report", 
            "Project Plan",
            "Meeting Notes",
            "Research Paper",
            "Creative Brief"
        ]
        
        # Pre-select template if coming from gallery
        default_template_index = 0
        if 'selected_template' in st.session_state:
            try:
                default_template_index = template_options.index(st.session_state.selected_template)
                # Clear the session state after using it
                del st.session_state.selected_template
            except ValueError:
                default_template_index = 0
        
        template_style = st.selectbox(
            "Template Type:",
            template_options,
            index=default_template_index,
            help="Auto-detect analyzes your content and picks the best template"
        )
        
        # Transform button
        transform_clicked = st.button("🚀 TRANSFORM", type="primary", use_container_width=True)
        
        # Only transform when button is explicitly clicked
        if transform_clicked:
            if content.strip():
                with st.spinner("Creating your template..."):
                    engine = NoteTemplateEngine()
                    transformed, analysis = engine.transform_content(content, template_style)
                    st.session_state.transformed = transformed
                    st.session_state.analysis = analysis
                    st.session_state.template_used = template_style if template_style != "Auto-detect" else engine.detect_best_template(content)
                    st.session_state.original_content = content
                st.success("✨ Transformation complete!")
            else:
                st.warning("⚠️ Please enter some content first")
    
    with col2:
        st.markdown("### 📋 Output")
        
        # Show content analysis if we have content
        if content and content.strip():
            engine = NoteTemplateEngine()
            analysis = engine.analyze_content(content)
            
            # Show metrics in a clean container
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Words", analysis.get('word_count', 0))
            with col_b:
                st.metric("Reading Time", f"{analysis.get('reading_time_mins', 1)}m")
            with col_c:
                st.metric("Complexity", analysis.get('readability', 'Moderate'))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show key topics
            if analysis.get('key_topics'):
                topics = [topic[0] for topic in analysis.get('key_topics', [])[:3]]
                st.info(f"**🏷️ Key Topics:** {', '.join(topics)}")
            
            # Show suggested template
            suggested_template = engine.detect_best_template(content)
            st.info(f"**💡 Suggested Template:** {suggested_template}")
        
        # Show transformed template only if transformation was completed
        if 'transformed' in st.session_state and st.session_state.get('transformed'):
            # Template used
            if 'template_used' in st.session_state:
                st.success(f"**📋 Template:** {st.session_state.template_used}")
            
            # Show transformed content as HTML infographic
            st.markdown("**Your Template:**")
            
            # Generate HTML infographic preview
            with st.spinner("Generating template preview..."):
                try:
                    # Convert markdown to proper HTML with infographic styling
                    transformed_text = st.session_state.transformed
                    
                    # Process markdown formatting
                    html_lines = []
                    lines = transformed_text.split('\n')
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('# '):
                            html_lines.append(f'<h1>{line[2:]}</h1>')
                        elif line.startswith('## '):
                            html_lines.append(f'<h2>{line[3:]}</h2>')
                        elif line.startswith('### '):
                            html_lines.append(f'<h3>{line[4:]}</h3>')
                        elif line.startswith('**') and line.endswith('**'):
                            html_lines.append(f'<p><strong>{line[2:-2]}</strong></p>')
                        elif line.startswith('- '):
                            html_lines.append(f'<li>{line[2:]}</li>')
                        elif line:
                            # Handle bold text within lines
                            line = line.replace('**', '<strong>').replace('**', '</strong>')
                            html_lines.append(f'<p>{line}</p>')
                        else:
                            html_lines.append('<br>')
                    
                    # Create full HTML with infographic styling
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <style>
                            body {{ 
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                                line-height: 1.8; 
                                margin: 0;
                                padding: 30px;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                min-height: 100vh;
                            }}
                            .container {{
                                background: white;
                                border-radius: 15px;
                                padding: 40px;
                                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                                max-width: 1000px;
                                margin: 0 auto;
                            }}
                            h1 {{ 
                                color: #2c3e50; 
                                border-bottom: 4px solid #3498db; 
                                padding-bottom: 15px; 
                                font-size: 2.5em;
                                margin-bottom: 30px;
                                text-align: center;
                            }}
                            h2 {{ 
                                color: #34495e; 
                                border-bottom: 2px solid #bdc3c7; 
                                padding-bottom: 10px; 
                                font-size: 1.8em;
                                margin-top: 30px;
                                margin-bottom: 20px;
                            }}
                            h3 {{ 
                                color: #7f8c8d; 
                                font-size: 1.4em;
                                margin-top: 25px;
                                margin-bottom: 15px;
                            }}
                            p {{ 
                                color: #2c3e50;
                                font-size: 1.1em;
                                margin-bottom: 15px;
                            }}
                            strong {{ 
                                color: #e74c3c; 
                                font-weight: 600;
                            }}
                            li {{
                                color: #2c3e50;
                                font-size: 1.1em;
                                margin-bottom: 8px;
                                padding-left: 10px;
                            }}
                            ul {{
                                background: #f8f9fa;
                                padding: 20px;
                                border-radius: 8px;
                                border-left: 5px solid #3498db;
                                margin: 20px 0;
                            }}
                            .header-info {{
                                background: linear-gradient(90deg, #3498db, #2ecc71);
                                color: white;
                                padding: 20px;
                                border-radius: 10px;
                                margin-bottom: 30px;
                                text-align: center;
                            }}
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header-info">
                                <h4 style="margin:0; color: white;">Generated by NOTELY - Professional Template Engine</h4>
                            </div>
                            {''.join(html_lines)}
                        </div>
                    </body>
                    </html>
                    """
                    
                    # Display as wide HTML infographic
                    st.components.v1.html(html_content, height=700, scrolling=True)
                    
                except Exception as e:
                    st.error(f"Could not generate template preview: {str(e)}")
                    # Fallback to simple display
                    st.code(st.session_state.transformed, language="markdown")
            
            # Download options
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥 Download Template",
                    data=st.session_state.transformed,
                    file_name=f"notely_template_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            
            with col_d2:
                if 'original_content' in st.session_state:
                    st.download_button(
                        label="📄 Download Original",
                        data=st.session_state.original_content,
                        file_name=f"original_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        elif content and content.strip():
            # Show message to transform when content is available
            st.info("👈 Click 'TRANSFORM' to generate your formatted template!")
        else:
            st.info("👈 Enter content and hit Transform to see your formatted template here!")
            
            st.markdown("#### Supported Files")
            st.markdown("""
            - 📄 **Text:** .txt, .md  
            - 📑 **PDF:** .pdf
            - 📝 **Word:** .docx
            """)

def templates_page():
    """Display the template gallery page"""
    st.markdown("### 📋 Template Gallery")
    st.markdown("Choose from our collection of professional templates:")
    
    # Initialize template engine to get descriptions
    engine = NoteTemplateEngine()
    templates = list(engine.templates.keys())
    
    # Sample content for each template
    samples = {
        "Cornell Notes": "Today's lecture covered machine learning fundamentals including supervised vs unsupervised learning. Key concepts: data preprocessing, model training, evaluation metrics. Neural networks use interconnected nodes to process information through multiple layers.",
        "Business Report": "Q4 performance analysis shows 15% revenue growth to $2.5M with 25,000 new users acquired. Customer satisfaction maintained at 4.8/5 stars. Key metrics indicate strong market position and user engagement trends.",
        "Project Plan": "Mobile app development project spanning 3 months. Key deliverables include UI/UX design completion by Feb 15, backend development by March 1, testing phase by March 15, and final deployment by March 30.",
        "Meeting Notes": "Sprint review meeting discussed quarterly planning, resource allocation, and upcoming project timelines. Team reviewed current progress, identified blockers, and established action items for next iteration.",
        "Research Paper": "This study examines artificial intelligence impact on workplace productivity in knowledge-based industries. Recent advances in AI technology have transformed approaches to complex problem-solving tasks and decision-making processes.",
        "Creative Brief": "Brand identity design for sustainable energy startup targeting 25-45 year old professionals. Design requirements include modern minimalist aesthetic, bold color palette, tech-forward imagery, and environmental consciousness messaging."
    }
    
    # PDF file mapping
    pdf_files = {
        "Cornell Notes": "cornell_notes.pdf",
        "Business Report": "business_report.pdf", 
        "Project Plan": "project_plan.pdf",
        "Meeting Notes": "meeting_notes.pdf",
        "Research Paper": "research_paper.pdf",
        "Creative Brief": "creative_brief.pdf"
    }
    
    # Create template cards in a 2x3 grid
    for i in range(0, len(templates), 2):
        col1, col2 = st.columns(2)
        
        # First template in row
        with col1:
            template = templates[i]
            create_template_card(template, engine.template_descriptions[template], 
                               samples[template], pdf_files.get(template))
        
        # Second template in row (if exists)
        if i + 1 < len(templates):
            with col2:
                template = templates[i + 1]
                create_template_card(template, engine.template_descriptions[template],
                                   samples[template], pdf_files.get(template))

def create_template_card(template_name: str, description: str, sample_content: str, pdf_file: str):
    """Create a template card with preview"""
    
    # Card container
    st.markdown(f"""
    <div class="template-card">
        <div class="template-title">{template_name}</div>
        <div class="template-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample content
    with st.expander(f"📖 View {template_name} Sample"):
        st.text_area(f"Sample content for {template_name}:", 
                    value=sample_content, height=100, 
                    key=f"sample_{template_name}", disabled=True)
    
    # PDF preview if available
    if pdf_file:
        # Use absolute path to showcase directory
        showcase_dir = "/Users/komalshahid/Desktop/Bellevue University/DSC670/term_project/showcase"
        pdf_path = os.path.join(showcase_dir, pdf_file)
        
        if os.path.exists(pdf_path):
            with st.expander(f"👁️ Preview {template_name} Template"):
                display_pdf_preview(pdf_path, template_name)
        else:
            st.warning(f"Preview not available for {template_name} (path: {pdf_path})")
    
    # Use template button
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(f"✨ Use Template", key=f"use_{template_name}", use_container_width=True):
            # Set session state for the Transform tab
            st.session_state.selected_template = template_name
            st.session_state.example_content = sample_content
            st.success(f"✅ {template_name} loaded! Switch to Transform tab to continue.")
            st.info("👆 Click the 'Transform' tab above to use this template")
    
    with col_btn2:
        if st.button(f"📋 Copy Sample", key=f"copy_{template_name}", use_container_width=True):
            st.session_state.example_content = sample_content
            st.success("✅ Sample content copied! Switch to Transform tab.")
            st.info("👆 Click the 'Transform' tab above to see the content")
    
    st.markdown("---")

def display_pdf_preview(pdf_path: str, template_name: str):
    """Display PDF infographic templates with proper error handling"""
    try:
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
                base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            
            # Display PDF using iframe with wide styling for infographics
            pdf_display = f"""
            <div style="width: 100%; height: 800px; border: 2px solid #34495e; border-radius: 10px; overflow: hidden; background: #f8f9fa;">
                <iframe 
                    src="data:application/pdf;base64,{base64_pdf}#view=FitH" 
                    width="100%" 
                    height="100%" 
                    style="border: none; display: block; background: white;"
                    type="application/pdf"
                    title="{template_name} Template Preview">
                    <p style="padding: 20px; text-align: center;">
                        Your browser does not support PDF preview. 
                        <a href="data:application/pdf;base64,{base64_pdf}" 
                           download="{template_name.lower().replace(' ', '_')}_template.pdf"
                           style="color: #3498db; text-decoration: underline;">
                           Click here to download {template_name} template
                        </a>
                    </p>
                </iframe>
            </div>
            <div style="text-align: center; margin-top: 10px; color: #7f8c8d;">
                <small>🖱️ Use your browser's zoom controls (Ctrl/Cmd + or -) for better viewing</small>
            </div>
            """
            
            st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Download button with better styling
            st.download_button(
                label=f"📥 Download {template_name} Infographic",
                data=pdf_data,
                file_name=f"{template_name.lower().replace(' ', '_')}_infographic.pdf",
                mime="application/pdf",
                use_container_width=True,
                help=f"Download the {template_name} template as PDF"
            )
            return True
        else:
            st.error(f"📄 Template not found: {pdf_path}")
            st.info("💡 Check if the showcase directory contains the PDF files")
            return False
    except Exception as e:
        st.error(f"❌ Error displaying PDF: {str(e)}")
        # Fallback: try direct file link
        try:
            with open(pdf_path, "rb") as f:
                pdf_data = f.read()
            st.download_button(
                label=f"📥 Download {template_name} (Preview Failed)",
                data=pdf_data,
                file_name=f"{template_name.lower().replace(' ', '_')}_template.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except:
            st.error("❌ Could not load PDF file")
        return False

if __name__ == "__main__":
    main()