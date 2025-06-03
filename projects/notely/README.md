# Notely - Smart Note Templates

**Course:** DSC 670 - Data Science Capstone  
**Date:** Spring 2025  
**Author:** Komal Shahid

## Project Overview

Notely is a smart note template generation application that transforms plain text notes into professionally formatted templates. The application analyzes content and recommends the most appropriate template based on the text structure and keywords.

## Features

- **Smart Analysis**: Automatically analyzes content and suggests the best template
- **Multiple Templates**: Choose from Cornell Notes, Business Reports, Project Plans, and more
- **File Support**: Upload text, PDF, or Word documents
- **Download Options**: Save templates in various formats

## Running the Application

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run src/notely_streamlit_app.py
   ```

## Project Structure

- `src/`: Contains the application source code
  - `notely_streamlit_app.py`: Main Streamlit application
  - `template_manager.py`: Template management utilities
  - `showcase_templates.py`: Template showcase examples

- `milestones/`: Project development documentation
  - `milestone1/`: Initial project proposal
  - `milestone2/`: Design and planning
  - `milestone3/`: Implementation and testing
  - `milestone4/`: Final presentation and deployment

## Technologies Used

- Streamlit for the web interface
- Python for text analysis and template generation
- PDF and document processing libraries 