# Getting Started with Notely

This guide will help you set up and run the Notely Smart Note-Taking Assistant on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Neo4j (for knowledge graph functionality)
- Node.js and npm (for the frontend)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/UKOMAL/Komal-Shahid-DS-Portfolio.git
   cd Komal-Shahid-DS-Portfolio/projects/notely
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Neo4j:
   - Download and install Neo4j Desktop from [neo4j.com](https://neo4j.com/download/)
   - Create a new database with a password
   - Export the password as an environment variable:
     ```bash
     export NEO4J_PASSWORD=your_password  # On Windows, use: set NEO4J_PASSWORD=your_password
     ```

## Running the Application

1. Start the backend:
   ```bash
   cd src
   python notely.py
   ```

2. The API server will start on `localhost:5000` by default.

## Usage Examples

### Processing a Note

```python
from notely import NotelyApp

app = NotelyApp()
app.initialize_nlp()
app.initialize_knowledge_graph()

note = """
Meeting with the development team on Monday.
Action items:
- Finalize API documentation
- Review pull requests
- Schedule user testing for next release
"""

processed = app.process_note(note, "user123")
print(processed)
```

### Expected Output

```json
{
  "original_text": "Meeting with the development team on Monday...",
  "word_count": 23,
  "categories": ["meeting", "development"],
  "entities": ["Monday", "API documentation", "pull requests", "user testing"],
  "summary": "Meeting with the development team on Monday to discuss documentation, code review, and testing.",
  "action_items": [
    "Finalize API documentation",
    "Review pull requests",
    "Schedule user testing for next release"
  ],
  "sentiment": "neutral",
  "timestamp": "2023-11-16T12:00:00Z"
}
```

## Next Steps

- Explore the knowledge graph visualization
- Connect to the frontend interface
- Configure custom NLP models
- Set up collaborative workspaces

## Troubleshooting

- **Issue**: Neo4j connection error
  **Solution**: Ensure Neo4j is running and the password is correctly set in the environment variable

- **Issue**: NLP model loading fails
  **Solution**: Check internet connection, as models are downloaded from Hugging Face

For more help, refer to the full documentation or open an issue on the repository. 