# AI Academic Assistant

A production-ready academic research helper built with **Streamlit**, **LangChain**, **Google Gemini API**, **FAISS**, and **PyMuPDF**.

## Architecture & Layout

The project follows a clean, modular architecture:

```
ai-academic-assistant/
├── .env.example              # Env template for configuration
├── .gitignore                # Git exclusions (data, cache, .env)
├── requirements.txt          # Python dependencies
├── README.md                 # Project guide
├── data/
│   ├── uploads/              # Local storage for uploaded PDF files
│   └── vector_store/         # Persisted FAISS index files
└── src/
    ├── main.py               # Streamlit application entrypoint
    ├── config.py             # App configuration validation (via Pydantic)
    ├── core/                 # Shared business logic and interfaces
    ├── prompts/              # Prompt templates & LLM instructions
    ├── services/             # Integrations (PDF loader, embeddings, vector store, LLM)
    ├── ui/                   # Reusable UI widgets and layout views
    └── utils/                # Logging and validation helpers
```

## Tech Stack Details

- **Streamlit**: Web interface for interactive paper uploading, querying, and chat.
- **LangChain**: Chains and components orchestrating RAG (Retrieval-Augmented Generation).
- **Google Gemini API**: Advanced LLM (`gemini-1.5-flash` or `gemini-1.5-pro`) and embedding models (`text-embedding-004`).
- **FAISS**: Local vector database for semantic search across paper text chunks.
- **PyMuPDF**: Highly-efficient PDF parsing library to extract raw text and metadata.

## Setup Instructions

### 1. Prerequisites
- Python 3.12+ installed.
- A Google Gemini API Key (get one from [Google AI Studio](https://aistudio.google.com/)).

### 2. Installation
Clone or navigate to the repository, create a virtual environment, and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API key:
```bash
copy .env.example .env
```
Edit `.env` and set `GEMINI_API_KEY`:
```ini
GEMINI_API_KEY=your_actual_api_key_here
```

### 4. Running the App
Run the Streamlit application:
```bash
streamlit run src/main.py
```
This will open the application in your default web browser at `http://localhost:8501`.
