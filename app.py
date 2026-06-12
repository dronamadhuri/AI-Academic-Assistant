import streamlit as st
import io
import os
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

from src.config import get_settings
from modules.vector_store import get_embeddings_model
from src.services.pdf_service import PDFService
from src.services.vector_service import VectorService
from src.services.summary_service import SummaryService, SummaryServiceError
from src.services.assignment_service import AssignmentService, AssignmentServiceError
from src.services.notes_service import NotesService, NotesServiceError
from src.services.viva_service import VivaService, VivaServiceError
from src.services.mcq_service import MCQService, MCQServiceError
from src.services.ppt_service import PPTService, PPTServiceError
from src.services.chat_service import ChatService, ChatServiceError

# New services
from modules.auth_service import initialize_db, register_user, authenticate_user
from src.services.citation_service import CitationService
from modules.export_service import export_to_docx, export_to_pdf, export_to_pptx
from src.prompts.templates import RESEARCH_MODE_MODIFIER, MERMAID_DIAGRAM_PROMPT
from modules.llm_service import GeminiClient

# Initialize global configuration settings
settings = get_settings()

def get_api_key() -> Optional[str]:
    """Retrieve Gemini API key from session state or environment settings."""
    if "api_key_override" in st.session_state and st.session_state.api_key_override.strip():
        return st.session_state.api_key_override.strip()
    return settings.gemini_api_key

def initialize_session_state() -> None:
    """Initialize state management for authentication, PDF data, FAISS indices, chat history, and generated outputs."""
    # Authenticated User info
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""
        
    # PDF storage and search index
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "pdf_metadata_list" not in st.session_state:
        st.session_state.pdf_metadata_list = []  # List of dicts representing each processed file
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    
    # Material caches
    if "summary_cache" not in st.session_state:
        st.session_state.summary_cache = {}  # Keyed by active doc
    if "assignment_cache" not in st.session_state:
        st.session_state.assignment_cache = {}  # Keyed by (active_doc, page_length)
    if "notes_cache" not in st.session_state:
        st.session_state.notes_cache = {}  # Keyed by active doc
    if "viva_cache" not in st.session_state:
        st.session_state.viva_cache = {}  # Keyed by active doc
    if "mcq_cache" not in st.session_state:
        st.session_state.mcq_cache = {}  # Keyed by active doc
    if "ppt_cache" not in st.session_state:
        st.session_state.ppt_cache = {}  # Keyed by active doc
    if "diagram_cache" not in st.session_state:
        st.session_state.diagram_cache = {}  # Keyed by active doc
        
    # Interactive MCQ user choices
    if "mcq_user_answers" not in st.session_state:
        st.session_state.mcq_user_answers = {}  # Keyed by (active_doc, question_idx)

def reset_session_state() -> None:
    """Wipe out all document details, indexing states, and cached generations."""
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.session_state.pdf_metadata_list = []
    st.session_state.pdf_uploaded = False
    st.session_state.summary_cache = {}
    st.session_state.assignment_cache = {}
    st.session_state.notes_cache = {}
    st.session_state.viva_cache = {}
    st.session_state.mcq_cache = {}
    st.session_state.ppt_cache = {}
    st.session_state.diagram_cache = {}
    st.session_state.mcq_user_answers = {}

def apply_premium_styles() -> None:
    """Inject custom CSS to apply high-end visual design, dark settings, responsive fonts, and layout enhancements."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap" rel="stylesheet">
        
        <style>
            /* Apply custom typography */
            html, body, .stApp, .stMarkdown, .stText, .stButton, p, h1, h2, h3, h4, h5, h6, li, label {
                font-family: 'Outfit', sans-serif !important;
            }
            
            /* Title styling */
            .academic-header {
                font-family: 'Playfair Display', serif !important;
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }
            
            .academic-subtitle {
                font-size: 1.1rem;
                color: #9ca3af;
                margin-bottom: 2rem;
                font-weight: 300;
            }
            
            /* Premium card design (glassmorphism look) */
            .premium-card {
                background-color: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.15);
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            .premium-card:hover {
                border-color: rgba(99, 102, 241, 0.4);
            }
            
            /* Section labels */
            .section-label {
                font-weight: 600;
                color: #a855f7;
                font-size: 0.95rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 0.5rem;
            }
            
            /* Interactive presentation slide preview canvas */
            .ppt-slide-canvas {
                background: #0f172a;
                border: 2px solid #334155;
                border-radius: 10px;
                padding: 2rem;
                margin: 1rem 0;
                min-height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
            }
            .ppt-slide-title {
                font-size: 1.6rem;
                font-weight: 700;
                color: #e2e8f0;
                border-bottom: 1px solid #1e293b;
                padding-bottom: 0.8rem;
                margin-bottom: 1rem;
            }
            .ppt-slide-bullet {
                font-size: 1.1rem;
                color: #cbd5e1;
                margin-bottom: 0.5rem;
                line-height: 1.5;
            }
            .ppt-slide-footer {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: #64748b;
                border-top: 1px solid #1e293b;
                padding-top: 0.5rem;
                margin-top: 1.5rem;
            }
            
            /* Custom styles */
            .stAlert {
                border-radius: 10px !important;
                border: 1px solid rgba(255,255,255,0.05) !important;
            }
            
            .stButton>button {
                border-radius: 8px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            
            .mcq-block {
                padding: 1.2rem;
                border-left: 3px solid #6366f1;
                background-color: rgba(99, 102, 241, 0.04);
                margin-bottom: 1rem;
                border-radius: 0 8px 8px 0;
            }
            
            /* Custom header styles for export buttons placement */
            .export-bar {
                display: flex;
                gap: 10px;
                margin-bottom: 1.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_mermaid(code: str) -> None:
    """Renders Mermaid.js diagram markup dynamically inside a browser iframe component."""
    # Strip markdown markers if present
    clean_code = code.replace("```mermaid", "").replace("```", "").strip()
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                background-color: transparent;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                overflow: hidden;
            }}
            .mermaid {{
                background-color: transparent !important;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
            {clean_code}
        </div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                securityLevel: 'loose',
                themeVariables: {{
                    background: 'transparent',
                    primaryColor: '#6366f1',
                    primaryTextColor: '#f8fafc',
                    lineColor: '#a855f7'
                }}
            }});
        </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=450, scrolling=True)

def render_voice_button(text: str, msg_idx: int) -> None:
    """Renders an inline HTML audio speaker button that reads text using browser Web Speech API."""
    # Clean text from quotes or formatting characters to ensure safe JavaScript syntax injection
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    html_code = f"""
    <button onclick="speak_{msg_idx}()" style="
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 4px 10px;
        border-radius: 6px;
        cursor: pointer;
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        margin-top: 4px;
        transition: background 0.2s ease;
    " onmouseover="this.style.background='rgba(99, 102, 241, 0.25)'" onmouseout="this.style.background='rgba(99, 102, 241, 0.15)'">
        🔊 Read Aloud
    </button>
    <script>
        function speak_{msg_idx}() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance("{safe_text}");
            msg.rate = 1.05;
            window.speechSynthesis.speak(msg);
        }}
    </script>
    """
    st.components.v1.html(html_code, height=36)

def handle_multi_pdf_upload(uploaded_files: List[Any], api_key: str) -> None:
    """Processes newly uploaded PDF files, generates individual FAISS stores, and merges them into the global database."""
    if not uploaded_files:
        return
        
    embeddings = get_embeddings_model(api_key=api_key, model_name=settings.embedding_model)
    vector_service = VectorService(embeddings=embeddings)
    pdf_service = PDFService(chunk_size=1000, chunk_overlap=200)
    citation_service = CitationService(api_key=api_key)
    
    new_uploads_occurred = False
    
    for f in uploaded_files:
        # Check if file name has already been processed in the current session
        already_indexed = any(meta["name"] == f.name for meta in st.session_state.pdf_metadata_list)
        if already_indexed:
            continue
            
        new_uploads_occurred = True
        with st.spinner(f"Processing and indexing '{f.name}'..."):
            try:
                pdf_bytes = f.read()
                pdf_stream = io.BytesIO(pdf_bytes)
                
                # Extract text
                raw_text = pdf_service.extract_text_from_pdf(pdf_stream)
                if not raw_text.strip():
                    st.sidebar.warning(f"Skipping empty or scanned PDF: '{f.name}'")
                    continue
                    
                # Create chunks
                pdf_stream.seek(0)
                chunks = pdf_service.load_and_split_pdf(pdf_stream, source_name=f.name)
                
                # Create vector database for this document
                temp_db = vector_service.create_vector_store(chunks)
                
                # Infer citations
                fallback_headers = chunks[0].metadata if chunks else {}
                extracted_meta = citation_service.extract_metadata(raw_text, f.name, fallback_headers)
                citations = citation_service.generate_citations(extracted_meta)
                
                # Count pages
                total_pages = chunks[0].metadata.get("total_pages", 1) if chunks else 1
                
                # Merge into the global vector store
                if st.session_state.vector_store is None:
                    st.session_state.vector_store = temp_db
                else:
                    st.session_state.vector_store = vector_service.merge_vector_stores(
                        st.session_state.vector_store, temp_db
                    )
                    
                # Cache metadata
                st.session_state.pdf_metadata_list.append({
                    "name": f.name,
                    "text": raw_text,
                    "pages": total_pages,
                    "size": len(pdf_bytes),
                    "meta": extracted_meta,
                    "citations": citations
                })
                
            except Exception as e:
                st.sidebar.error(f"Failed to index '{f.name}': {str(e)}")
                
    if new_uploads_occurred:
        st.session_state.pdf_uploaded = len(st.session_state.pdf_metadata_list) > 0
        st.rerun()

def render_login_screen() -> None:
    """Renders user authentication landing overlay."""
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown(
            "<div style='text-align:center; margin-top: 2rem;'>", 
            unsafe_allow_html=True
        )
        st.markdown("<h1 class='academic-header'>🎓 AI Academic Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9ca3af;'>Please log in or register a new account to access the academic workspace.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔑 User Login", "📝 Create Account"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username:")
                password = st.text_input("Password:", type="password")
                btn_login = st.form_submit_button("Log In", use_container_width=True)
                
                if btn_login:
                    if authenticate_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username.strip().lower()
                        st.success("Access Granted! Loading workspace...")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please verify your username and password.")
                        
        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Desired Username:")
                new_password = st.text_input("Password:", type="password")
                new_password_confirm = st.text_input("Confirm Password:", type="password")
                btn_register = st.form_submit_button("Register Account", use_container_width=True)
                
                if btn_register:
                    if new_password != new_password_confirm:
                        st.error("Passwords do not match.")
                    elif not new_username.strip() or not new_password:
                        st.error("Fields cannot be empty.")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message + " You can now switch to the Login tab.")
                        else:
                            st.error(message)

def main() -> None:
    # Initialize SQLite database schema
    initialize_db()
    
    initialize_session_state()
    apply_premium_styles()
    
    # ------------------ AUTHENTICATION OVERLAY ------------------
    if not st.session_state.authenticated:
        render_login_screen()
        return

    # ------------------ SIDEBAR CONFIGURATION ------------------
    with st.sidebar:
        st.markdown(f"<h2 style='margin-bottom:0.2rem;'>👤 Welcome, {st.session_state.username.capitalize()}</h2>", unsafe_allow_html=True)
        if st.button("Logout 🚪", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            reset_session_state()
            st.rerun()
            
        st.divider()
        st.markdown("<h3>⚙️ Configuration</h3>", unsafe_allow_html=True)
        
        # 1. API Key Field (Optional override)
        api_key_env = os.getenv("GEMINI_API_KEY", "")
        key_placeholder = "Loaded from .env" if api_key_env else "Enter Gemini API Key..."
        
        st.text_input(
            "Google Gemini API Key:",
            type="password",
            key="api_key_override",
            placeholder=key_placeholder,
            help="Provide your API key to override the default key configured in the application environment."
        )
        
        active_api_key = get_api_key()
        
        # Guard rails: Verify API key configuration
        if not active_api_key:
            st.warning("⚠️ Google Gemini API Key is missing. Please configure it in your '.env' file or input it above to proceed.")
            
        # 2. Research Mode Toggle
        research_mode = st.toggle(
            "🔬 Research Paper Mode", 
            help="Tuning LLM prompts to adopt a highly formal peer-reviewed journal editor tone, critical methodology analysis, and dense academic prose."
        )
        
        st.divider()
        
        # 3. File Uploading Widget
        st.markdown("<h3>📄 Document Upload</h3>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload research PDF papers (Multi-PDF):",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader_widget",
            disabled=not active_api_key
        )
        
        if uploaded_files and active_api_key:
            handle_multi_pdf_upload(uploaded_files, active_api_key)
            
        st.divider()
        
        # 4. System Status Metrics
        st.markdown("<h3>📊 Index Details</h3>", unsafe_allow_html=True)
        if st.session_state.pdf_uploaded:
            st.success("🟢 Multi-PDF Index Active")
            st.write(f"**Loaded Files:** {len(st.session_state.pdf_metadata_list)}")
            
            # Display citations list
            with st.expander("📚 APA/IEEE Citations", expanded=False):
                for idx, doc in enumerate(st.session_state.pdf_metadata_list):
                    st.markdown(f"**{idx + 1}. {doc['name']}**")
                    st.text_area("APA Style:", doc["citations"]["apa"], height=70, key=f"apa_{idx}")
                    st.text_area("IEEE Style:", doc["citations"]["ieee"], height=70, key=f"ieee_{idx}")
                    st.divider()
                    
            if st.button("❌ Clear Index", use_container_width=True):
                reset_session_state()
                st.rerun()
        else:
            st.info("⚪ No papers indexed. Upload research publications to build a search index.")
            
        st.divider()
        st.markdown(
            "<div style='font-size:0.8rem; color:#64748b; text-align:center;'>"
            "AI Academic Assistant v1.1.0<br/>Powered by Google Gemini & LangChain"
            "</div>", 
            unsafe_allow_html=True
        )

    # ------------------ MAIN SCREEN DASHBOARD ------------------
    st.markdown("<h1 class='academic-header'>🎓 AI Academic Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p class='academic-subtitle'>Compile studies, draft essays, generate interactive MCQ quizzes, and build flowcharts across multiple indexed papers.</p>", unsafe_allow_html=True)
    
    # Global check: Verify active PDF documents
    if not active_api_key:
        st.info("👋 Welcome! Please enter your Gemini API Key in the sidebar configuration to unlock search features and LLM generation adapters.")
        return
        
    if not st.session_state.pdf_uploaded:
        st.info("💡 Getting Started: Drag and drop or browse for research PDF papers in the sidebar to extract text, index vectors, and start learning.")
        return

    # PDF selection toolbar for material generation
    st.markdown("<h4 style='margin-bottom: 0.4rem;'>Select Active Publication for Material Generation:</h4>", unsafe_allow_html=True)
    doc_options = ["Combined (All Indexed Publications)"] + [doc["name"] for doc in st.session_state.pdf_metadata_list]
    selected_doc_name = st.selectbox(
        "Generate summaries, notes, assignments, and presentations based on:",
        options=doc_options,
        key="active_document_dropdown"
    )
    
    # Concatenate texts if combined, otherwise select individual text
    if selected_doc_name == "Combined (All Indexed Publications)":
        active_text = "\n\n".join([doc["text"] for doc in st.session_state.pdf_metadata_list])
        doc_key = "combined_all"
    else:
        active_text = next(doc["text"] for doc in st.session_state.pdf_metadata_list if doc["name"] == selected_doc_name)
        doc_key = selected_doc_name
        
    # Inject Research Mode modifier into active text if enabled
    if research_mode:
        active_text_for_llm = RESEARCH_MODE_MODIFIER + "\n\n" + active_text
    else:
        active_text_for_llm = active_text

    # Build primary Navigation Tabs
    tab_chat, tab_summary, tab_assignment, tab_notes, tab_viva, tab_mcq, tab_ppt = st.tabs([
        "💬 Chat with PDF",
        "📝 Summarizer",
        "✍️ Assignment",
        "📖 Study Notes",
        "❓ Viva Questions",
        "🎯 MCQs Quiz",
        "📊 Slide Deck"
    ])

    # ------------------ TAB 1: CHAT WITH PDF ------------------
    with tab_chat:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>💬 Multi-PDF RAG Chat</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Ask questions about the paper. The system will search local FAISS vectors across all uploaded documents and answer ONLY using details mentioned in the documents.</p>", unsafe_allow_html=True)
        
        # Display chat container
        chat_container = st.container()
        with chat_container:
            for idx, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    if message["role"] == "assistant" and message["content"] != "Information not found in uploaded document.":
                        render_voice_button(message["content"], idx)
                    
        # Check for user input queries
        if user_query := st.chat_input("Ask a question about the documents..."):
            # Display user message instantly
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_query)
            
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Query the RAG chat service
            with st.spinner("Searching document index..."):
                try:
                    chat_service = ChatService(api_key=active_api_key)
                    # Use the combined vector store
                    answer = chat_service.query(user_query, st.session_state.vector_store)
                    
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.write(answer)
                            render_voice_button(answer, len(st.session_state.chat_history))
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Chat execution failed: {str(e)}")

    # ------------------ TAB 2: SUMMARIZER ------------------
    with tab_summary:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>📝 Structured Executive Summary</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Generate high-density academic summary covering key topics, core concepts, and definitions.</p>", unsafe_allow_html=True)
        
        cached_summary = st.session_state.summary_cache.get(doc_key)
        
        if cached_summary is None:
            if st.button("Generate Summary", type="primary", key="btn_summary"):
                with st.spinner("Synthesizing executive summary..."):
                    try:
                        summary_service = SummaryService(api_key=active_api_key)
                        res = summary_service.summarize_pdf_content(active_text_for_llm)
                        st.session_state.summary_cache[doc_key] = res
                        st.rerun()
                    except SummaryServiceError as e:
                        st.error(f"Summarizer failed: {str(e)}")
        else:
            data = cached_summary
            
            # Exporters bar
            col_d1, col_d2, col_d3 = st.columns([2, 2, 8])
            with col_d1:
                docx_data = export_to_docx(f"Summary - {selected_doc_name}", data)
                st.download_button(
                    "📥 Export as Word",
                    data=docx_data,
                    file_name=f"Summary_{doc_key}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_summary_docx_{doc_key}"
                )
            with col_d2:
                pdf_data = export_to_pdf(f"Summary - {selected_doc_name}", data)
                st.download_button(
                    "📥 Export as PDF",
                    data=pdf_data,
                    file_name=f"Summary_{doc_key}.pdf",
                    mime="application/pdf",
                    key=f"dl_summary_pdf_{doc_key}"
                )
            
            st.write("")
            
            # Executive Summary card
            st.markdown(
                f"""
                <div class='premium-card'>
                    <div class='section-label'>📌 Executive Summary</div>
                    <p style='font-size: 1.05rem; line-height: 1.6; color: #f1f5f9;'>{data.get("executive_summary", "")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Expander layout for lists
            with st.expander("🔑 Key Topics Covered", expanded=True):
                st.markdown(data.get("key_topics", "No topics parsed."))
                
            with st.expander("💡 Core Concepts & Theories", expanded=True):
                st.markdown(data.get("important_concepts", "No concepts parsed."))
                
            with st.expander("📖 Glossary & Key Definitions", expanded=True):
                st.markdown(data.get("important_definitions", "No definitions parsed."))
                
            st.divider()
            
            # Diagram Visualizer
            st.markdown("<h3>📊 Concept Map Diagram</h3>", unsafe_allow_html=True)
            cached_diagram = st.session_state.diagram_cache.get(doc_key)
            if cached_diagram is None:
                if st.button("Generate Diagram Flowchart", key=f"btn_diag_{doc_key}"):
                    with st.spinner("Generating concept map..."):
                        try:
                            client = GeminiClient(api_key=active_api_key, model_name=settings.gemini_model, temperature=0.2)
                            response = client.execute_prompt(MERMAID_DIAGRAM_PROMPT, {"context": data.get("important_concepts", "")})
                            st.session_state.diagram_cache[doc_key] = response
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to generate diagram: {str(e)}")
            else:
                render_mermaid(cached_diagram)
                with st.expander("Show Raw Diagram Syntax Code"):
                    st.code(cached_diagram, language="mermaid")
                    if st.button("Clear Diagram", key=f"clear_diag_{doc_key}"):
                        st.session_state.diagram_cache[doc_key] = None
                        st.rerun()
                        
            st.write("")
            if st.button("🔄 Regenerate Summary", key=f"regen_summary_{doc_key}"):
                st.session_state.summary_cache[doc_key] = None
                st.session_state.diagram_cache[doc_key] = None
                st.rerun()

    # ------------------ TAB 3: ASSIGNMENT ------------------
    with tab_assignment:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>✍️ Academic Assignment Writer</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Generate well-structured scholarly papers based on the document text with bibliography.</p>", unsafe_allow_html=True)
        
        col_length, col_action = st.columns([2, 3])
        with col_length:
            length_choice = st.selectbox(
                "Select Assignment Target Length:",
                options=[3, 5, 10],
                format_func=lambda x: f"{x} Pages (approx. {x * 250} words)",
                key=f"dropdown_len_{doc_key}"
            )
            
        cached_assign = st.session_state.assignment_cache.get((doc_key, length_choice))
        
        with col_action:
            st.write("") # vertical spacing helper
            st.write("")
            btn_label = "Generate Assignment" if not cached_assign else "Regenerate Assignment"
            clicked = st.button(btn_label, type="primary" if not cached_assign else "secondary", key=f"btn_assign_{doc_key}")
            
        if clicked:
            with st.spinner(f"Writing {length_choice}-page assignment... This may take a minute."):
                try:
                    assignment_service = AssignmentService(api_key=active_api_key)
                    res = assignment_service.generate_assignment(active_text_for_llm, page_length=length_choice)
                    st.session_state.assignment_cache[(doc_key, length_choice)] = res
                    st.rerun()
                except AssignmentServiceError as e:
                    st.error(f"Assignment generator failed: {str(e)}")
                    
        if cached_assign:
            st.divider()
            
            # Exporters bar
            col_d1, col_d2, col_d3 = st.columns([2, 2, 8])
            with col_d1:
                docx_data = export_to_docx(cached_assign.get('title', 'Assignment'), cached_assign)
                st.download_button(
                    "📥 Export as Word",
                    data=docx_data,
                    file_name=f"Assignment_{doc_key}_{length_choice}p.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_assign_docx_{doc_key}_{length_choice}"
                )
            with col_d2:
                pdf_data = export_to_pdf(cached_assign.get('title', 'Assignment'), cached_assign)
                st.download_button(
                    "📥 Export as PDF",
                    data=pdf_data,
                    file_name=f"Assignment_{doc_key}_{length_choice}p.pdf",
                    mime="application/pdf",
                    key=f"dl_assign_pdf_{doc_key}_{length_choice}"
                )
                
            st.markdown(f"<h2 style='text-align:center; font-family:Playfair Display;'>{cached_assign.get('title', 'Academic Essay')}</h2>", unsafe_allow_html=True)
            
            # Navigating layout via tabs
            tab_intro, tab_content, tab_conclusion, tab_refs = st.tabs([
                "📍 Introduction",
                "📖 Main Analysis",
                "🎯 Conclusion",
                "📚 References"
            ])
            
            with tab_intro:
                st.markdown(cached_assign.get("introduction", ""))
                
            with tab_content:
                st.markdown(cached_assign.get("main_content", ""))
                
            with tab_conclusion:
                st.markdown(cached_assign.get("conclusion", ""))
                
            with tab_refs:
                st.markdown(cached_assign.get("references", ""))

    # ------------------ TAB 4: STUDY NOTES ------------------
    with tab_notes:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>📖 Structured Study Notes</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Synthesize chapter notes, highlights, revision guidelines, and exam preparation tips.</p>", unsafe_allow_html=True)
        
        cached_notes = st.session_state.notes_cache.get(doc_key)
        
        if cached_notes is None:
            if st.button("Generate Study Notes", type="primary", key=f"btn_notes_{doc_key}"):
                with st.spinner("Compiling study notes..."):
                    try:
                        notes_service = NotesService(api_key=active_api_key)
                        res = notes_service.generate_notes(active_text_for_llm)
                        st.session_state.notes_cache[doc_key] = res
                        st.rerun()
                    except NotesServiceError as e:
                        st.error(f"Study notes compilation failed: {str(e)}")
        else:
            notes = cached_notes
            
            # Exporters bar
            col_d1, col_d2, col_d3 = st.columns([2, 2, 8])
            with col_d1:
                docx_data = export_to_docx(f"Study Notes - {selected_doc_name}", notes)
                st.download_button(
                    "📥 Export as Word",
                    data=docx_data,
                    file_name=f"Study_Notes_{doc_key}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_notes_docx_{doc_key}"
                )
            with col_d2:
                pdf_data = export_to_pdf(f"Study Notes - {selected_doc_name}", notes)
                st.download_button(
                    "📥 Export as PDF",
                    data=pdf_data,
                    file_name=f"Study_Notes_{doc_key}.pdf",
                    mime="application/pdf",
                    key=f"dl_notes_pdf_{doc_key}"
                )
                
            st.write("")
            
            tab_n1, tab_n2, tab_n3, tab_n4 = st.tabs([
                "📋 Chapter-wise Notes",
                "💡 Key Theories",
                "✅ Revision Points",
                "🎯 Exam Tips"
            ])
            
            with tab_n1:
                st.markdown(notes.get("chapter_notes", ""))
            with tab_n2:
                st.markdown(notes.get("key_concepts", ""))
            with tab_n3:
                st.markdown(notes.get("revision_points", ""))
            with tab_n4:
                st.markdown(notes.get("exam_tips", ""))
                
            st.divider()
            if st.button("🔄 Regenerate Notes", key=f"regen_notes_{doc_key}"):
                st.session_state.notes_cache[doc_key] = None
                st.rerun()

    # ------------------ TAB 5: VIVA QUESTIONS ------------------
    with tab_viva:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>❓ Oral Examination Prep (Viva Voce)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Generate 20 viva questions and suggested answer blueprints across beginner, intermediate, and advanced levels.</p>", unsafe_allow_html=True)
        
        cached_viva = st.session_state.viva_cache.get(doc_key)
        
        if cached_viva is None:
            if st.button("Generate Viva Prep", type="primary", key=f"btn_viva_{doc_key}"):
                with st.spinner("Generating 60 oral exam questions..."):
                    try:
                        viva_service = VivaService(api_key=active_api_key)
                        res = viva_service.generate_viva_questions(active_text_for_llm)
                        st.session_state.viva_cache[doc_key] = res
                        st.rerun()
                    except VivaServiceError as e:
                        st.error(f"Viva generator failed: {str(e)}")
        else:
            viva = cached_viva
            
            # Exporters bar
            col_d1, col_d2, col_d3 = st.columns([2, 2, 8])
            with col_d1:
                docx_data = export_to_docx(f"Viva Blueprint - {selected_doc_name}", viva)
                st.download_button(
                    "📥 Export as Word",
                    data=docx_data,
                    file_name=f"Viva_Blueprint_{doc_key}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_viva_docx_{doc_key}"
                )
            with col_d2:
                pdf_data = export_to_pdf(f"Viva Blueprint - {selected_doc_name}", viva)
                st.download_button(
                    "📥 Export as PDF",
                    data=pdf_data,
                    file_name=f"Viva_Blueprint_{doc_key}.pdf",
                    mime="application/pdf",
                    key=f"dl_viva_pdf_{doc_key}"
                )
                
            st.write("")
            
            def display_viva_questions(markdown_content: str):
                if not markdown_content.strip():
                    st.info("No questions generated for this category.")
                    return
                st.markdown(markdown_content)
                
            tab_v1, tab_v2, tab_v3 = st.tabs([
                "🟢 Beginner Level",
                "🟡 Intermediate Level",
                "🔴 Advanced Level"
            ])
            
            with tab_v1:
                display_viva_questions(viva.get("beginner", ""))
            with tab_v2:
                display_viva_questions(viva.get("intermediate", ""))
            with tab_v3:
                display_viva_questions(viva.get("advanced", ""))
                
            st.divider()
            if st.button("🔄 Regenerate Viva Blueprint", key=f"regen_viva_{doc_key}"):
                st.session_state.viva_cache[doc_key] = None
                st.rerun()

    # ------------------ TAB 6: MCQS QUIZ ------------------
    with tab_mcq:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>🎯 Interactive MCQs Assessment</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Generate a set of 25 multiple-choice questions with options, answers, and explanations to test your knowledge.</p>", unsafe_allow_html=True)
        
        cached_mcq = st.session_state.mcq_cache.get(doc_key)
        
        if cached_mcq is None:
            if st.button("Generate MCQs Quiz", type="primary", key=f"btn_mcq_{doc_key}"):
                with st.spinner("Generating 25 interactive questions..."):
                    try:
                        mcq_service = MCQService(api_key=active_api_key)
                        res = mcq_service.generate_mcqs(active_text_for_llm)
                        st.session_state.mcq_cache[doc_key] = res
                        st.rerun()
                    except MCQServiceError as e:
                        st.error(f"MCQ generator failed: {str(e)}")
        else:
            mcqs = cached_mcq
            
            # Exporters bar (Word file containing questions and choices)
            mcq_txt_dict = {}
            for q_idx, item in enumerate(mcqs):
                mcq_txt_dict[f"Q{q_idx+1}. {item.get('question')}"] = (
                    f"A) {item['options'].get('A', '')}\n"
                    f"B) {item['options'].get('B', '')}\n"
                    f"C) {item['options'].get('C', '')}\n"
                    f"D) {item['options'].get('D', '')}\n\n"
                    f"Correct Answer: {item.get('answer', '')}\n"
                    f"Explanation: {item.get('explanation', '')}"
                )
            
            col_d1, col_d2 = st.columns([3, 9])
            with col_d1:
                docx_data = export_to_docx(f"MCQs Assessment - {selected_doc_name}", mcq_txt_dict)
                st.download_button(
                    "📥 Download Quiz as Word",
                    data=docx_data,
                    file_name=f"MCQ_Quiz_{doc_key}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_mcq_docx_{doc_key}"
                )
                
            st.write("")
            st.write(f"📝 Total generated questions: **{len(mcqs)}**")
            
            # Render MCQ Quiz
            for idx, item in enumerate(mcqs):
                st.markdown(f"<div class='mcq-block'>", unsafe_allow_html=True)
                st.markdown(f"**Q{idx + 1}. {item.get('question')}**")
                
                options = item.get("options", {})
                option_keys = ["A", "B", "C", "D"]
                
                # Check for cached choice
                stored_answer = st.session_state.mcq_user_answers.get((doc_key, idx), None)
                selected_idx = option_keys.index(stored_answer) if stored_answer in option_keys else None
                
                selected_option = st.radio(
                    f"Choose option for Q{idx+1}:",
                    options=option_keys,
                    format_func=lambda x: f"{x}) {options.get(x, '')}",
                    key=f"widget_mcq_{doc_key}_{idx}",
                    index=selected_idx if selected_idx is not None else 0
                )
                
                # Check answers
                btn_check = st.button(f"Submit Answer Q{idx+1}", key=f"btn_mcq_chk_{doc_key}_{idx}")
                if btn_check or stored_answer is not None:
                    st.session_state.mcq_user_answers[(doc_key, idx)] = selected_option
                    
                    correct_answer = item.get("answer", "").strip().upper()
                    is_correct = (selected_option == correct_answer)
                    
                    if is_correct:
                        st.success(f"✓ Correct! The answer is {correct_answer}.")
                    else:
                        st.error(f"✗ Incorrect. Your choice: {selected_option} | Correct Answer: {correct_answer}")
                        
                    st.info(f"**Explanation:** {item.get('explanation')}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.write("")
                
            st.divider()
            if st.button("🔄 Regenerate Quiz", key=f"regen_mcq_{doc_key}"):
                st.session_state.mcq_cache[doc_key] = None
                # Clear options for this document
                for k in list(st.session_state.mcq_user_answers.keys()):
                    if k[0] == doc_key:
                        del st.session_state.mcq_user_answers[k]
                st.rerun()

    # ------------------ TAB 7: SLIDE DECK ------------------
    with tab_ppt:
        st.markdown("<h3 style='margin-bottom:0.2rem;'>📊 Presentation Slides Planner</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8; font-size:0.95rem; margin-bottom:1.5rem;'>Synthesize an academic 10-slide deck complete with visual slide points and presenter speaker notes.</p>", unsafe_allow_html=True)
        
        cached_ppt = st.session_state.ppt_cache.get(doc_key)
        
        if cached_ppt is None:
            if st.button("Generate Slide Deck", type="primary", key=f"btn_ppt_{doc_key}"):
                with st.spinner("Synthesizing 10 slide frameworks..."):
                    try:
                        ppt_service = PPTService(api_key=active_api_key)
                        res = ppt_service.generate_slides(active_text_for_llm)
                        st.session_state.ppt_cache[doc_key] = res
                        st.rerun()
                    except PPTServiceError as e:
                        st.error(f"Slide generator failed: {str(e)}")
        else:
            slides = cached_ppt
            
            # Exporters bar (PPTX File download)
            col_d1, col_d2 = st.columns([3, 9])
            with col_d1:
                pptx_data = export_to_pptx(slides)
                st.download_button(
                    "📥 Download Slide Deck (.pptx)",
                    data=pptx_data,
                    file_name=f"Presentation_{doc_key}.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    key=f"dl_ppt_pptx_{doc_key}"
                )
                
            st.write("")
            
            # Horizontal slide deck navigator
            slide_idx = st.slider("Select Slide Navigation:", min_value=1, max_value=len(slides), step=1, key=f"slider_ppt_{doc_key}")
            current_slide = slides[slide_idx - 1]
            
            # Slide Canvas Preview
            st.markdown(
                f"""
                <div class='ppt-slide-canvas'>
                    <div>
                        <div class='ppt-slide-title'>{current_slide.get("title", f"Slide {slide_idx}")}</div>
                        <div style='padding-left: 1.2rem; line-height: 1.8;'>
                            {"".join([f"<div class='ppt-slide-bullet'>• {bullet}</div>" for bullet in current_slide.get("bullets", [])])}
                        </div>
                    </div>
                    <div class='ppt-slide-footer'>
                        <span>🎓 AI Academic Assistant</span>
                        <span>Slide {slide_idx} of {len(slides)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Speaker notes drawer
            with st.expander("🎙️ Speaker Notes (Presenter Script)", expanded=True):
                st.info(current_slide.get("speaker_notes", "No speaker script available."))
                
            st.divider()
            if st.button("🔄 Regenerate Slides", key=f"regen_ppt_{doc_key}"):
                st.session_state.ppt_cache[doc_key] = None
                st.rerun()

if __name__ == "__main__":
    main()
