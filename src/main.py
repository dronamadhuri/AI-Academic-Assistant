import streamlit as st
from src.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

def main() -> None:
    """Streamlit Application Entrypoint."""
    st.set_page_config(
        page_title="AI Academic Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    st.title("🎓 AI Academic Assistant")
    st.markdown(
        """
        Welcome to the **AI Academic Assistant**! This application helps you parse, index, 
        and run academic-level QA and summarization over your research papers.
        """
    )
    
    # UI Layout placeholders
    st.info("Please configure your `.env` variables and upload PDFs via the sidebar to begin.")

if __name__ == "__main__":
    main()
