from typing import List, Callable, Dict, Any
import streamlit as st

def render_sidebar(
    on_file_upload: Callable[[List[Any]], None],
    status_metrics: Dict[str, Any]
) -> None:
    """
    Renders the sidebar interface for uploading papers and displaying vector database info.
    
    Args:
        on_file_upload: Callback function invoked when files are uploaded.
        status_metrics: Dictionary containing system metrics to display.
    """
    with st.sidebar:
        st.header("📄 Upload Papers")
        uploaded_files = st.file_uploader(
            "Upload research PDF files:",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            on_file_upload(uploaded_files)

        st.divider()
        st.subheader("⚙️ System Status")
        st.write(f"**Index Active:** {status_metrics.get('index_active', False)}")
        st.write(f"**Indexed Chunks:** {status_metrics.get('chunks_count', 0)}")
        st.write(f"**Uploaded Files:** {len(status_metrics.get('files_list', []))}")

def render_chat_interface(
    messages: List[Dict[str, str]],
    on_send_message: Callable[[str], None]
) -> None:
    """
    Renders the RAG chat dialog message thread and prompts user input.
    
    Args:
        messages: Historical chat messages containing keys: "role" and "content".
        on_send_message: Callback function when user submits a new prompt.
    """
    st.subheader("🔍 Ask the Assistant")
    
    # Display historical chat messages
    for message in messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User chat input field
    if user_query := st.chat_input("Ask a question about the uploaded papers..."):
        # Display the prompt in the UI immediately
        with st.chat_message("user"):
            st.write(user_query)
        # Execute message processing callback
        on_send_message(user_query)

def render_document_summaries(documents_metadata: List[Dict[str, Any]]) -> None:
    """
    Renders academic paper summaries or metadata index overview tabs.
    
    Args:
        documents_metadata: List of dicts representing parsed papers' profiles.
    """
    st.subheader("📚 Loaded Publications")
    if not documents_metadata:
        st.info("No documents indexed yet.")
        return

    for doc in documents_metadata:
        with st.expander(f"📖 {doc.get('title', 'Unknown Paper')}"):
            st.write(f"**File Size:** {doc.get('file_size', 'N/A')} bytes")
            st.write(f"**Page Count:** {doc.get('page_count', 'N/A')}")
            st.write(f"**Short Abstract:** {doc.get('abstract', 'No summary generated yet.')}")
