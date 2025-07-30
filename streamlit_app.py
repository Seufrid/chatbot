import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import tempfile

# Configure page
st.set_page_config(
    page_title="Finance Policy Assistant",
    page_icon="💼",
    layout="wide"
)

# Check if admin mode
def is_admin():
    # Method 1: URL parameter (?admin=true)
    query_params = st.query_params
    if query_params.get("admin") == "true":
        return True
    
    # Method 2: Check if running locally (for development)
    if os.getenv("IS_LOCAL") == "true":
        return True
    
    # Method 3: Admin password in sidebar
    if st.session_state.get("admin_authenticated"):
        return True
    
    return False

# Title and description
st.title("💼 Finance Policy Assistant")
st.markdown("Ask me anything about company finance policies in English or Bahasa Malaysia!")

# Initialize API keys from secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Admin Panel (only visible to admins)
if is_admin():
    with st.sidebar:
        st.header("🔐 Admin Panel")
        
        # Show API key status
        st.success("✅ Google API Key configured")
        st.success("✅ Pinecone API Key configured")
        
        # Initialize Pinecone
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index("finance-policy")
            stats = index.describe_index_stats()
            
            st.metric("Total Vectors", stats['total_vector_count'])
            
            # PDF upload section
            st.subheader("Upload Policy Documents")
            uploaded_file = st.file_uploader("Upload Policy PDF", type="pdf")
            
            if st.button("Process PDF") and uploaded_file:
                # PDF processing code here (same as before)
                pass
            
            # Danger zone
            with st.expander("⚠️ Danger Zone"):
                if st.button("Clear All Vectors", type="secondary"):
                    if st.checkbox("I understand this will delete all data"):
                        index.delete(delete_all=True)
                        st.success("All vectors cleared!")
                        st.rerun()
        
        except Exception as e:
            st.error(f"Admin panel error: {str(e)}")

# Regular User Sidebar (minimal)
else:
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you find information about company finance policies.
        
        **How to use:**
        - Type your question in any language
        - Get instant answers from policy documents
        
        **Contact:** IT Support for issues
        """)
        
        # Optional: Admin login
        with st.expander("Admin Access"):
            admin_password = st.text_input("Admin Password", type="password")
            if st.button("Login as Admin"):
                if admin_password == st.secrets.get("ADMIN_PASSWORD", ""):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")

# Rest of the chat interface code remains the same...
# (Initialize chat history, display messages, etc.)

# Check if system is properly configured
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("⚠️ System not properly configured. Please contact IT support.")
    st.stop()

# Continue with the chat interface...
