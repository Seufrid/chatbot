import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import tempfile
import time
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="Finance Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for policy names
if "policy_display_names" not in st.session_state:
    st.session_state.policy_display_names = {}

# Custom CSS to hide Streamlit elements for non-admin users
def apply_custom_css():
    if not is_admin():
        st.markdown("""
        <style>
        /* Hide only the footer for regular users, keep the menu visible */
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Clean up the interface */
        .css-1d391kg {padding-top: 1rem;}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        /* Admin view styling */
        .admin-header {
            background-color: #ff4b4b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Check if admin mode
def is_admin():
    query_params = st.query_params
    if query_params.get("admin") == "true":
        return True
    if st.session_state.get("admin_authenticated"):
        return True
    return False

# Apply custom CSS
apply_custom_css()

# Initialize API keys from secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            return pc
        except Exception as e:
            if is_admin():
                st.error(f"Pinecone initialization error: {str(e)}")
            return None
    return None

# Functions to manage policy display names
def save_policy_names():
    """Save policy display names to Streamlit secrets or local storage"""
    # In production, you'd want to save this to a database or persistent storage
    # For now, we'll keep it in session state
    pass

def get_display_name(namespace):
    """Get the display name for a namespace, or return formatted namespace if not set"""
    if namespace in st.session_state.policy_display_names:
        return st.session_state.policy_display_names[namespace]
    return namespace.replace('_', ' ').title()

# Test function for debugging (admin only)
def test_search_function(query):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        stats = index.describe_index_stats()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        results_summary = []
