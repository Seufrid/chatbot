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

# Initialize session state for theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Initialize session state for policy display names
if 'policy_display_names' not in st.session_state:
    st.session_state.policy_display_names = {}

# Check if admin mode
def is_admin():
    query_params = st.query_params
    if query_params.get("admin") == "true":
        return True
    if st.session_state.get("admin_authenticated"):
        return True
    return False

# Theme toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Apply custom CSS based on theme and user type
def apply_custom_css():
    # Base theme colors
    if st.session_state.dark_mode:
        bg_color = "#0e1117"
        text_color = "#fafafa"
        sidebar_bg = "#262730"
        card_bg = "#1e1e1e"
    else:
        bg_color = "#ffffff"
        text_color = "#262730"
        sidebar_bg = "#f0f2f6"
        card_bg = "#f8f9fa"
    
    # CSS for all users
    base_css = f"""
    <style>
    /* Theme styling */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    
    .stSidebar {{
        background-color: {sidebar_bg};
    }}
    
    /* Theme toggle button styling */
    .theme-toggle {{
        position: fixed;
        top: 14px;
        right: 50px;
        z-index: 999;
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        padding: 8px;
        border-radius: 4px;
        transition: background-color 0.3s;
    }}
    
    .theme-toggle:hover {{
        background-color: {card_bg};
    }}
    </style>
    """
    
    # Additional CSS for non-admin users to hide Streamlit Cloud features
    if not is_admin():
        hide_features_css = """
        <style>
        /* Hide Streamlit Cloud deployment features */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Hide the toolbar buttons (star, fork, github) */
        [data-testid="stToolbar"] {
            display: none !important;
        }
        
        /* Hide share button and related features */
        button[title="Share this app"] {
            display: none !important;
        }
        
        /* Hide view app source */
        button[title="View app source"] {
            display: none !important;
        }
        
        /* Hide the manage app button */
        ._profileContainer_gzau3_53 {
            display: none !important;
        }
        
        /* Clean up spacing */
        .css-1d391kg {padding-top: 1rem;}
        .stApp > header {
            height: 0px !important;
        }
        </style>
        """
        st.markdown(base_css + hide_features_css, unsafe_allow_html=True)
    else:
        # Admin styling
        admin_css = """
        <style>
        /* Admin view styling */
        .admin-header {
            background-color: #ff4b4b;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Keep some features visible for admin */
        [data-testid="stToolbar"] {
            right: 100px !important;
        }
        </style>
        """
        st.markdown(base_css + admin_css, unsafe_allow_html=True)
    
    # Add theme toggle button
    theme_icon = "🌙" if not st.session_state.dark_mode else "☀️"
    st.markdown(f"""
    <button class="theme-toggle" onclick="window.parent.postMessage('toggle_theme', '*')">
        {theme_icon}
    </button>
    """, unsafe_allow_html=True)
    
    # JavaScript to handle theme toggle
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data === 'toggle_theme') {
            const doc = window.parent.document;
            doc.querySelector('[data-testid="stSidebar"] button').click();
        }
    });
    </script>
    """, unsafe_allow_html=True)

# Apply custom CSS
apply_custom_css()

# Add theme toggle button in sidebar (as backup)
with st.sidebar:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_toggle", help="Toggle theme"):
            toggle_theme()
            st.rerun()

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

# Test function for debugging (admin only)
def test_search_function(query):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        stats = index.describe_index_stats()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        results_summary = []
        
        for namespace in stats.get('namespaces', {}):
            if namespace:
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=1,
                        include_metadata=True,
                        include_values=False,
                        namespace=namespace
                    )
                    
                    results_summary.append(f"Namespace '{namespace}': {len(results['matches'])} matches")
                    
                    if results['matches']:
                        first_match = results['matches'][0]
                        results_summary.append(f"  - Score: {first_match['score']:.3f}")
                        text_content = first_match.get('metadata', {}).get('text', '')
                        if text_content:
                            snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
                            results_summary.append(f"  - Text snippet: {snippet}")
                        
                except Exception as e:
                    results_summary.append(f"Namespace '{namespace}': Error - {str(e)}")
        
        return "\n".join(results_summary)
            
    except Exception as e:
        return f"Error in test: {str(e)}"

# Get display name for a policy
def get_policy_display_name(namespace):
    if namespace in st.session_state.policy_display_names:
        return st.session_state.policy_display_names[namespace]
    # Default formatting if no custom name
    return namespace.replace('_', ' ').replace('(', '').replace(')', '').title()

# Search function
def get_relevant_context(query, k=6):
    try:
        if not PINECONE_API_KEY:
            return None
            
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        all_results = []
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        for namespace in namespaces:
            if namespace:
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=k,
                        include_metadata=True,
                        include_values=False,
                        namespace=namespace
                    )
                    
                    for match in results['matches']:
                        text_content = match['metadata'].get('text', '')
                        
                        if text_content and len(text_content.strip()) > 20:
                            # Use display name instead of raw namespace
                            display_name = get_policy_display_name(namespace)
                            all_results.append({
                                'score': match['score'],
                                'text': text_content,
                                'source': match['metadata'].get('source_file', display_name),
                                'page': match['metadata'].get('page', 'Unknown')
                            })
                        
                except Exception as e:
                    continue
        
        if not all_results:
            return ""
        
        all_results.sort(key=lambda x: x['score'], reverse=True)
        best_results = all_results[:k]
        
        context_parts = []
        for result in best_results:
            context_parts.append(
                f"[Source: {result['source']}, Page: {result['page']}, Relevance: {result['score']:.3f}]\n{result['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
        
    except Exception as e:
        if is_admin():
            st.error(f"Search error: {str(e)}")
        return None

# Function to generate response
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        chat_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages[-10:]
        ])
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context.

Context from company policies:
{context}

Recent conversation:
{chat_history}

User Question: {query}

Instructions:
- Answer based on the provided context from the policy documents
- Be specific and cite the source document and page when possible
- If the answer isn't in the context, say so politely
- Be concise but thorough
- Include relevant policy references when available
- Respond in the same language as the question (English or Bahasa Malaysia)

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Streamlined text validation function
def validate_pdf_text(documents):
    total_text = ""
    for doc in documents:
        total_text += doc.page_content.strip() + " "
    
    text_length = len(total_text.strip())
    word_count = len(total_text.split())
    
    if text_length < 100:
        return False, "Very little text found. This might be a scanned/image-based PDF that needs OCR processing."
    
    if word_count < 50:
        return False, "Very few words found. The PDF might have formatting issues."
    
    return True, f"Text validation passed: {word_count:,} words found"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# ADMIN VIEW
if is_admin():
    # Admin header
    st.markdown('<div class="admin-header">🔐 Admin Dashboard</div>', unsafe_allow_html=True)
    
    # Create two columns for admin layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.header("Admin Controls")
        
        # Logout button
        if st.button("🚪 Logout", type="secondary", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()
        
        # System Status
        st.subheader("System Status")
        st.success("✅ Google API Key" if GOOGLE_API_KEY else "❌ Google API Key")
        st.success("✅ Pinecone API Key" if PINECONE_API_KEY else "❌ Pinecone API Key")
        
        # Initialize Pinecone and show stats
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                st.metric("Total Vectors", f"{stats['total_vector_count']:,}")
                st.metric("Dimensions", stats['dimension'])
                
                # Usage bar
                usage_percent = (stats['total_vector_count'] / 100000) * 100
                st.progress(usage_percent / 100, text=f"Usage: {usage_percent:.1f}%")
                
                # Policy Display Names Editor
                if stats.get('namespaces'):
                    st.subheader("Policy Names")
                    st.caption("Edit display names")
                    
                    for ns in stats['namespaces'].keys():
                        if ns:
                            current_name = get_policy_display_name(ns)
                            new_name = st.text_input(
                                f"Name for '{ns}':",
                                value=current_name,
                                key=f"name_{ns}"
                            )
                            if new_name != current_name:
                                st.session_state.policy_display_names[ns] = new_name
                    
                    # Save/Load buttons for persistence
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save Names", use_container_width=True):
                            # In production, save to database/file
                            st.success("Names saved!")
                    with col2:
                        if st.button("🔄 Reset Names", use_container_width=True):
                            st.session_state.policy_display_names = {}
                            st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Test Tools
        with st.expander("🧪 Test Tools"):
            if st.button("Test Search"):
                with st.spinner("Testing..."):
                    result = test_search_function("reimbursement")
                    st.code(result)
            
            if st.button("Test Context"):
                with st.spinner("Testing..."):
                    context = get_relevant_context("reimbursement", k=2)
                    if context:
                        st.success("✅ Context found!")
                        st.text_area("Preview", context[:500] + "...", height=150)
                    else:
                        st.error("❌ No context")
    
    with col2:
        # Document Management Section
        st.header("Document Management")
        
        # Upload Section
        tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🗑️ Manage Documents", "💬 Chat Preview"])
        
        with tab1:
            st.subheader("Upload Policy Documents")
            
            # Show available space
            if pc and 'stats' in locals():
                remaining = 100000 - stats['total_vector_count']
                st.info(f"📊 Available space: ~{remaining//1000}k vectors")
            
            uploaded_file = st.file_uploader("Choose PDF file", type="pdf", key="pdf_upload")
            
            if uploaded_file:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📁 {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.1f} MB)")
                with col2:
                    process_btn = st.button("Process PDF", type="primary", use_container_width=True)
                
                if process_btn:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            # Processing steps (abbreviated for space)
                            status_text.text("📁 Saving uploaded file...")
                            progress_bar.progress(15)
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_file_path = tmp_file.name
                            
                            # Load PDF
                            status_text.text("📄 Loading PDF content...")
                            progress_bar.progress(30)
                            
                            loader = PyPDFLoader(tmp_file_path)
                            documents = loader.load()
                            
                            st.write(f"📄 Loaded {len(documents)} pages")
                            
                            # Validate
                            status_text.text("🔍 Validating content...")
                            progress_bar.progress(40)
                            
                            has_valid_text, validation_message = validate_pdf_text(documents)
                            
                            if not has_valid_text:
                                st.error(f"❌ {validation_message}")
                                os.unlink(tmp_file_path)
                                st.stop()
                            
                            # Process and upload
                            status_text.text("✂️ Processing document...")
                            progress_bar.progress(60)
                            
                            # Add metadata
                            for i, doc in enumerate(documents):
                                doc.metadata.update({
                                    'source_file': uploaded_file.name,
                                    'page': i + 1,
                                    'upload_date': datetime.now().isoformat()
                                })
                            
                            # Split
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200
                            )
                            chunks = text_splitter.split_documents(documents)
                            valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 20]
                            
                            st.write(f"✂️ Created {len(valid_chunks)} chunks")
                            
                            # Upload
                            status_text.text("🧠 Creating embeddings...")
                            progress_bar.progress(80)
                            
                            embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2"
                            )
                            
                            namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
                            vectorstore = PineconeVectorStore.from_documents(
                                documents=valid_chunks,
                                embedding=embeddings,
                                index_name="finance-policy",
                                namespace=namespace
                            )
                            
                            status_text.text("✅ Upload complete!")
                            progress_bar.progress(100)
                            
                            st.success(f"✅ Successfully processed {uploaded_file.name}")
                            st.balloons()
                            
                            # Clean up
                            os.unlink(tmp_file_path)
                            time.sleep(2)
                            st.cache_resource.clear()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                                os.unlink(tmp_file_path)
                        finally:
                            progress_bar.empty()
                            status_text.empty()
        
        with tab2:
            if 'stats' in locals() and stats.get('namespaces'):
                st.subheader("Delete Documents")
                
                # Document selection with display names
                namespace_list = [ns for ns in stats['namespaces'].keys() if ns]
                doc_to_delete = st.selectbox(
                    "Select document",
                    namespace_list,
                    format_func=lambda x: get_policy_display_name(x)
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    confirm_delete = st.checkbox("Confirm deletion")
                with col2:
                    if st.button("Delete", type="secondary", disabled=not confirm_delete):
                        try:
                            with st.spinner("Deleting..."):
                                index.delete(namespace=doc_to_delete, delete_all=True)
                                # Remove custom name if exists
                                if doc_to_delete in st.session_state.policy_display_names:
                                    del st.session_state.policy_display_names[doc_to_delete]
                                st.success("✅ Deleted successfully")
                                time.sleep(2)
                                st.cache_resource.clear()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Clear all section
                st.markdown("---")
                st.subheader("⚠️ Clear All Documents")
                confirm_all = st.checkbox("I understand this will delete ALL data permanently")
                if st.button("🗑️ CLEAR ALL", type="secondary", disabled=not confirm_all):
                    try:
                        with st.spinner("Clearing all..."):
                            index.delete(delete_all=True)
                            st.session_state.policy_display_names = {}
                            st.success("✅ All documents cleared!")
                            time.sleep(2)
                            st.cache_resource.clear()
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.info("No documents uploaded yet")
        
        with tab3:
            st.subheader("Chat Preview")
            st.caption("Test the chatbot functionality")
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input for testing
            if prompt := st.chat_input("Test a question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Searching..."):
                        context = get_relevant_context(prompt)
                        if context is None:
                            response = "Database connection error."
                        elif context == "":
                            response = "No relevant information found in the documents."
                        else:
                            response = generate_response(prompt, context)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

# USER VIEW
else:
    # Clean header for users
    st.title("💼 Finance Policy Assistant")
    st.markdown("Ask me anything about company finance policies in English or Bahasa Malaysia!")
    
    # Sidebar for users
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you find information about company finance policies.
        
        **How to use:**
        - Type your question in any language
        - Get instant answers from policy documents
        - Available 24/7
        """)
        
        # Show system status
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                if stats['total_vector_count'] > 0:
                    st.success("✅ System ready")
                    
                    # Show available documents with custom display names
                    if stats.get('namespaces'):
                        st.markdown("**📚 Available policies:**")
                        for ns in stats['namespaces'].keys():
                            if ns:
                                display_name = get_policy_display_name(ns)
                                st.markdown(f"• {display_name}")
                else:
                    st.warning("⚠️ No policies available")
                    st.info("Please contact IT support")
                    
            except Exception as e:
                st.error("❌ System offline")
                st.info("Please contact IT support")
        
        # Admin login (hidden in expander)
        with st.expander("🔐 Admin Access"):
            admin_password = st.text_input("Password", type="password", key="admin_pass")
            if st.button("Login"):
                if admin_password == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Check system configuration
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        st.error("⚠️ System not properly configured. Please contact IT support.")
        st.stop()
    
    # Chat input
    if prompt := st.chat_input("Ask about finance policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                context = get_relevant_context(prompt)
                
                if context is None:
                    response = "I couldn't connect to the document database. Please try again or contact IT support."
                elif context == "":
                    response = f"""I couldn't find specific information about "{prompt}" in the policy documents. 

Please try:
- Rephrasing your question
- Using different keywords
- Asking about these common topics:
  • Purchase orders and procurement
  • Reimbursement procedures
  • Budget approval processes
  • Asset management
  • Financial reporting requirements"""
                else:
                    response = generate_response(prompt, context)
                
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>Finance Policy Assistant • Powered by AI</div>", 
        unsafe_allow_html=True
    )
