import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import os
import tempfile
from datetime import datetime

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
    
    # Method 2: Check if admin authenticated in session
    if st.session_state.get("admin_authenticated"):
        return True
    
    return False

# Title and description
st.title("💼 Finance Policy Assistant")
st.markdown("Ask me anything about company finance policies in English or Bahasa Malaysia!")

# Initialize API keys from secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")  # Change this!

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    if PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Create index if it doesn't exist
            index_name = "finance-policy"
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=384,  # for all-MiniLM-L6-v2
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            return pc
        except Exception as e:
            st.error(f"Pinecone initialization error: {str(e)}")
            return None
    return None

# Admin Panel
if is_admin():
    with st.sidebar:
        st.header("🔐 Admin Panel")
        
        # Logout button
        if st.button("Logout", type="secondary"):
            st.session_state.admin_authenticated = False
            st.query_params.clear()
            st.rerun()
        
        # Show API key status
        st.success("✅ Google API Key configured" if GOOGLE_API_KEY else "❌ Google API Key missing")
        st.success("✅ Pinecone API Key configured" if PINECONE_API_KEY else "❌ Pinecone API Key missing")
        
        # Initialize Pinecone and show stats
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Vectors", stats['total_vector_count'])
                with col2:
                    st.metric("Dimensions", stats['dimension'])
                
                # Show namespaces (different PDFs)
                if stats['namespaces']:
                    st.subheader("Uploaded Documents")
                    for ns, ns_stats in stats['namespaces'].items():
                        if ns:  # Skip empty namespace
                            st.text(f"📄 {ns}: {ns_stats['vector_count']} chunks")
                
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
        
        # PDF upload section
        st.subheader("Upload Policy Documents")
        uploaded_file = st.file_uploader(
            "Choose PDF file", 
            type="pdf",
            help="Upload finance policy documents. Each PDF will be processed and added to the knowledge base."
        )
        
        if st.button("Process PDF", type="primary") and uploaded_file:
            with st.spinner(f"Processing {uploaded_file.name}... This may take a few minutes."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load and process PDF
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    # Add metadata to each document
                    for i, doc in enumerate(documents):
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'page': i + 1,
                            'upload_date': datetime.now().isoformat(),
                            'total_pages': len(documents)
                        })
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    
                    # Add to vector store (using filename as namespace for organization)
                    namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name="finance-policy",
                        namespace=namespace
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    st.info(f"📊 Created {len(chunks)} searchable chunks from {len(documents)} pages")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)
        
        # Danger zone
        with st.expander("⚠️ Danger Zone"):
            st.warning("These actions cannot be undone!")
            
            # Delete specific document
            if stats.get('namespaces'):
                doc_to_delete = st.selectbox(
                    "Select document to delete",
                    [ns for ns in stats['namespaces'].keys() if ns]
                )
                if st.button("Delete Selected Document", type="secondary"):
                    if st.checkbox("I understand this will delete this document"):
                        try:
                            index.delete(namespace=doc_to_delete, delete_all=True)
                            st.success(f"Deleted {doc_to_delete}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting: {str(e)}")
            
            # Clear all vectors
            if st.button("Clear ALL Vectors", type="secondary"):
                if st.checkbox("I understand this will delete ALL data"):
                    try:
                        index.delete(delete_all=True)
                        st.success("All vectors cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing vectors: {str(e)}")

# Regular User Sidebar
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
        
        # Show system status for users
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                if stats['total_vector_count'] > 0:
                    st.success(f"✅ System ready with {len(stats.get('namespaces', {}))} policy documents")
                else:
                    st.warning("⚠️ No policies uploaded yet")
            except:
                st.error("❌ System connection error")
        
        # Admin login
        with st.expander("Admin Access"):
            admin_password = st.text_input("Admin Password", type="password", key="admin_pass")
            if st.button("Login as Admin"):
                if admin_password == ADMIN_PASSWORD:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep track of conversation context
MAX_HISTORY = 10
if len(st.session_state.messages) > MAX_HISTORY * 2:  # Keep more history
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get relevant context from Pinecone
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_relevant_context(query, k=5):  # Increased k for better context
    try:
        if PINECONE_API_KEY:
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Load existing vector store
            vectorstore = PineconeVectorStore(
                index_name="finance-policy",
                embedding=embeddings
            )
            
            # Search across all namespaces
            docs = vectorstore.similarity_search_with_score(query, k=k)
            
            # Format context with metadata
            context_parts = []
            for doc, score in docs:
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                context_parts.append(
                    f"[Source: {source}, Page: {page}, Relevance: {score:.2f}]\n{doc.page_content}\n"
                )
            
            return "\n---\n".join(context_parts)
        return ""
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# Function to generate response using Gemini
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Include chat history for better context
        chat_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages[-6:]  # Last 6 messages
        ])
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context. You can respond in both English and Bahasa Malaysia based on the user's preference.

Context from company policies:
{context}

Recent conversation:
{chat_history}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say "I couldn't find specific information about that in the policy documents"
- Be concise but thorough
- Include relevant policy references when available
- Respond in the same language as the question
- If asked in Bahasa Malaysia, respond in Bahasa Malaysia
- Consider the conversation history for context continuity
- If multiple sources provide information, synthesize them coherently

Answer:"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Check if system is properly configured
if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    st.error("⚠️ System not properly configured. Please contact IT support.")
    st.info("Admin: Add API keys to Streamlit secrets")
    st.stop()

# Chat input
if prompt := st.chat_input("Ask about finance policies... (Tanya tentang polisi kewangan...)"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            # Get relevant context
            context = get_relevant_context(prompt)
            
            if not context:
                response = """I couldn't find any relevant information in the policy documents. 

This could mean:
- No policy documents have been uploaded yet
- Your question doesn't match any content in the uploaded policies
- The system is having connection issues

Please try rephrasing your question or contact IT support if the problem persists."""
            else:
                # Generate response
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("*Finance Policy Assistant - Powered by AI*")
with col2:
    st.markdown(f"*{datetime.now().strftime('%Y-%m-%d')}*")
with col3:
    if is_admin():
        st.markdown("*🔐 Admin Mode*")
