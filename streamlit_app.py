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
    query_params = st.query_params
    if query_params.get("admin") == "true":
        return True
    if st.session_state.get("admin_authenticated"):
        return True
    return False

# Title and description
st.title("💼 Finance Policy Assistant")
st.markdown("Ask me anything about company finance policies in English or Bahasa Malaysia!")

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
            
            # Create index if it doesn't exist
            index_name = "finance-policy"
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
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
                
                # Debug info
                st.write("**Debug Info:**")
                st.write(f"Index stats: {stats}")
                
                # Show namespaces (different PDFs)
                if stats.get('namespaces'):
                    st.subheader("Uploaded Documents")
                    for ns, ns_stats in stats['namespaces'].items():
                        if ns:
                            st.text(f"📄 {ns}: {ns_stats['vector_count']} chunks")
                        else:
                            st.text(f"📄 Default namespace: {ns_stats['vector_count']} chunks")
                
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
        
        # PDF upload section
        st.subheader("Upload Policy Documents")
        uploaded_file = st.file_uploader(
            "Choose PDF file", 
            type="pdf",
            help="Upload finance policy documents."
        )
        
        if st.button("Process PDF", type="primary") and uploaded_file:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load and process PDF
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    st.write(f"Loaded {len(documents)} pages from PDF")
                    
                    # Add metadata
                    for i, doc in enumerate(documents):
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'page': i + 1,
                            'upload_date': datetime.now().isoformat(),
                            'total_pages': len(documents)
                        })
                    
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    st.write(f"Split into {len(chunks)} chunks")
                    
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Add to vector store WITHOUT namespace (simpler approach)
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name="finance-policy"
                        # Removed namespace to avoid complications
                    )
                    
                    # Clean up
                    os.unlink(tmp_file_path)
                    
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    st.info(f"📊 Created {len(chunks)} searchable chunks from {len(documents)} pages")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    if 'tmp_file_path' in locals():
                        os.unlink(tmp_file_path)

# Regular User Sidebar
else:
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you find information about company finance policies.
        
        **How to use:**
        - Type your question in any language
        - Get instant answers from policy documents
        """)
        
        # Show system status with more details
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                if stats['total_vector_count'] > 0:
                    st.success(f"✅ System ready")
                    st.info(f"📊 {stats['total_vector_count']} document chunks available")
                    
                    # Show which documents are available
                    if stats.get('namespaces'):
                        st.write("**Available documents:**")
                        for ns, ns_stats in stats['namespaces'].items():
                            if ns:
                                st.text(f"• {ns}")
                            else:
                                st.text(f"• Policy documents ({ns_stats['vector_count']} chunks)")
                else:
                    st.warning("⚠️ No policies uploaded yet")
                    st.info("Contact admin to upload policy documents")
            except Exception as e:
                st.error(f"❌ System connection error: {str(e)}")
        
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

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get relevant context with better debugging
def get_relevant_context(query, k=5):
    try:
        if PINECONE_API_KEY:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize vector store
            vectorstore = PineconeVectorStore(
                index_name="finance-policy",
                embedding=embeddings
            )
            
            # Check if index has any data
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index("finance-policy")
            stats = index.describe_index_stats()
            
            if stats['total_vector_count'] == 0:
                return None  # No documents uploaded
            
            # Perform search
            docs = vectorstore.similarity_search_with_score(query, k=k)
            
            if not docs:
                return ""  # No relevant documents found
            
            context_parts = []
            for doc, score in docs:
                source = doc.metadata.get('source_file', 'Policy Document')
                page = doc.metadata.get('page', 'Unknown')
                context_parts.append(
                    f"[Source: {source}, Page: {page}, Score: {score:.3f}]\n{doc.page_content}\n"
                )
            
            return "\n---\n".join(context_parts)
        return None
    except Exception as e:
        # Show error in admin mode only
        if is_admin():
            st.error(f"Error retrieving context: {str(e)}")
        return None

# Function to generate response
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        chat_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages[-6:]
        ])
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context.

Context from company policies:
{context}

Recent conversation:
{chat_history}

User Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say "I couldn't find specific information about that in the policy documents"
- Be concise but thorough
- Respond in the same language as the question

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

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
            
            if context is None:  # No documents at all
                response = """I couldn't find any policy documents in the system. 

**Next steps:**
1. Contact your IT administrator to upload policy documents
2. Or access admin mode to upload documents yourself

The system is working, but no policy documents have been uploaded yet."""
            elif context == "":  # Documents exist but no matches
                response = f"""I found policy documents in the system, but none seem to contain information about "{prompt}".

**Try:**
- Rephrasing your question
- Using different keywords
- Asking about general policy topics

If you believe this information should be available, contact your administrator."""
            else:  # Found relevant content
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Powered by AI*")
