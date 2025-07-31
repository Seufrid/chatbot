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
                
                # Show namespaces
                if stats.get('namespaces'):
                    st.subheader("Uploaded Documents")
                    for ns, ns_stats in stats['namespaces'].items():
                        if ns:
                            display_name = ns.replace('_', ' ').replace('(', '').replace(')', '')
                            st.text(f"📄 {display_name}: {ns_stats['vector_count']} chunks")
                
                # Test search button
                if st.button("Test Search Function"):
                    st.write("Testing search with query: 'reimbursement'")
                    test_result = test_search_function("reimbursement")
                    st.write(f"Test result: {test_result}")
                
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
        
        # PDF upload section
        st.subheader("Upload Policy Documents")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
        
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
                    
                    # Create embeddings
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Add to vector store with namespace
                    namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name="finance-policy",
                        namespace=namespace
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
        """)
        
        # Show system status
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                if stats['total_vector_count'] > 0:
                    st.success(f"✅ System ready")
                    st.info(f"📊 {stats['total_vector_count']} document chunks available")
                else:
                    st.warning("⚠️ No policies uploaded yet")
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

# Test function for debugging
def test_search_function(query):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        # Get embedding for query
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        # Try direct Pinecone query first
        results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            include_values=False
        )
        
        if results['matches']:
            return f"Found {len(results['matches'])} matches using direct query"
        else:
            return "No matches found with direct query"
            
    except Exception as e:
        return f"Error in test: {str(e)}"

# SIMPLIFIED search function
def get_relevant_context(query, k=3):
    try:
        if not PINECONE_API_KEY:
            return None
            
        # Direct Pinecone query approach
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        # Get embedding for the query
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        # Search directly in Pinecone across all namespaces
        all_results = []
        
        # Get stats to find namespaces
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        
        # Search each namespace
        for namespace in namespaces:
            if namespace:  # Skip empty namespace
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=k,
                        include_metadata=True,
                        include_values=False,
                        namespace=namespace
                    )
                    
                    for match in results['matches']:
                        all_results.append({
                            'score': match['score'],
                            'text': match['metadata'].get('text', ''),
                            'source': match['metadata'].get('source_file', namespace),
                            'page': match['metadata'].get('page', 'Unknown')
                        })
                        
                except Exception as e:
                    continue
        
        # Also try without namespace (default namespace)
        try:
            results = index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                include_values=False
            )
            
            for match in results['matches']:
                all_results.append({
                    'score': match['score'],
                    'text': match['metadata'].get('text', ''),
                    'source': match['metadata'].get('source_file', 'Policy Document'),
                    'page': match['metadata'].get('page', 'Unknown')
                })
                
        except Exception as e:
            pass
        
        if not all_results:
            return ""
        
        # Sort by score (higher is better for Pinecone)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take best results
        best_results = all_results[:k]
        
        context_parts = []
        for result in best_results:
            if result['text']:  # Only include if we have text
                context_parts.append(
                    f"[Source: {result['source']}, Page: {result['page']}, Score: {result['score']:.3f}]\n{result['text']}\n"
                )
        
        return "\n---\n".join(context_parts) if context_parts else ""
        
    except Exception as e:
        if is_admin():
            st.error(f"Search error: {str(e)}")
        return None

# Function to generate response
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context.

Context from company policies:
{context}

User Question: {query}

Instructions:
- Answer based on the provided context from the policy documents
- Be specific and cite the source document when possible
- If the answer isn't in the context, say so politely
- Be concise but thorough
- Include relevant policy references when available

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
            
            if context is None:
                response = "I couldn't connect to the document database. Please try again."
            elif context == "":
                response = f"""I searched through all policy documents but couldn't find specific information about "{prompt}". 

**Available documents:**
- Asset Management Policy
- Financial Policies and Procedures  
- Purchasing Policies
- Debt Management and Write-Off Policy

Try rephrasing your question or asking about topics like procurement, asset management, or financial procedures."""
            else:
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Powered by AI*")
