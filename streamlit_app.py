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

# Test function for debugging
def test_search_function(query):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("finance-policy")
        
        # Get stats first
        stats = index.describe_index_stats()
        
        # Get embedding for query
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embeddings.embed_query(query)
        
        # Try searching each namespace
        results_summary = []
        
        for namespace in stats.get('namespaces', {}):
            if namespace:
                try:
                    results = index.query(
                        vector=query_embedding,
                        top_k=1,  # Just get one for testing
                        include_metadata=True,
                        include_values=False,
                        namespace=namespace
                    )
                    
                    results_summary.append(f"Namespace '{namespace}': {len(results['matches'])} matches")
                    
                    # Show first match details
                    if results['matches']:
                        first_match = results['matches'][0]
                        results_summary.append(f"  - Score: {first_match['score']:.3f}")
                        results_summary.append(f"  - Has text field: {'text' in first_match.get('metadata', {})}")
                        
                        # Show a snippet of the text content
                        text_content = first_match.get('metadata', {}).get('text', '')
                        if text_content:
                            snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
                            results_summary.append(f"  - Text snippet: {snippet}")
                        else:
                            results_summary.append(f"  - No text content found")
                        
                except Exception as e:
                    results_summary.append(f"Namespace '{namespace}': Error - {str(e)}")
        
        return "\n".join(results_summary)
            
    except Exception as e:
        return f"Error in test: {str(e)}"

# FIXED search function
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
                        # Extract text from metadata
                        text_content = match['metadata'].get('text', '')
                        
                        # Only include if we have actual text content
                        if text_content and len(text_content.strip()) > 20:
                            all_results.append({
                                'score': match['score'],
                                'text': text_content,
                                'source': match['metadata'].get('source_file', namespace.replace('_', ' ')),
                                'page': match['metadata'].get('page', 'Unknown')
                            })
                        
                except Exception as e:
                    continue
        
        if not all_results:
            return ""
        
        # Sort by score (higher is better for Pinecone)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take best results
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
        model = genai.GenerativeModel('gemini-pro')
        
        chat_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages[-4:]
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
- Respond in the same language as the question

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

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
                
                # Show usage bar
                usage_percent = (stats['total_vector_count'] / 100000) * 100
                st.progress(usage_percent / 100, text=f"Usage: {stats['total_vector_count']:,}/100,000 vectors ({usage_percent:.1f}%)")
                
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
                    st.code(test_result)
                
                # Test context extraction
                if st.button("Test Context Extraction"):
                    st.write("Testing full context extraction...")
                    context = get_relevant_context("reimbursement", k=2)
                    if context:
                        st.write("✅ Context found!")
                        st.text_area("Context Preview", context[:500] + "..." if len(context) > 500 else context, height=200)
                    else:
                        st.write("❌ No context returned")
                
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
        
        # PDF upload section with improved handling
        st.subheader("Upload Policy Documents")
        
        # Show current usage
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                remaining = 100000 - stats['total_vector_count']
                st.info(f"📊 Available space: ~{remaining//1000}k vectors remaining")
            except:
                pass

        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

        if st.button("Process PDF", type="primary") and uploaded_file:
            # Check if file already exists
            existing_namespaces = list(stats.get('namespaces', {}).keys())
            proposed_namespace = uploaded_file.name.replace('.pdf', '').replace(' ', '_')
            
            if any(proposed_namespace in ns for ns in existing_namespaces):
                st.warning(f"⚠️ A document with similar name already exists. This will add to the existing document.")
            
            with st.spinner(f"Processing {uploaded_file.name}..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Save file
                    status_text.text("📁 Saving uploaded file...")
                    progress_bar.progress(10)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    st.write(f"📁 File size: {len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB")
                    
                    # Step 2: Load PDF
                    status_text.text("📄 Loading PDF content...")
                    progress_bar.progress(20)
                    
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    st.write(f"📄 Loaded {len(documents)} pages from PDF")
                    
                    if len(documents) == 0:
                        st.error("❌ No content found in PDF. Please check if the PDF is valid.")
                        os.unlink(tmp_file_path)
                        st.stop()
                    
                    # Step 3: Add metadata
                    status_text.text("🏷️ Adding metadata...")
                    progress_bar.progress(30)
                    
                    for i, doc in enumerate(documents):
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'page': i + 1,
                            'upload_date': datetime.now().isoformat(),
                            'total_pages': len(documents)
                        })
                    
                    # Step 4: Split documents
                    status_text.text("✂️ Splitting into chunks...")
                    progress_bar.progress(40)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    st.write(f"✂️ Split into {len(chunks)} chunks")
                    
                    # Step 5: Check limits
                    status_text.text("🔍 Checking vector limits...")
                    progress_bar.progress(50)
                    
                    new_total = stats['total_vector_count'] + len(chunks)
                    if new_total > 100000:
                        st.error(f"❌ Cannot upload: Would exceed free tier limit ({new_total:,}/100,000 vectors)")
                        st.info(f"💡 Try reducing chunk size or removing old documents first")
                        os.unlink(tmp_file_path)
                        st.stop()
                    
                    # Step 6: Create embeddings
                    status_text.text("🧠 Creating embeddings...")
                    progress_bar.progress(60)
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    # Step 7: Upload to Pinecone
                    status_text.text("📤 Uploading to vector database...")
                    progress_bar.progress(70)
                    
                    namespace = proposed_namespace
                    st.write(f"📤 Uploading to namespace: {namespace}")
                    
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        index_name="finance-policy",
                        namespace=namespace
                    )
                    
                    # Step 8: Verify upload
                    status_text.text("🔍 Verifying upload...")
                    progress_bar.progress(90)
                    
                    # Wait for Pinecone to update
                    time.sleep(3)
                    
                    # Clear cache and get fresh stats
                    st.cache_resource.clear()
                    pc = init_pinecone()
                    index = pc.Index("finance-policy")
                    new_stats = index.describe_index_stats()
                    new_count = new_stats['total_vector_count']
                    
                    # Step 9: Complete
                    status_text.text("✅ Upload completed!")
                    progress_bar.progress(100)
                    
                    if new_count > stats['total_vector_count']:
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        st.info(f"📊 Added {len(chunks)} chunks ({stats['total_vector_count']:,} → {new_count:,} total)")
                        
                        # Clean up
                        os.unlink(tmp_file_path)
                        st.write("🧹 Cleaned up temporary files")
                        
                        # Force refresh after a delay
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Upload verification failed - vector count didn't increase")
                        st.write(f"Expected: {stats['total_vector_count'] + len(chunks):,}, Got: {new_count:,}")
                        st.info("💡 This might be a temporary delay. Try refreshing the page in a few seconds.")
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.write("Full error details:")
                    st.code(str(e))
                    
                    # Clean up on error
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                        st.write("🧹 Cleaned up temporary files after error")
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

        # Add management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Stats", type="secondary"):
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            if st.button("🗑️ Manage Documents", type="secondary"):
                st.session_state.show_document_manager = not st.session_state.get('show_document_manager', False)
        
        # Document management section
        if st.session_state.get('show_document_manager', False):
            st.subheader("🗑️ Document Management")
            
            if stats.get('namespaces'):
                doc_to_delete = st.selectbox(
                    "Select document to delete",
                    [ns for ns in stats['namespaces'].keys() if ns],
                    format_func=lambda x: x.replace('_', ' ')
                )
                
                if st.button("Delete Selected Document", type="secondary"):
                    if st.checkbox("I understand this will permanently delete this document"):
                        try:
                            index.delete(namespace=doc_to_delete, delete_all=True)
                            st.success(f"Deleted {doc_to_delete.replace('_', ' ')}")
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting: {str(e)}")
                
                # Clear all vectors
                with st.expander("⚠️ Danger Zone"):
                    st.warning("These actions cannot be undone!")
                    if st.button("Clear ALL Documents", type="secondary"):
                        if st.checkbox("I understand this will delete ALL data"):
                            try:
                                index.delete(delete_all=True)
                                st.success("All documents cleared!")
                                time.sleep(2)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error clearing documents: {str(e)}")

# Regular User Sidebar
else:
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you find information about company finance policies.
        
        **Available documents:**
        - Asset Management Policy
        - Financial Policies and Procedures  
        - Purchasing Policies
        - Debt Management and Write-Off Policy
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

Try rephrasing your question or asking about topics like:
- Asset management procedures
- Financial approval processes
- Purchasing and procurement policies
- Debt management procedures"""
            else:
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Powered by AI*")
