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

# Get display name for a policy
def get_policy_display_name(namespace):
    if namespace in st.session_state.policy_display_names:
        return st.session_state.policy_display_names[namespace]
    # Default formatting if no custom name
    return namespace.replace('_', ' ').replace('(', '').replace(')', '').title()

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

# Search function
def get_relevant_context(query, k=6):
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
- Respond in the same language as the question

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Streamlined text validation function
def validate_pdf_text(documents):
    """Quick validation to ensure PDF has readable text"""
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
                            display_name = get_policy_display_name(ns)
                            st.text(f"📄 {display_name}: {ns_stats['vector_count']} chunks")
                
                # Policy Display Names Editor
                if stats.get('namespaces'):
                    with st.expander("📝 Edit Policy Names"):
                        st.caption("Customize how policies appear to users")
                        
                        for ns in stats['namespaces'].keys():
                            if ns:
                                current_name = get_policy_display_name(ns)
                                new_name = st.text_input(
                                    f"Display name for '{ns}':",
                                    value=current_name,
                                    key=f"name_{ns}",
                                    help=f"Original filename: {ns}"
                                )
                                if new_name != current_name and new_name.strip():
                                    st.session_state.policy_display_names[ns] = new_name.strip()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Names", use_container_width=True):
                                st.success("✅ Names updated!")
                                st.rerun()
                        with col2:
                            if st.button("🔄 Reset All", use_container_width=True):
                                st.session_state.policy_display_names = {}
                                st.rerun()
                
                # Test buttons
                if st.button("Test Search Function"):
                    test_result = test_search_function("reimbursement")
                    st.code(test_result)
                
                if st.button("Test Context Extraction"):
                    context = get_relevant_context("reimbursement", k=2)
                    if context:
                        st.write("✅ Context found!")
                        st.text_area("Context Preview", context[:500] + "..." if len(context) > 500 else context, height=200)
                    else:
                        st.write("❌ No context returned")
                
            except Exception as e:
                st.error(f"Error fetching stats: {str(e)}")
        
        # Streamlined PDF upload section
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
                    progress_bar.progress(15)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    st.write(f"📁 File size: {len(uploaded_file.getvalue()) / 1024 / 1024:.1f} MB")
                    
                    # Step 2: Load PDF
                    status_text.text("📄 Loading PDF content...")
                    progress_bar.progress(30)
                    
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    st.write(f"📄 Loaded {len(documents)} pages from PDF")
                    
                    if len(documents) == 0:
                        st.error("❌ No pages found in PDF. Please check if the PDF is valid.")
                        os.unlink(tmp_file_path)
                        st.stop()
                    
                    # Step 3: Validate text content
                    status_text.text("🔍 Validating text content...")
                    progress_bar.progress(40)
                    
                    has_valid_text, validation_message = validate_pdf_text(documents)
                    
                    if not has_valid_text:
                        st.error(f"❌ Cannot process PDF: Insufficient text content")
                        st.info(f"💡 Issue: {validation_message}")
                        st.info("🔧 Solution: Use an OCR tool to convert the PDF to text-searchable format first.")
                        os.unlink(tmp_file_path)
                        st.stop()
                    
                    st.write(f"✅ {validation_message}")
                    
                    # Step 4: Add metadata
                    status_text.text("🏷️ Adding metadata...")
                    progress_bar.progress(50)
                    
                    for i, doc in enumerate(documents):
                        doc.metadata.update({
                            'source_file': uploaded_file.name,
                            'page': i + 1,
                            'upload_date': datetime.now().isoformat(),
                            'total_pages': len(documents)
                        })
                    
                    # Step 5: Split documents
                    status_text.text("✂️ Splitting into chunks...")
                    progress_bar.progress(60)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    
                    chunks = text_splitter.split_documents(documents)
                    
                    # Filter out very short chunks
                    valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 20]
                    
                    st.write(f"✂️ Created {len(valid_chunks)} searchable chunks")
                    
                    # Step 6: Check limits
                    status_text.text("🔍 Checking vector limits...")
                    progress_bar.progress(70)
                    
                    new_total = stats['total_vector_count'] + len(valid_chunks)
                    if new_total > 100000:
                        st.error(f"❌ Cannot upload: Would exceed free tier limit ({new_total:,}/100,000 vectors)")
                        os.unlink(tmp_file_path)
                        st.stop()
                    
                    # Step 7: Create embeddings and upload
                    status_text.text("🧠 Creating embeddings and uploading...")
                    progress_bar.progress(80)
                    
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    namespace = proposed_namespace
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=valid_chunks,
                        embedding=embeddings,
                        index_name="finance-policy",
                        namespace=namespace
                    )
                    
                    # Step 8: Verify upload
                    status_text.text("🔍 Verifying upload...")
                    progress_bar.progress(90)
                    
                    time.sleep(3)  # Wait for Pinecone to update
                    
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
                        st.info(f"📊 Added {len(valid_chunks)} chunks ({stats['total_vector_count']:,} → {new_count:,} total)")
                        
                        # Clean up
                        os.unlink(tmp_file_path)
                        
                        # Force refresh
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Upload verification failed - vector count didn't increase")
                        st.info("💡 This might be a temporary delay. Try refreshing the page in a few minutes.")
                
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    
                    # Clean up on error
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

        # Management buttons
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
                # Individual document deletion
                st.write("**Delete Individual Document:**")
                doc_to_delete = st.selectbox(
                    "Select document to delete",
                    [ns for ns in stats['namespaces'].keys() if ns],
                    format_func=lambda x: get_policy_display_name(x)
                )
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    delete_single = st.button("Delete Selected", type="secondary", key="delete_single")
                with col2:
                    confirm_single = st.checkbox("Confirm deletion", key="confirm_single")
                
                if delete_single and confirm_single:
                    try:
                        with st.spinner(f"Deleting {doc_to_delete}..."):
                            index.delete(namespace=doc_to_delete, delete_all=True)
                            # Remove custom name if exists
                            if doc_to_delete in st.session_state.policy_display_names:
                                del st.session_state.policy_display_names[doc_to_delete]
                            st.success(f"✅ Deleted {get_policy_display_name(doc_to_delete)}")
                            time.sleep(3)
                            st.cache_resource.clear()
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting: {str(e)}")
                
                st.markdown("---")
            
            # Clear all documents
            st.write("**⚠️ DANGER ZONE - Clear All Documents:**")
            st.warning("This will permanently delete ALL policy documents from the database!")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                clear_all = st.button("🗑️ CLEAR ALL", type="secondary", key="clear_all")
            with col2:
                confirm_all = st.checkbox("I understand this will delete ALL data permanently", key="confirm_all")
            
            if clear_all and confirm_all:
                try:
                    with st.spinner("Clearing all documents..."):
                        # Force clear cache first
                        st.cache_resource.clear()
                        
                        # Reinitialize connection
                        pc = Pinecone(api_key=PINECONE_API_KEY)
                        index = pc.Index("finance-policy")
                        
                        # Delete all vectors
                        index.delete(delete_all=True)
                        
                        # Clear custom names
                        st.session_state.policy_display_names = {}
                        
                        st.success("✅ All documents cleared!")
                        
                        # Wait and refresh
                        time.sleep(3)
                        st.cache_resource.clear()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error clearing all documents: {str(e)}")
                    st.write("Try the manual Python script method instead.")

# Regular User Sidebar
else:
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you find information about company finance policies.
        
        **How to use:**
        - Type your question in any language
        - Get instant answers from policy documents
        - Available 24/7
        """)
        
        # Show system status and available documents dynamically
        pc = init_pinecone()
        if pc:
            try:
                index = pc.Index("finance-policy")
                stats = index.describe_index_stats()
                
                if stats['total_vector_count'] > 0:
                    st.success(f"✅ System ready")
                    st.info(f"📊 {stats['total_vector_count']} document chunks available")
                    
                    # Show actually available documents with custom names
                    if stats.get('namespaces'):
                        st.markdown("**Available documents:**")
                        for ns in stats['namespaces'].keys():
                            if ns:
                                display_name = get_policy_display_name(ns)
                                st.markdown(f"• {display_name}")
                    
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
- Stock management procedures
- Debt management procedures"""
            else:
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Powered by AI*")
