import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
import os
import tempfile

# Configure page
st.set_page_config(
    page_title="Finance Policy Assistant",
    page_icon="💼",
    layout="wide"
)

# Title and description
st.title("💼 Finance Policy Assistant")
st.markdown("Ask me anything about company finance policies in English or Bahasa Malaysia!")

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    # Get Pinecone credentials from Streamlit secrets or sidebar
    if "PINECONE_API_KEY" in st.secrets and "PINECONE_ENV" in st.secrets:
        pc_api_key = st.secrets["PINECONE_API_KEY"]
        pc_env = st.secrets["PINECONE_ENV"]
        return pc_api_key, pc_env
    return None, None

# Sidebar for setup
with st.sidebar:
    st.header("Setup")
    
    # API Key inputs
    api_key = st.text_input("Enter Google Gemini API Key", type="password",
                           value=st.secrets.get("GOOGLE_API_KEY", ""))
    
    # Pinecone setup (only show if not in secrets)
    pc_api_key, pc_env = init_pinecone()
    
    if not pc_api_key:
        st.subheader("Pinecone Setup")
        pc_api_key = st.text_input("Pinecone API Key", type="password")
        pc_env = st.text_input("Pinecone Environment", placeholder="e.g., us-east-1-aws")
    
    if api_key:
        genai.configure(api_key=api_key)
    
    # PDF upload for initial setup
    st.subheader("Upload Policy Document")
    uploaded_file = st.file_uploader("Upload Policy PDF", type="pdf")
    
    if st.button("Process PDF") and uploaded_file and api_key and pc_api_key and pc_env:
        with st.spinner("Processing PDF... This may take a few minutes."):
            try:
                # Initialize Pinecone
                pc = pinecone.Pinecone(api_key=pc_api_key)
                index = pc.Index("finance-policy")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                
                # Load and process PDF
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()
                
                # Split documents into chunks
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
                
                # Create vector store in Pinecone
                vectorstore = Pinecone.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name="finance-policy"
                )
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                st.success(f"✅ PDF processed successfully! {len(chunks)} chunks created.")
                st.session_state.pdf_processed = True
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Check index status
    if pc_api_key and pc_env:
        try:
            pc = pinecone.Pinecone(api_key=pc_api_key)
            index = pc.Index("finance-policy")
            stats = index.describe_index_stats()
            if stats['total_vector_count'] > 0:
                st.success(f"✅ Vector database ready: {stats['total_vector_count']} vectors")
            else:
                st.warning("⚠️ No vectors in database. Please upload a PDF.")
        except:
            st.error("❌ Could not connect to Pinecone")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep track of more messages for context
MAX_HISTORY = 10
if len(st.session_state.messages) > MAX_HISTORY:
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get relevant context from Pinecone
def get_relevant_context(query, k=3):
    try:
        pc_api_key, pc_env = init_pinecone()
        if not pc_api_key:
            # Try to get from sidebar inputs
            pc_api_key = st.session_state.get('pc_api_key')
            pc_env = st.session_state.get('pc_env')
        
        if pc_api_key and pc_env:
            # Initialize Pinecone
            pc = pinecone.Pinecone(api_key=pc_api_key)
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load existing vector store
            vectorstore = Pinecone.from_existing_index(
                index_name="finance-policy",
                embedding=embeddings
            )
            
            # Search for relevant documents
            docs = vectorstore.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            return context
        return ""
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# Function to generate response using Gemini
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Include chat history for better context
        chat_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in st.session_state.messages[-4:]  # Last 4 messages
        ])
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context. You can respond in both English and Bahasa Malaysia based on the user's preference.

Context from company policies:
{context}

Recent conversation:
{chat_history}

User Question: {query}

Instructions:
- Answer based on the provided context
- If the answer isn't in the context, say so politely
- Be concise but helpful
- Respond in the same language as the question when possible
- If asked in Bahasa Malaysia, respond in Bahasa Malaysia
- Consider the conversation history for context

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Chat input
if prompt := st.chat_input("Ask about finance policies... (Tanya tentang polisi kewangan...)"):
    # Check if API keys are provided
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar first.")
        st.stop()
    
    if not pc_api_key or not pc_env:
        st.error("Please enter your Pinecone credentials in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get relevant context
            context = get_relevant_context(prompt)
            
            if not context:
                response = "I couldn't find any relevant information in the policy documents. Please make sure you've uploaded the policy PDF first."
            else:
                # Generate response
                response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Ask questions about company policies in English or Bahasa Malaysia*")
