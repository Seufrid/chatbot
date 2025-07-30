import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
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

# Sidebar for setup
with st.sidebar:
    st.header("Setup")
    
    # API Key input
    api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
    
    # PDF upload for initial setup (one-time)
    st.subheader("Initial Setup")
    uploaded_file = st.file_uploader("Upload Policy PDF (one-time setup)", type="pdf")
    
    if st.button("Process PDF") and uploaded_file and api_key:
        with st.spinner("Processing PDF... This may take a few minutes."):

            try:
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
                
                # Create embeddings (using free HuggingFace embeddings)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Create vector store
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                vectorstore.persist()
                
                # Clean up temp file
                os.unlink(tmp_file_path)
                
                st.success("✅ PDF processed successfully! You can now ask questions.")
                st.session_state.pdf_processed = True
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Increase the context window size by keeping track of more messages
MAX_HISTORY = 10  # You can adjust this number to your needs
if len(st.session_state.messages) > MAX_HISTORY:
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to get relevant context from vector store
def get_relevant_context(query, k=3):
    try:
        # Load existing vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=k)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Include recent chat messages for better context
        chat_context = "\n".join([message["content"] for message in st.session_state.messages])
        
        # Combine document context and chat context
        full_context = context + "\n\n" + chat_context
        return full_context
    except:
        return ""

# Function to generate response using Gemini
def generate_response(query, context):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""You are a helpful assistant for finance department employees. Answer questions about company finance policies based on the provided context. You can respond in both English and Bahasa Malaysia based on the user's preference.

Context from company policies:
{context}

User Question: {query}

Instructions:
- Answer based on the provided context
- If the answer isn't in the context, say so politely
- Be concise but helpful
- Respond in the same language as the question when possible
- If asked in Bahasa Malaysia, respond in Bahasa Malaysia

Answer:"""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Chat input
if prompt := st.chat_input("Ask about finance policies... (Tanya tentang polisi kewangan...)"):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar first.")
        st.stop()
    
    # Check if PDF has been processed
    if not os.path.exists("./chroma_db"):
        st.error("Please upload and process the policy PDF first using the sidebar.")
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
            
            # Generate response
            response = generate_response(prompt, context)
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("*Finance Policy Assistant - Ask questions about company policies in English or Bahasa Malaysia*")
