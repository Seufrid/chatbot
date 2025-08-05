# 💼 Finance Policy Assistant

A multilingual AI-powered chatbot that helps finance department employees quickly find information about company policies. The assistant supports both English and Bahasa Malaysia, making policy information accessible to all staff members.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Bilingual Support**: Seamlessly handles queries in both English and Bahasa Malaysia
- **Smart Document Search**: Uses advanced vector embeddings to find relevant policy information
- **Policy Citation**: Provides accurate references to specific policy sections
- **Admin Dashboard**: Upload, manage, and customize policy documents
- **Real-time Responses**: Powered by Google's Gemini AI for natural conversations
- **Secure Access**: Password-protected admin interface for document management

## 🚀 Demo

The application is currently deployed for internal testing. Access requires proper credentials.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: Google Gemini 2.5 Flash
- **Vector Database**: Pinecone
- **Embeddings**: Multilingual Sentence Transformers
- **Document Processing**: LangChain + PyPDF
- **Language**: Python 3.8+

## 📋 Prerequisites

Before running this application, you'll need:

1. **API Keys**:
   - Google AI (Gemini) API key - [Get it here](https://makersuite.google.com/app/apikey)
   - Pinecone API key - [Sign up for free](https://www.pinecone.io/)

2. **Python Environment**:
   - Python 3.8 or higher
   - pip package manager

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Seufrid/finance-policy-chatbot.git
   cd finance-policy-chatbot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Streamlit secrets**
   
   Create a `.streamlit/secrets.toml` file in your project root:
   ```toml
   GOOGLE_API_KEY = "your-google-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   ADMIN_PASSWORD = "your-secure-admin-password"
   ```

5. **Create Pinecone Index**
   - Log into your Pinecone dashboard
   - Create a new index named `finance-policy`
   - Set dimensions to `384` (for the multilingual model)
   - Choose your preferred environment

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the application**
   - Open your browser to `http://localhost:8501`
   - For admin access, append `?admin=true` to the URL

## 📚 Usage Guide

### For Employees

1. Simply type your question about finance policies in the chat
2. Ask in either English or Bahasa Malaysia
3. The assistant will search through policy documents and provide relevant information with citations

**Example queries:**
- "How do I claim medical expenses?"
- "Bagaimana proses tuntutan perbelanjaan?"
- "What are the limits for petty cash?"
- "Siapa yang perlu meluluskan pembelian asset?"

### For Administrators

1. Access admin panel: `http://localhost:8501?admin=true`
2. Enter the admin password
3. **Upload documents**: 
   - Click "Upload Documents" tab
   - Select PDF policy documents
   - Documents are automatically processed and indexed
4. **Manage documents**:
   - View all uploaded policies
   - Delete individual documents
   - Customize display names for better user experience
5. **Test the system**:
   - Use the chat preview to test responses
   - Monitor system status and usage

## 🔐 Security Considerations

- Store API keys securely in Streamlit secrets or environment variables
- Never commit sensitive credentials to version control
- Use strong admin passwords
- Regularly rotate API keys
- Monitor usage to prevent abuse

## 🏗️ Project Structure

```
finance-policy-chatbot/
├── streamlit_app.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore              # Git ignore file
└── .streamlit/             # Streamlit configuration
    └── secrets.toml        # API keys (do not commit!)
```

## 🤝 Contributing

This is an internal project for the hospital's finance department. For any improvements or bug reports, please contact the IT department.

### Development Guidelines

1. Test all changes thoroughly before deployment
2. Ensure bilingual support is maintained
3. Document any new features or changes
4. Follow Python PEP 8 style guidelines

## 🐛 Troubleshooting

### Common Issues

1. **"System not properly configured" error**
   - Check if all API keys are correctly set in secrets.toml
   - Verify Pinecone index exists and is named correctly

2. **PDF upload fails**
   - Ensure PDF contains searchable text (not scanned images)
   - Check file size limits
   - Verify Pinecone quota hasn't been exceeded

3. **No search results found**
   - Confirm documents have been uploaded successfully
   - Check if query terms match document content
   - Try rephrasing the question

4. **Language detection issues**
   - The system auto-detects language based on keywords
   - If detection fails, try adding more context to your question

## 📈 Future Enhancements

- [ ] OCR support for scanned PDFs
- [ ] Integration with Microsoft Teams
- [ ] Advanced analytics dashboard
- [ ] Multi-department support
- [ ] Voice input capabilities
- [ ] Mobile-responsive design
- [ ] Export conversation history
- [ ] Automated policy updates

## 📄 License

This project is proprietary software developed for internal use at the hospital. All rights reserved.

## 👥 Contact

For technical support or questions:
- **Developer**: [Your Name]
- **Department**: Finance Department IT Support
- **Email**: [your.email@hospital.com]

## 🙏 Acknowledgments

- Finance Department team for policy documents and requirements
- Hospital IT Department for infrastructure support
- Open source community for the amazing tools that made this possible

---

**Note**: This chatbot is designed for internal use only. Do not share access credentials or expose the application to external networks without proper security review.
