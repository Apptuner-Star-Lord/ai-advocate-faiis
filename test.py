import streamlit as st
import sqlite3, uuid, json, re, os
from datetime import datetime
from time import sleep
from speech_integration import VoiceChat
# import pytesseract
import easyocr
from PIL import Image
import pdfplumber
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import HumanMessage, AIMessage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Configure page
st.set_page_config(
    page_title="Legal Adviser Bot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for Claude-like UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styling */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Chat message styling - Claude-like */
.stChatMessage {
    background-color: transparent;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    padding: 1rem;
}

.stChatMessage[data-testid="chat-message-user"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 20%;
    border-radius: 18px 18px 4px 18px;
}

.stChatMessage[data-testid="chat-message-assistant"] {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 18px 18px 18px 4px;
    margin-right: 10%;
}

/* Suggestion pills styling */
.suggestion-pill {
    display: inline-block;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 16px;
    margin: 4px 8px 4px 0;
    border-radius: 20px;
    border: none;
    cursor: pointer;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.3s ease;
    text-decoration: none;
}

.suggestion-pill:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}

/* Document analysis card */
.doc-analysis-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
}

.doc-analysis-card h3 {
    margin-top: 0;
    font-weight: 600;
}

/* Action buttons container */
.action-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 1rem 0;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
}

/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background-color: white;
    color: #495057;
    border: 1.5px solid #dee2e6;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #f8f9fa;
    border-color: #667eea;
    color: #667eea;
    transform: translateY(-1px);
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    border-right: 1px solid #e9ecef;
}

/* Chat input styling */
.stChatInputContainer {
    border: 2px solid #e9ecef;
    border-radius: 12px;
    background-color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.stChatInputContainer:focus-within {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* File uploader styling */
.stFileUploader {
    border: 2px dashed #dee2e6;
    border-radius: 12px;
    padding: 2rem;
    height: 220px;
    text-align: center;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #667eea;
    background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
}

/* Success/Error messages */
.stSuccess {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #c3e6cb;
    border-radius: 8px;
    color: #155724;
}

.stError {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #f5c6cb;
    border-radius: 8px;
    color: #721c24;
}

/* Expander styling */
.streamlit-expander {
    border: 1px solid #e9ecef;
    border-radius: 8px;
    background-color: #f8f9fa;
}

/* Loading spinner */
.stSpinner {
    text-align: center;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        margin-left: 5%;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        margin-right: 5%;
    }
}

/* Typing indicator */
.typing-indicator {
    display: flex;
    align-items: center;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 18px 18px 18px 4px;
    margin-right: 10%;
    margin-bottom: 1rem;
}

.typing-dots {
    display: flex;
    gap: 4px;
}

.typing-dots span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #667eea;
    animation: typing 1.4s infinite ease-in-out;
}

.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}

/* Document container styling */
.email-draft-container {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #e9ecef;
    color: #262626;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

.email-draft-header {
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e9ecef;
}

.email-draft-title {
    color: #495057;
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
}

.small-copy-btn {
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: all 0.2s ease;
    opacity: 0.6;
}

.small-copy-btn:hover {
    opacity: 1;
    background: rgba(102, 126, 234, 0.1);
}

/* Existing styles continue */
</style>
""", unsafe_allow_html=True)

# --- Suggestion Functions ---
def get_document_suggestions(doc_type, extracted_text=""):
    """Generate contextual suggestions based on document type and content"""
    
    if "legal notice" in extracted_text.lower() or "notice" in doc_type.lower():
        return [
            "📝 Draft a professional reply to this legal notice",
            "⚖️ Analyze the legal claims mentioned",
            "📋 Check compliance requirements",
            "🕐 Review response timeline and deadlines",
            "💡 Suggest negotiation strategies"
        ]
    elif "contract" in extracted_text.lower() or "agreement" in extracted_text.lower():
        return [
            "📊 Review contract terms and conditions",
            "⚠️ Identify potential legal risks",
            "🔍 Check for missing clauses",
            "💰 Analyze payment and penalty terms",
            "📅 Review termination conditions"
        ]
    elif "court" in extracted_text.lower() or "petition" in extracted_text.lower():
        return [
            "⚖️ Analyze the legal arguments",
            "📋 Review evidence requirements",
            "🕐 Check procedural timelines",
            "💡 Suggest counter-arguments",
            "📝 Draft response strategy"
        ]
    else:
        return [
            "🔍 Analyze this legal document",
            "📝 Explain key legal terms",
            "⚖️ Identify legal implications",
            "💡 Provide actionable advice",
            "📋 Summarize important points"
        ]

def get_general_suggestions():
    """Get general legal consultation suggestions"""
    return [
        "💼 Property law consultation",
        "👔 Employment law guidance", 
        "🏪 Business law advice",
        "👨‍👩‍👧‍👦 Family law matters",
        "🚗 Consumer rights issues",
        "📋 Contract drafting help"
    ]

def get_follow_up_suggestions(last_response):
    """Generate follow-up suggestions based on the last response"""
    if "reply" in last_response.lower() or "response" in last_response.lower():
        return [
            "✏️ Refine the draft response",
            "📧 Format for official communication",
            "⏰ Set reminder for response deadline",
            "💡 Add stronger legal arguments",
            "👥 Review with legal counsel"
        ]
    elif "analysis" in last_response.lower():
        return [
            "📝 Get step-by-step action plan",
            "⚖️ Understand legal precedents",
            "💰 Estimate potential costs",
            "🕐 Timeline for resolution",
            "📞 When to consult a lawyer"
        ]
    else:
        return [
            "🔍 Need more details?",
            "📋 Want a summary?",
            "💡 Explore related topics",
            "📝 Draft related documents",
            "❓ Ask follow-up questions"
        ]

# --- Database Setup ---
def init_database():
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            retrieved_context TEXT,
            uploaded_file_text TEXT,
            suggestions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_documents (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            file_name TEXT,
            file_type TEXT,
            extracted_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = init_database()

# --- Document Handling Functions ---
def create_pdf_download(content, doc_id, sender_info=None, receiver_info=None):
    """Create a PDF buffer for downloadable content"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Add letterhead
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Legal Document")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add sender/receiver info if provided
    y_position = height - 100
    if sender_info:
        c.drawString(50, y_position, "From:")
        c.drawString(100, y_position, sender_info)
        y_position -= 20
    
    if receiver_info:
        c.drawString(50, y_position, "To:")
        c.drawString(100, y_position, receiver_info)
        y_position -= 40
    else:
        y_position -= 20
    
    # Add content
    c.setFont("Helvetica", 11)
    text_object = c.beginText(50, y_position)
    for line in content.split('\n'):
        text_object.textLine(line)
    c.drawText(text_object)
    
    c.save()
    buffer.seek(0)
    return buffer

def render_message_with_copy(content, message_id, is_email=False, sender_info=None, receiver_info=None):
    """Render message with copy button and download options for emails"""

    # Check if content contains downloadable document formats
    is_downloadable = any(keyword in content.lower() for keyword in [
        "email draft",
        "legal notice",
        "affidavit",
        "noc",
        "notice of claim",
        "agreement",
        "contract",
        "declaration",
        "memorandum",
        "letter of intent"
    ])

    # Define the common content display (no extra <div> wrapping)
    content_html = f"""
    <div style="position: relative;">
        <button class="small-copy-btn" onclick="copyContent('{message_id}')" style="position: absolute; right: 0; top: 0;">
            <em>📋</em>
        </button>
        <div id="content_{message_id}">
            {content.replace(chr(10), '<br>')}
        </div>
    </div>
    """

    if is_email and is_downloadable:
        # Container for formal documents
        st.markdown(f"""
        <div class="email-draft-container">
            <div class="email-draft-header" style="display: flex; justify-content: space-between; align-items: center;">
                <h4 class="email-draft-title">📧 Document Ready</h4>
            </div>
            {content_html}
        
        """, unsafe_allow_html=True)

        # Download buttons only for formal documents
        col1, col2 = st.columns([1, 1])

        with col1:
            # Text download
            text_content = content
            st.download_button(
                label="📄 Download TXT",
                data=text_content,
                file_name=f"legal_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"txt_download_{message_id}"
            )

        with col2:
            # PDF download
            pdf_buffer = create_pdf_download(content, message_id, sender_info, receiver_info)
            st.download_button(
                label="📑 Download PDF",
                data=pdf_buffer,
                file_name=f"legal_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key=f"pdf_download_{message_id}"
            )
    else:
        # Regular message
        st.markdown(content_html, unsafe_allow_html=True)

# --- Database Setup ---
def init_database():
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            retrieved_context TEXT,
            uploaded_file_text TEXT,
            suggestions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_documents (
            id TEXT PRIMARY KEY,
            chat_id TEXT,
            file_name TEXT,
            file_type TEXT,
            extracted_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES chats(id)
        )
    """)
    conn.commit()
    return conn, cursor

conn, cursor = init_database()

# --- File Processing Functions ---
def extract_text_from_image(image):
    try:
        reader = easyocr.Reader(['en'])

        result = reader.readtext(image)
        return result
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def save_document_to_chat(chat_id, file_name, file_type, extracted_text):
    cursor.execute(
        "INSERT INTO chat_documents (id, chat_id, file_name, file_type, extracted_text) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), chat_id, file_name, file_type, extracted_text)
    )
    conn.commit()

def get_chat_documents(chat_id):
    cursor.execute("SELECT file_name, file_type, extracted_text FROM chat_documents WHERE chat_id = ? ORDER BY created_at", (chat_id,))
    return cursor.fetchall()

# --- Model Initialization ---
@st.cache_resource
def init_models():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.load_local("legal_qa_faiss_index", embeddings, allow_dangerous_deserialization=True)

        if os.getenv('env') == 'LOCAL':
            llm = OllamaLLM(model="mistral", temperature=0.1, max_tokens=512)
        else:
            llm = OllamaLLM(base_url='https://a97b-112-196-43-19.ngrok-free.app', model="mistral", temperature=0.1, max_tokens=512)

        return vectordb, llm
    except Exception as e:
        st.error(f"❌ Failed to initialize models: {e}")
        return None, None

vector_db, llm = init_models()

# --- Helper Functions ---
def load_chats():
    cursor.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC")
    return cursor.fetchall()

def get_messages(chat_id):
    print(f"Fetching messages for chat_id: {chat_id}")  # Debug log
    if not chat_id:
        return []
    cursor.execute("SELECT role, content, retrieved_context, uploaded_file_text, suggestions FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,))
    return cursor.fetchall()

def create_new_chat():
    new_chat_id = str(uuid.uuid4())
    placeholder_title = "New Legal Consultation"
    cursor.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (new_chat_id, placeholder_title))
    conn.commit()
    return new_chat_id

def update_chat_title(chat_id, new_title):
    safe_title = new_title.strip()[:50]
    if len(safe_title) < 3:
        safe_title = "Legal Consultation"
    cursor.execute("UPDATE chats SET title = ? WHERE id = ?", (safe_title, chat_id))
    conn.commit()

def delete_chat(chat_id):
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()

def save_message(chat_id, role, content, retrieved_context=None, uploaded_file_text=None, suggestions=None):
    print(f"Saving message: chat_id={chat_id}, role={role}, content={content[:50]}...")  # Debug log
    context_json = json.dumps(retrieved_context) if retrieved_context else None
    suggestions_json = json.dumps(suggestions) if suggestions else None
    cursor.execute(
        "INSERT INTO messages (id, chat_id, role, content, retrieved_context, uploaded_file_text, suggestions) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), chat_id, role, content, context_json, uploaded_file_text, suggestions_json)
    )
    conn.commit()

def is_greeting_or_casual(text):
    text_lower = text.lower().strip()
    greetings = [
        r'\b(hi|hello|hey|hii|hiii|heyyy)\b',
        r'\bhow are you\b',
        r'\bhow\'s it going\b',
        r'\bgood morning\b',
        r'\bgood afternoon\b',
        r'\bgood evening\b',
        r'\bnice to meet you\b',
        r'\bthanks?\b',
        r'\bthank you\b',
        r'\bokay?\b',
        r'\bok\b',
        r'\byes\b',
        r'\bno\b',
        r'\bbye\b',
        r'\bgoodbye\b',
        r'\bsee you\b',
        r'\btake care\b',
        r'\bwhat\'s up\b',
        r'\bwassup\b',
        r'\bhru\b',
        r'\bwbu\b',
    ]
    
    for pattern in greetings:
        if re.search(pattern, text_lower):
            return True
    
    if len(text_lower.split()) <= 3 and not any(legal_word in text_lower for legal_word in 
        ['law', 'legal', 'court', 'case', 'lawyer', 'attorney', 'contract', 'agreement', 'notice', 'sue', 'rights']):
        return True
    
    return False

def get_casual_response(text):
    text_lower = text.lower().strip()
    
    if any(greeting in text_lower for greeting in ['hi', 'hello', 'hey']):
        return "Hello! I'm your AI Legal Adviser. I'm here to help you with legal questions, document analysis, and provide professional legal guidance. How can I assist you today? ⚖️"
    elif 'how are you' in text_lower:
        return "I'm doing well, thank you for asking! I'm ready to help you with any legal questions or document analysis you might need. What legal matter can I assist you with today?"
    elif any(thanks in text_lower for thanks in ['thank', 'thanks']):
        return "You're welcome! I'm always here to help with your legal questions. Is there anything else you'd like to know about legal matters?"
    elif text_lower in ['ok', 'okay', 'yes', 'no']:
        return "I'm here whenever you need legal assistance. Feel free to ask me about legal documents, Indian law, or any legal questions you might have!"
    elif any(bye in text_lower for bye in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Remember, I'm always here whenever you need legal guidance or document analysis. Take care! ⚖️"
    else:
        return "I'm your AI Legal Adviser, specialized in helping with legal questions and document analysis. How can I assist you with legal matters today? ⚖️"

def generate_legal_response(prompt, chat_history, doc_text="", retrieved_context=None):
    """Generate enhanced legal response with context"""
    
    # Create context string
    context_str = ""
    if retrieved_context:
        context_str = "\n\nRelevant Legal Information:\n" + "\n".join(retrieved_context)
    
    if doc_text:
        context_str += f"\n\nUploaded Document Content:\n{doc_text[:1500]}..."
    
    # Enhanced prompt template
    enhanced_prompt = f"""You are an expert AI Legal Adviser specializing in Indian law. Your role is to provide accurate, professional, and actionable legal guidance.

Context Information:{context_str}

Previous Conversation:
{chat_history}

Current Question: {prompt}

Please provide a comprehensive response that:
1. Addresses the specific legal question clearly
2. References relevant laws and regulations when applicable
3. Provides practical, actionable advice
4. Explains complex legal terms in simple language
5. Suggests next steps where appropriate
6. Mentions when professional legal consultation is recommended

Format your response professionally with clear sections where appropriate."""

    try:
        # Initialize response generation
        full_response = ""
        
        # Process response stream        # Process response stream
        for chunk in llm.stream(enhanced_prompt):
            if isinstance(chunk, dict):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response += content
                    yield content
            elif isinstance(chunk, str):
                full_response += chunk
                yield chunk
        
        # For non-streaming calls, return the full response
        if not full_response:
            response = llm(enhanced_prompt)
            if isinstance(response, dict):
                return str(response.get("message", {}).get("content", "Error: Empty response"))
            return str(response)
        
        return full_response
            
    except Exception as e:
        error_msg = f"I apologize, but I encountered an error while processing your request. Please try rephrasing your question or contact support if the issue persists. Error details: {str(e)}"
        yield error_msg

def get_vectordb():
    """Get the vector database for context retrieval"""
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db, st.session_state.llm = init_models()
    return st.session_state.vector_db

# --- Initialize Session State ---
if 'current_chat_id' not in st.session_state:
    chats = load_chats()
    if chats:
        st.session_state.current_chat_id = chats[0][0]
    else:
        st.session_state.current_chat_id = create_new_chat()

if 'messages_loaded' not in st.session_state:
    st.session_state.messages_loaded = False

if 'show_suggestions' not in st.session_state:
    st.session_state.show_suggestions = True

# Initialize voice chat state
if 'voice_chat_active' not in st.session_state:
    st.session_state.voice_chat_active = False
if 'voice_chat' not in st.session_state:
    st.session_state.voice_chat = None

# --- Ensure the app displays content properly
# st.title("⚖️ AI Legal Adviser")
# st.markdown("*Professional legal guidance powered by AI*")

# # Add a simple debug message to confirm the app is running
# st.write("App is running successfully!")

# Ensure the sidebar and main interface are initialized correctly
# with st.sidebar:
#     st.markdown("### 💬 Legal Consultations")
#     st.button("✨ New Consultation")

# # Add a placeholder for the main interface
# st.markdown("### Welcome to the AI Legal Adviser")
# st.write("Use the sidebar to start a new consultation or upload a legal document.")

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("### 💬 Legal Consultations")
    
    # New Chat Button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("✨ New Consultation", use_container_width=True, type="primary"):
            new_chat_id = create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages_loaded = False
            st.rerun()
    
    with col2:
        if st.button("🧹", help="Clear all chats", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                cursor.execute("DELETE FROM messages")
                cursor.execute("DELETE FROM chat_documents")
                cursor.execute("DELETE FROM chats")
                conn.commit()
                st.session_state.current_chat_id = create_new_chat()
                st.session_state.confirm_clear = False
                st.success("✅ All chats cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm")
    
    st.divider()
    
    # Display chats
    chats = load_chats()
    
    if chats:
        for chat_id, title, created_at in chats:
            chat_container = st.container()
            
            with chat_container:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(
                        title,
                        key=f"chat_{chat_id}",
                        use_container_width=True,
                        type="primary" if chat_id == st.session_state.current_chat_id else "secondary"
                    ):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages_loaded = False
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        # Delete current chat
                        delete_chat(chat_id)

                        # Update session state with new chat ID
                        remaining_chats = load_chats()
                        if remaining_chats:
                            st.session_state.current_chat_id = remaining_chats[0][0]
                        else:
                            # Create a new chat if we deleted the last one
                            new_chat_id = create_new_chat()
                            st.session_state.current_chat_id = new_chat_id
                        
                        st.session_state.messages_loaded = False
                        st.rerun()
    else:
        st.info("💡 No consultations yet. Start your first legal consultation!")

# --- Main Interface ---
st.markdown("# ⚖️ AI Legal Adviser")
st.markdown("*Professional legal guidance powered by AI*")

# File Upload Section
st.subheader("📁 Upload Legal Document")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Drop your legal document here or click to browse",
        type=["pdf", "png", "jpg", "jpeg"],
        key=f"file_upload_{st.session_state.current_chat_id}",
        help="Supported formats: PDF, PNG, JPG, JPEG"
    )

with col2:
    st.markdown("""📋 Document Tips""")
    st.info("""
    **Tips for better analysis:**
    - Ensure text is clear and readable
    - Upload complete documents
    - Multiple pages? Use PDF format
    - High resolution for image files
    """)

# Voice Chat Integration
st.markdown("### 🎙️ Voice Chat")
col1, col2 = st.columns(2)

with col1:
    start_voice = st.button(
        "Start Voice Chat",
        key="start_voice",
        help="Click to start voice conversation with the AI",
        use_container_width=True
    )
    
with col2:
    stop_voice = st.button(
        "Stop Voice Chat",
        key="stop_voice",
        help="Click to stop voice conversation",
        use_container_width=True,
        type="secondary"
    )

# Initialize voice chat state if not exists
if 'voice_chat_active' not in st.session_state:
    st.session_state.voice_chat_active = False
if 'voice_chat' not in st.session_state:
    st.session_state.voice_chat = None

# Handle voice chat controls
if start_voice and not st.session_state.voice_chat_active:
    st.session_state.voice_chat_active = True
    st.session_state.voice_chat = VoiceChat(
        save_message_callback=save_message,
        chat_id=st.session_state.current_chat_id,
        vector_db=vector_db,
        generate_legal_response_callback=generate_legal_response
    )
    st.session_state.voice_chat.start()
    st.rerun()

if stop_voice and st.session_state.voice_chat_active:
    if st.session_state.voice_chat:
        st.session_state.voice_chat.stop_speaking = True
        del st.session_state.voice_chat
    st.session_state.voice_chat_active = False
    st.rerun()

if st.session_state.voice_chat_active:
    st.markdown("""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; margin: 1rem 0;">
        <h4 style="margin: 0;">🎙️ Voice Chat Active</h4>
        <p style="margin: 0.5rem 0;">
            Speak clearly into your microphone to ask questions.<br>
            Press SPACE to stop AI's response at any time.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Continue with existing chat interface
uploaded_text = ""
doc_analysis_done = False

# Display chat documents summary
chat_docs = get_chat_documents(st.session_state.current_chat_id)
if chat_docs:
    st.info(f"📎 {len(chat_docs)} document(s) uploaded to this consultation")

st.divider()

# Display chat messages with enhanced styling
current_messages = get_messages(st.session_state.current_chat_id)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)
for role, content, context, file_text, suggestions in current_messages:
    if role == "user":
        memory.chat_memory.add_user_message(content)
    elif role == "assistant":
        memory.chat_memory.add_ai_message(content)

# Enhanced message display
for i, (role, content, retrieved_context, file_text, suggestions) in enumerate(current_messages):
    with st.chat_message(role):
        # Use the new render_message_with_copy function
        message_id = f"msg_{i}"
        is_email = any(keyword in content.lower() for keyword in ["email draft", "legal notice", "letter"])
        if role == "user":
            st.markdown(f'<div id="{message_id}" class="chat-message {role}">{content}</div>', unsafe_allow_html=True)
        elif role == "assistant" and not suggestions:
            st.markdown(f'<div id="{message_id}" class="chat-message {role}">{content}</div>', unsafe_allow_html=True)
        # Show suggestions for assistant messages
        if role == "assistant" and suggestions:
            try:
                render_message_with_copy(content, message_id, is_email)
                suggestion_list = json.loads(suggestions)
                if suggestion_list:
                    st.markdown("##### 💫 Continue the conversation:")
                    cols = st.columns(min(len(suggestion_list), 3))
                    for j, suggestion in enumerate(suggestion_list[:3]):
                        with cols[j]:
                            if st.button(
                                suggestion, 
                                key=f"msg_suggest_{i}_{j}",
                                use_container_width=True,
                                type="secondary"
                            ):
                                st.session_state[f"suggested_prompt_{st.session_state.current_chat_id}"] = suggestion
                                st.rerun()
            except:
                pass
        
        # Show context
        if role == "assistant" and retrieved_context:
            try:
                context_data = json.loads(retrieved_context)
                if context_data:
                    with st.expander("📚 Legal References Used"):
                        for j, doc_content in enumerate(context_data, 1):
                            st.markdown(f"**Reference {j}:**")
                            st.text(doc_content[:400] + "..." if len(doc_content) > 400 else doc_content)
            except:
                pass

# Show general suggestions if no messages
if not current_messages and not doc_analysis_done:
    st.markdown("### 🚀 How can I help you today?")
    
    general_suggestions = get_general_suggestions()
    cols = st.columns(2)
    
    for i, suggestion in enumerate(general_suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"general_{i}", use_container_width=True, type="secondary"):
                st.session_state[f"suggested_prompt_{st.session_state.current_chat_id}"] = suggestion
                st.rerun()

# Handle suggested prompts (continuation)
suggested_prompt_key = f"suggested_prompt_{st.session_state.current_chat_id}"
if suggested_prompt_key in st.session_state:
    prompt = st.session_state[suggested_prompt_key]
    del st.session_state[suggested_prompt_key]
    
    # Process the suggested prompt
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message
    save_message(st.session_state.current_chat_id, "user", prompt, uploaded_file_text=uploaded_text if uploaded_text else None)
    
    # Update chat title if it's a new chat
    if not current_messages:
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        update_chat_title(st.session_state.current_chat_id, title)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your legal question..."):
            # Check if it's a casual greeting
            if is_greeting_or_casual(prompt):
                response = get_casual_response(prompt)
                suggestions = get_general_suggestions()
            else:
                # Retrieve relevant context if vector DB is available
                retrieved_context = None
                if vector_db:
                    try:
                        docs = vector_db.similarity_search(prompt, k=3)
                        retrieved_context = [doc.page_content for doc in docs]
                    except Exception as e:
                        st.error(f"⚠️ Context retrieval failed: {e}")
                
                # Get chat history for context
                chat_history = ""
                for role, content, _, _, _ in current_messages[-5:]:  # Last 5 messages for context
                    chat_history += f"{role.title()}: {content}\n"
                
                # Generate response
                response = generate_legal_response(
                    prompt, 
                    chat_history, 
                    uploaded_text, 
                    retrieved_context
                )
                
                # Generate follow-up suggestions
                suggestions = get_follow_up_suggestions(response)
        
        st.markdown(response)
        
        # Show follow-up suggestions
        if suggestions:
            st.markdown("##### 💫 Continue the conversation:")
            cols = st.columns(min(len(suggestions), 3))
            for j, suggestion in enumerate(suggestions[:3]):
                with cols[j]:
                    if st.button(
                        suggestion, 
                        key=f"followup_{j}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        st.session_state[f"suggested_prompt_{st.session_state.current_chat_id}"] = suggestion
                        st.rerun()
    
    # Save assistant response
    save_message(
        st.session_state.current_chat_id, 
        "assistant", 
        response, 
        retrieved_context, 
        suggestions=suggestions
    )
    
    st.rerun()

# Chat input
if prompt := st.chat_input("💬 Ask your legal question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message
    save_message(st.session_state.current_chat_id, "user", prompt, uploaded_file_text=uploaded_text if uploaded_text else None)
    
    # Update chat title for new chats
    if not current_messages:
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        update_chat_title(st.session_state.current_chat_id, title)
    
    retrieved_context = None
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Show typing indicator
        typing_placeholder = st.empty()
        typing_placeholder.markdown('''
        <div class="typing-indicator">
            <span style="margin-right: 10px;">AI Legal Adviser is thinking</span>
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Simulate thinking time
        sleep(1)
        typing_placeholder.empty()
        
        # Check if it's a casual greeting
        if is_greeting_or_casual(prompt):
            response = get_casual_response(prompt)
            suggestions = get_general_suggestions()
        else:
            # Retrieve relevant context
            if vector_db:
                try:
                    with st.spinner("🔍 Searching legal database..."):
                        docs = vector_db.similarity_search(prompt, k=3)
                        retrieved_context = [doc.page_content for doc in docs]
                        
                        if retrieved_context:
                            st.success(f"✅ Found {len(retrieved_context)} relevant legal references")
                except Exception as e:
                    st.warning(f"⚠️ Could not access legal database: {e}")
            
            # Prepare chat history
            chat_history = ""
            for role, content, _, _, _ in current_messages[-5:]:  # Last 5 messages
                chat_history += f"{role.title()}: {content}\n"
            
            # Generate legal response
            with st.spinner("⚖️ Crafting legal advice..."):
                response = generate_legal_response(
                    prompt, 
                    chat_history, 
                    uploaded_text, 
                    retrieved_context
                )
            
            # Generate contextual suggestions
            suggestions = get_follow_up_suggestions(response)
        
        # Display response
        st.markdown(response)
        
        # Show follow-up suggestions
        if suggestions:
            st.markdown("##### 💫 What would you like to do next?")
            
            # Create responsive columns
            num_cols = min(len(suggestions), 3)
            cols = st.columns(num_cols)
            
            for j, suggestion in enumerate(suggestions[:num_cols]):
                with cols[j]:
                    if st.button(
                        suggestion, 
                        key=f"response_suggest_{j}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        st.session_state[f"suggested_prompt_{st.session_state.current_chat_id}"] = suggestion
                        st.rerun()
    
    # Save assistant response
    save_message(
        st.session_state.current_chat_id, 
        "assistant", 
        response, 
        retrieved_context, 
        suggestions=suggestions
    )
    
    st.rerun()

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;'>
    <p>⚖️ <strong>AI Legal Adviser</strong> | Professional Legal Guidance</p>
    <p style='font-size: 0.8rem;'>
        <em>This AI provides general legal information and should not replace professional legal advice. 
        For specific legal matters, please consult with a qualified attorney.</em>
    </p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
        👨‍💻 Developed By: Apptunix Pvt Ltd | 
        🐑 Powered by Ollama Mistral 7b and ⚖️ legal knowledge base (Indian Law)
    </p>
</div>
""", unsafe_allow_html=True)

# --- Additional Features ---

# Keyboard shortcuts info
with st.expander("⌨️ Keyboard Shortcuts & Tips"):
    st.markdown("""
    **Shortcuts:**
    - `Ctrl + Enter` - Send message
    - `Ctrl + /` - Focus on chat input
    
    **Tips for better results:**
    - Be specific about your legal issue
    - Mention relevant dates and locations
    - Upload relevant documents for analysis
    - Ask follow-up questions for clarification
    
    **Document Upload Tips:**
    - Ensure text is clear and readable
    - Use high-resolution images
    - PDF format works best for multi-page documents
    - Supported formats: PDF, PNG, JPG, JPEG
    """)

# Legal disclaimer
with st.expander("⚠️ Important Legal Disclaimer"):
    st.markdown("""
    **IMPORTANT DISCLAIMER:**
    
    This AI Legal Adviser is designed to provide general legal information and guidance. It is not a substitute for professional legal advice from a qualified attorney.
    
    **Please note:**
    - This service does not create an attorney-client relationship
    - Information provided is for educational purposes only
    - Laws vary by jurisdiction and change over time
    - For specific legal matters, always consult with a licensed attorney
    - Do not share sensitive personal information
    
    **Limitation of Liability:**
    The creators of this AI tool are not responsible for any decisions made based on the information provided. Always seek professional legal counsel for important legal matters.
    """)

# Emergency legal resources
with st.expander("🆘 Emergency Legal Resources"):
    st.markdown("""
    **If you need immediate legal assistance:**
    
    **Legal Aid Services:**
    - National Legal Services Authority (NALSA): [nalsa.gov.in](https://nalsa.gov.in)
    - State Legal Services Authority in your state
    - District Legal Services Authority in your district
    
    **Emergency Contacts:**
    - Police: 100
    - Women Helpline: 1091
    - Child Helpline: 1098
    - Senior Citizens Helpline: 14567
    
    **Online Legal Resources:**
    - India Code: [indiacode.nic.in](https://indiacode.nic.in)
    - Supreme Court of India: [sci.gov.in](https://sci.gov.in)
    - Bar Council of India: [barcouncilofindia.org](https://barcouncilofindia.org)
    """)

# Analytics (optional - for usage tracking)
if st.session_state.current_chat_id:
    # Count messages in current chat
    cursor.execute("SELECT COUNT(*) FROM messages WHERE chat_id = ?", (st.session_state.current_chat_id,))
    message_count = cursor.fetchone()[0]
    
    # Count documents in current chat
    cursor.execute("SELECT COUNT(*) FROM chat_documents WHERE chat_id = ?", (st.session_state.current_chat_id,))
    doc_count = cursor.fetchone()[0]
    
    # Display stats in sidebar
    with st.sidebar:
        st.divider()
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", message_count)
        with col2:
            st.metric("Documents", doc_count)

# --- Enhanced CSS for Copy Button ---
st.markdown("""
<style>
.small-copy-btn {
    background-color: transparent;
    border: none;
    cursor: pointer;
    color: #667eea;
    font-size: 1.2rem;
    margin-left: 10px;
    transition: color 0.3s ease;
}

.small-copy-btn:hover {
    color: #764ba2;
}

/* Email draft container */
.email-draft-container {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    position: relative;
}

.email-draft-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.email-draft-title {
    font-size: 1.1rem;
    margin: 0;
    color: #495057;
}

.email-draft-subtitle {
    font-size: 0.9rem;
    color: #868e96;
    margin: 0;
}

/* Legal document link */
.legal-doc-link {
    color: #667eea;
    text-decoration: none;
    font-weight: 500;
}

.legal-doc-link:hover {
    text-decoration: underline;
}

/* Button styles */
.stButton > button {
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3);
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background-color: white;
    color: #495057;
    border: 1.5px solid #dee2e6;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #f8f9fa;
    border-color: #667eea;
    color: #667eea;
    transform: translateY(-1px);
}

/* Spinner */
.stSpinner {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
}

/* Responsive design adjustments */
@media (max-width: 768px) {
    .email-draft-container {
        padding: 0.8rem;
    }
    
    .email-draft-title {
        font-size: 1rem;
    }
    
    .email-draft-subtitle {
        font-size: 0.8rem;
    }
}
</style>
""", unsafe_allow_html=True)

# The following script is for Streamlit's HTML injection, not React.
# The error you mention is from React, but Streamlit uses its own frontend.
# However, to avoid confusion and ensure compatibility, let's make sure the JS is robust.

st.markdown("""
<script>
window.copyContent = function(messageId) {
    const contentElem = document.getElementById('content_' + messageId);
    if (!contentElem) return;
    const content = contentElem.innerText;
    navigator.clipboard.writeText(content).then(() => {
        const btn = event.target.closest('.small-copy-btn');
        if (btn) {
            const originalContent = btn.innerHTML;
            btn.innerHTML = '<em>✅</em>';
            setTimeout(() => {
                btn.innerHTML = originalContent;
            }, 2000);
        }
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}
</script>
""", unsafe_allow_html=True)