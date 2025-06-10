import streamlit as st
import sqlite3
import uuid
import json
import re
from datetime import datetime
from time import sleep
import pytesseract
from PIL import Image
import pdfplumber

from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import HumanMessage, AIMessage

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Configure page
st.set_page_config(
    page_title="Legal Adviser Bot",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for button styling
st.markdown("""
<style>
/* Primary buttons (New Chat button) */
.stButton > button[kind="primary"] {
    background-color: #0068C9 !important;  /* Blue */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1E82D7 !important;  /* Light Blue on hover */
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* Secondary buttons (Chat selection buttons) */
.stButton > button[kind="secondary"] {
    background-color: #37474F !important;  /* Blue-gray */
    color: white !important;
    border: 1px solid #546E7A !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    background-color: #455A64 !important;
    border-color: #78909C !important;
    transform: translateY(-1px) !important;
}

/* Delete buttons */
.stButton > button:has([title="Delete chat"]) {
    background-color: #D32F2F !important;  /* Red */
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 35px !important;
    height: 35px !important;
    padding: 0 !important;
    transition: all 0.3s ease !important;
}

.stButton > button:has([title="Delete chat"]):hover {
    background-color: #B71C1C !important;  /* Darker red */
    transform: scale(1.1) !important;
    box-shadow: 0 2px 6px rgba(211, 47, 47, 0.4) !important;
}

/* Send button styling */
.stButton > button[data-testid="baseButton-secondary"] {
    background: linear-gradient(45deg, #1976D2, #42A5F5) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(45deg, #1565C0, #1E88E5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3) !important;
}

/* Active chat button */
.stButton > button[kind="primary"][aria-pressed="true"] {
    background: linear-gradient(45deg, #FF6B35, #FF8E53) !important;  /* Orange gradient for active */
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.3) !important;
}

/* File uploader button */
.stFileUploader > div > button {
    background-color: #6A1B9A !important;  /* Purple */
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.stFileUploader > div > button:hover {
    background-color: #4A148C !important;  /* Darker purple */
    transform: translateY(-1px) !important;
}

/* Custom button class for special styling */
.custom-legal-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.custom-legal-button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 6px 16px rgba(102, 126, 234, 0.3) !important;
}

/* Sidebar styling */
.css-1d391kg {  /* Sidebar background */
    background-color: #F8F9FA !important;
}

/* Make buttons more responsive */
@media (max-width: 768px) {
    .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.8rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# --- Greeting Detection Functions ---
def is_greeting_or_casual(text):
    """
    Check if the input is a greeting or casual conversation that doesn't require legal expertise
    """
    text_lower = text.lower().strip()
    
    # Common greetings
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
        r'\bhru\b',  # how are you
        r'\bwbu\b',  # what about you
    ]
    
    for pattern in greetings:
        if re.search(pattern, text_lower):
            return True
    
    # Check if it's a very short message (likely casual)
    if len(text_lower.split()) <= 3 and not any(legal_word in text_lower for legal_word in 
        ['law', 'legal', 'court', 'case', 'lawyer', 'attorney', 'contract', 'agreement', 'notice', 'sue', 'rights']):
        return True
    
    return False

def get_casual_response(text):
    """
    Generate appropriate casual responses for greetings and simple conversations
    """
    text_lower = text.lower().strip()
    
    if any(greeting in text_lower for greeting in ['hi', 'hello', 'hey']):
        return "Hello! I'm your Legal Adviser Bot. I'm here to help you with legal questions and document analysis. How can I assist you today? ⚖️"
    
    elif 'how are you' in text_lower:
        return "I'm doing well, thank you for asking! I'm ready to help you with any legal questions or document analysis you might need. What legal matter can I assist you with today?"
    
    elif any(thanks in text_lower for thanks in ['thank', 'thanks']):
        return "You're welcome! I'm always here to help with your legal questions. Is there anything else you'd like to know about legal matters?"
    
    elif text_lower in ['ok', 'okay', 'yes', 'no']:
        return "I'm here whenever you need legal assistance. Feel free to ask me about legal documents, Indian law, or any legal questions you might have!"
    
    elif any(bye in text_lower for bye in ['bye', 'goodbye', 'see you']):
        return "Goodbye! Remember, I'm always here whenever you need legal guidance or document analysis. Take care! ⚖️"
    
    elif 'good morning' in text_lower:
        return "Good morning! I hope you're having a great day. I'm your Legal Adviser Bot, ready to help with any legal questions or document analysis you need."
    
    elif 'good afternoon' in text_lower:
        return "Good afternoon! I'm your Legal Adviser Bot, here to assist you with legal questions and document analysis. How can I help you today?"
    
    elif 'good evening' in text_lower:
        return "Good evening! I'm ready to help you with any legal matters or document analysis you might need. What can I assist you with?"
    
    else:
        return "I'm your Legal Adviser Bot, specialized in helping with legal questions and document analysis. How can I assist you with legal matters today? ⚖️"

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
    """Extract text from image using OCR"""
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
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
    """Save uploaded document to chat"""
    cursor.execute(
        "INSERT INTO chat_documents (id, chat_id, file_name, file_type, extracted_text) VALUES (?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), chat_id, file_name, file_type, extracted_text)
    )
    conn.commit()

def get_chat_documents(chat_id):
    """Get all documents for a chat"""
    cursor.execute("SELECT file_name, file_type, extracted_text FROM chat_documents WHERE chat_id = ? ORDER BY created_at", (chat_id,))
    return cursor.fetchall()

# --- Model Initialization ---
@st.cache_resource
def init_models():
    try:
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load FAISS vector store from local directory
        vectordb = FAISS.load_local("legal_qa_faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Initialize LLM
        llm = OllamaLLM(base_url='https://9052-112-196-43-19.ngrok-free.app', model="mistral", temperature=0.1, max_tokens=512)

        return vectordb, llm
    except Exception as e:
        st.error(f"❌ Failed to initialize models: {e}")
        return None, None

vector_db, llm = init_models()

# --- Helper Functions ---
def load_chats():
    """Load all chats from database"""
    cursor.execute("SELECT id, title, created_at FROM chats ORDER BY created_at DESC")
    return cursor.fetchall()

def get_messages(chat_id):
    """Get all messages for a specific chat"""
    if not chat_id:
        return []
    cursor.execute("SELECT role, content, retrieved_context, uploaded_file_text FROM messages WHERE chat_id = ? ORDER BY created_at", (chat_id,))
    return cursor.fetchall()

def create_new_chat():
    """Create a new chat session"""
    new_chat_id = str(uuid.uuid4())
    placeholder_title = "New Chat"
    cursor.execute("INSERT INTO chats (id, title) VALUES (?, ?)", (new_chat_id, placeholder_title))
    conn.commit()
    return new_chat_id

def update_chat_title(chat_id, new_title):
    """Update chat title based on first message"""
    safe_title = new_title.strip()[:50]
    if len(safe_title) < 3:
        safe_title = "New Chat"
    cursor.execute("UPDATE chats SET title = ? WHERE id = ?", (safe_title, chat_id))
    conn.commit()

def delete_chat(chat_id):
    """Delete a chat and all its messages"""
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chat_documents WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()

def save_message(chat_id, role, content, retrieved_context=None, uploaded_file_text=None):
    """Save a message to the database"""
    context_json = json.dumps(retrieved_context) if retrieved_context else None
    cursor.execute(
        "INSERT INTO messages (id, chat_id, role, content, retrieved_context, uploaded_file_text) VALUES (?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), chat_id, role, content, context_json, uploaded_file_text)
    )
    conn.commit()

# --- Initialize Session State ---
if 'current_chat_id' not in st.session_state:
    chats = load_chats()
    if chats:
        st.session_state.current_chat_id = chats[0][0]
    else:
        # Create first chat if none exist
        st.session_state.current_chat_id = create_new_chat()

if 'messages_loaded' not in st.session_state:
    st.session_state.messages_loaded = False

# --- Sidebar for Chat Management ---
with st.sidebar:
    st.title("💬 Legal Adviser Chats")
    
    # New Chat Button with custom styling
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_chat_id = create_new_chat()
            st.session_state.current_chat_id = new_chat_id
            st.session_state.messages_loaded = False
            st.rerun()
    
    with col2:
        # Optional: Add a clear all chats button
        if st.button("🧹", help="Clear all chats", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                # Clear all chats
                cursor.execute("DELETE FROM messages")
                cursor.execute("DELETE FROM chat_documents")
                cursor.execute("DELETE FROM chats")
                conn.commit()
                st.session_state.current_chat_id = create_new_chat()
                st.session_state.confirm_clear = False
                st.markdown(
                    """
                    <div style="
                        position: fixed;
                        top: 20px;
                        left: 50%;
                        transform: translateX(-50%);
                        background-color: #d4edda;
                        color: #155724;
                        padding: 12px 24px;
                        border: 1px solid #c3e6cb;
                        border-radius: 8px;
                        z-index: 9999;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        ✅ All chats cleared!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.markdown(
                    """
                    <div style="
                        position: fixed;
                        top: 20px;
                        left: 50%;
                        transform: translateX(-50%);
                        background-color: #fff3cd;
                        color: #856404;
                        padding: 12px 24px;
                        border: 1px solid #ffeeba;
                        border-radius: 8px;
                        z-index: 9999;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        ⚠️ Click again to confirm clearing all chats
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    st.divider()
    
    # Load and display chats
    chats = load_chats()
    
    if chats:
        st.subheader("Recent Chats")
        
        for chat_id, title, created_at in chats:
            # Create a container for each chat
            chat_container = st.container()
            
            with chat_container:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Chat selection button
                    if st.button(
                        title if title != "New Chat" else "💭 New Chat",
                        key=f"chat_{chat_id}",
                        use_container_width=True,
                        type="secondary" if chat_id != st.session_state.current_chat_id else "primary"
                    ):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages_loaded = False
                        st.rerun()
                
                with col2:
                    # Delete button
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        delete_chat(chat_id)
                        
                        # Select another chat or create new one
                        remaining_chats = load_chats()
                        if remaining_chats:
                            st.session_state.current_chat_id = remaining_chats[0][0]
                        else:
                            st.session_state.current_chat_id = create_new_chat()
                        
                        st.session_state.messages_loaded = False
                        st.rerun()
    else:
        st.info("No chats yet. Create your first chat!")

# --- Main Chat Interface ---
st.title("⚖️ Legal Adviser Bot")
st.markdown("*Ask me legal questions and I'll provide professional legal guidance*")

# File Upload Section
st.subheader("📁 Upload Legal Document (Optional)")
uploaded_file = st.file_uploader(
    "Upload Legal Notice/Document (PDF/Image)", 
    type=["pdf", "png", "jpg", "jpeg"],
    key=f"file_upload_{st.session_state.current_chat_id}"
)

uploaded_text = ""
if uploaded_file:
    with st.spinner("Processing uploaded document..."):
        if uploaded_file.type == "application/pdf":
            uploaded_text = extract_text_from_pdf(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            uploaded_text = extract_text_from_image(image)
    
    if uploaded_text.strip():
        # Save document to current chat
        save_document_to_chat(
            st.session_state.current_chat_id, 
            uploaded_file.name, 
            uploaded_file.type, 
            uploaded_text
        )
        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
        
        with st.expander("📄 View Extracted Text"):
            st.text_area("Extracted Content:", uploaded_text, height=200, disabled=True)
    else:
        st.error("❌ No readable text found in the document.")

# Display uploaded documents for current chat
chat_docs = get_chat_documents(st.session_state.current_chat_id)
if chat_docs:
    st.info(f"📎 {len(chat_docs)} document(s) uploaded to this chat")

st.divider()

# Get current chat messages
current_messages = get_messages(st.session_state.current_chat_id)

# Initialize memory with current chat messages
memory = ConversationBufferMemory(return_messages=True)
for role, content, context, file_text in current_messages:
    if role == "user":
        memory.chat_memory.add_user_message(content)
    elif role == "assistant":
        memory.chat_memory.add_ai_message(content)

# Display chat messages with context
for i, (role, content, retrieved_context, file_text) in enumerate(current_messages):
    with st.chat_message(role):
        st.markdown(content)
        
        # Show retrieved context for assistant messages
        if role == "assistant" and retrieved_context:
            try:
                context_data = json.loads(retrieved_context)
                if context_data:
                    with st.expander("📚 View Retrieved Legal Context"):
                        for j, doc_content in enumerate(context_data, 1):
                            st.markdown(f"**Document {j}:**")
                            st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
                            if j < len(context_data):
                                st.divider()
            except:
                pass  # Skip if context parsing fails

# Handle new user input
if prompt := st.chat_input("Ask your legal question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get all uploaded documents for this chat
    all_chat_docs = get_chat_documents(st.session_state.current_chat_id)
    combined_doc_text = "\n\n".join([doc[2] for doc in all_chat_docs])  # doc[2] is extracted_text
    
    # Save user message
    save_message(st.session_state.current_chat_id, "user", prompt, uploaded_file_text=combined_doc_text if combined_doc_text else None)
    memory.chat_memory.add_user_message(prompt)
    
    # Update chat title if it's a new chat
    current_chat = next((chat for chat in load_chats() if chat[0] == st.session_state.current_chat_id), None)
    if current_chat and current_chat[1] == "New Chat":
        # Create a meaningful title from the first question
        title_words = prompt.split()[:8]  # First 8 words
        new_title = " ".join(title_words)
        if len(new_title) > 50:
            new_title = new_title[:47] + "..."
        update_chat_title(st.session_state.current_chat_id, new_title)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        # Check if it's a greeting or casual conversation
        if is_greeting_or_casual(prompt):
            # Handle casual conversation without querying vector DB
            casual_response = get_casual_response(prompt)
            sleep(2)  # Simulate processing time
            st.markdown(casual_response)
            
            # Save casual response (no context needed)
            save_message(st.session_state.current_chat_id, "assistant", casual_response)
            memory.chat_memory.add_ai_message(casual_response)
        
        else:
            # Handle legal questions with vector DB search
            if vector_db and llm:
                with st.spinner("🔍 Searching legal documents..."):
                    try:
                        # Retrieve relevant document
                        docs = vector_db.max_marginal_relevance_search(prompt, k=3, fetch_k=10)
                        vector_context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Store context for later display
                        retrieved_docs = [doc.page_content for doc in docs] if docs else []
                        
                        # Prepare context - combine vector search results with uploaded documents
                        full_context = ""
                        if combined_doc_text:
                            full_context += f"Uploaded Documents:\n{combined_doc_text}\n\n"
                        if vector_context:
                            full_context += f"Legal Database Context:\n{vector_context}"
                        
                        # Prepare prompt with context and chat history
                        chat_history = memory.load_memory_variables({})['history']
                        
                        if combined_doc_text:
                            full_prompt = f"""You are a legal expert specializing in Indian law. The user has uploaded legal documents and is asking questions based on them.

Uploaded Documents:
{combined_doc_text}

Legal Database Context:
{vector_context}

Conversation History:
{chat_history}

Current Question: {prompt}

Please provide a professional legal response based on the uploaded documents first, then supplement with your knowledge of Indian law and the database context:"""
                        else:
                            full_prompt = f"""Legal Context:
{vector_context}

Conversation History:
{chat_history}

Current Question: {prompt}

Please provide a professional legal response based on the context above:"""
                        
                        # Stream the response
                        full_response = ""
                        response_placeholder = st.empty()
                        
                        for chunk in llm.stream(full_prompt):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▋")
                        
                        # Final response without cursor
                        response_placeholder.markdown(full_response)
                        
                        # Save assistant response with context
                        save_message(
                            st.session_state.current_chat_id, 
                            "assistant", 
                            full_response,
                            retrieved_context=retrieved_docs,
                            uploaded_file_text=combined_doc_text if combined_doc_text else None
                        )
                        memory.chat_memory.add_ai_message(full_response)
                        
                        # Show retrieved context in expander
                        if retrieved_docs:
                            with st.expander("📚 View Retrieved Legal Context"):
                                for i, doc_content in enumerate(retrieved_docs, 1):
                                    st.markdown(f"**Document {i}:**")
                                    st.text(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)
                                    if i < len(retrieved_docs):
                                        st.divider()
                    
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        error_response = "I apologize, but I encountered an error while processing your request. Please try again."
                        st.markdown(error_response)
                        save_message(st.session_state.current_chat_id, "assistant", error_response)
            else:
                error_msg = "Models not initialized. Please check your configuration."
                st.error(error_msg)
                save_message(st.session_state.current_chat_id, "assistant", error_msg)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>⚖️ Legal Adviser Bot - For informational purposes only. Apptunix Pvt. Ltd.</small>
    </div> 
    """, 
    unsafe_allow_html=True
)