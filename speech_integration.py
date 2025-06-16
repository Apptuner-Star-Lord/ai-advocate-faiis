import speech_recognition as sr
import pyttsx3
import ollama
import numpy as np
from pydub import AudioSegment
import threading
import queue
import time
import sqlite3, uuid
import re
from typing import Optional
import json
import keyboard
import streamlit as st

class EmotionDetector:
    def __init__(self):
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'wonderful', 'great', 'amazing'],
            'sarcastic': ['sarcastic', 'ironic', 'funny', 'hilarious', 'lol', 'haha'],
            'sad': ['sad', 'unfortunate', 'sorry', 'regret', 'disappointing'],
            'angry': ['angry', 'furious', 'annoyed', 'upset', 'terrible'],
            'neutral': []
        }
    
    def detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        for emotion, keywords in self.emotion_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return emotion
        return 'neutral'

class VoiceChat:
    def __init__(self, save_message_callback, chat_id, vector_db=None, generate_legal_response_callback=None):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.emotion_detector = EmotionDetector()
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.last_speech_time = 0
        self.speech_timeout = 0.5  # seconds
        self.current_emotion = 'neutral'
        self.speech_buffer = ""
        self.sentence_endings = ['.', '!', '?', '\n']
        self.stop_speaking = False
        self.speech_thread = None
        self.speaking_lock = threading.Lock()
        self.minimum_chunk_length = 3  # Minimum characters to process
        self.save_message = save_message_callback
        self.chat_id = chat_id
        self.vector_db = vector_db
        self.generate_legal_response = generate_legal_response_callback
        
        # Configure TTS engine
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        
        # Get available voices and set a default
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    def clean_text_for_speech(self, text: str) -> str:
        """Remove emojis and special symbols from text for speech synthesis"""
        # Remove emojis and special symbols using regex
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        cleaned_text = re.sub(r'[⚖️🟦😊😃😁]+', '', cleaned_text)  # Remove specific emojis
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Clean up extra spaces
        return cleaned_text.strip()

    def speak_with_emotion(self, text: str, emotion: str):
        """Speak text with appropriate emotion using rate and volume adjustments"""
        if self.stop_speaking or not text.strip():
            return

        with self.speaking_lock:  # Ensure only one speak operation at a time
            time.sleep(0.2)  # Increased pause before speaking to prevent cutting off
            
            # Reset to default values
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
            
            if emotion == 'sarcastic':
                self.engine.setProperty('rate', 180)
                self.engine.setProperty('volume', 0.9)
                text = text.replace('.', '...').replace('!', '...!')
            elif emotion == 'happy':
                self.engine.setProperty('rate', 170)
                self.engine.setProperty('volume', 1.0)
            elif emotion == 'sad':
                self.engine.setProperty('rate', 130)
                self.engine.setProperty('volume', 0.8)
            elif emotion == 'angry':
                self.engine.setProperty('rate', 160)
                self.engine.setProperty('volume', 1.0)
                text = text.replace('!', '! ').replace('.', '. ')
            else:  # neutral
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 1.0)            # Clean and normalize text for better speech
            text = self.clean_text_for_speech(text)  # Remove emojis and special symbols
            text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
            text = text.replace(',', ', ').replace('.', '. ')
            
            # Add a more noticeable pause at the start of the text
            text = f".... {text}"  # This creates a natural pause
            
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                time.sleep(0.1)  # Small pause after speaking
            except Exception as e:
                st.error(f"\nError in speech: {e}")    
    
    def process_text_chunk(self, chunk: str):
        """Process a chunk of text and speak it if it forms a complete sentence"""
        if self.stop_speaking:
            return

        self.speech_buffer += chunk
        
        # Find complete sentences using regex to avoid cutting words
        sentences = re.split(r'([.!?,])\s*', self.speech_buffer)
        
        if len(sentences) > 1:
            # Process complete sentences
            complete_text = ''
            remaining_text = ''
            
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    complete_text += sentences[i] + sentences[i+1]
                    
            if len(sentences) % 2 == 1:
                remaining_text = sentences[-1]
                
            if complete_text and len(complete_text) >= self.minimum_chunk_length:
                emotion = self.emotion_detector.detect_emotion(complete_text)
                self.current_emotion = emotion
                # Small pause to ensure text is displayed
                time.sleep(0.2)
                self.speak_with_emotion(complete_text, emotion)
                
            self.speech_buffer = remaining_text    

    def stream_legal_response(self, text: str, chat_history=""):
        """Stream response for legal questions with context"""
        try:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                completion_indicator = st.empty()
                full_response = ""
                buffer = ""
                
                # Check if it's a casual conversation
                if self.is_greeting_or_casual(text):
                    casual_response = self.get_casual_response(text)
                    response_placeholder.markdown(casual_response)
                    time.sleep(0.2)  # Ensure text is displayed
                    self.speak_with_emotion(casual_response, 'happy')
                    self.save_message(self.chat_id, "assistant", casual_response, None)
                    completion_indicator.markdown("*Listening for your next question...*")
                    return
                
                # Handle legal questions
                retrieved_context = None
                if self.vector_db:
                    try:
                        docs = self.vector_db.similarity_search(text, k=3)
                        retrieved_context = [doc.page_content for doc in docs]
                    except Exception as e:
                        st.warning(f"⚠️ Could not access legal database: {e}")
                
                if self.generate_legal_response:
                    for chunk in self.generate_legal_response(text, chat_history, retrieved_context=retrieved_context):
                        if self.stop_speaking:
                            break
                        
                        if isinstance(chunk, str):
                            buffer += chunk
                            # Display text immediately
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▋")
                            
                            # Look for natural break points (end of sentences or punctuation)
                            sentences = re.split(r'([.!?,])\s*', buffer)
                            
                            # Process complete sentences
                            if len(sentences) > 1:
                                for i in range(0, len(sentences)-1, 2):
                                    if i+1 < len(sentences):
                                        complete_sentence = sentences[i] + sentences[i+1] + " "
                                        self.process_text_chunk(complete_sentence)
                                
                                # Keep any remaining incomplete sentence in buffer
                                buffer = sentences[-1] if len(sentences) % 2 == 1 else ""
                
                # Process any remaining text in buffer
                if buffer:
                    self.process_text_chunk(buffer)
                
                # Save message and show completion indicator
                self.save_message(self.chat_id, "assistant", full_response, retrieved_context)
                completion_indicator.markdown("*Listening for your next question...*")

                # Process any remaining text in speech buffer
                if not self.stop_speaking and self.speech_buffer.strip():
                    time.sleep(0.2)  # Ensure text is displayed
                    self.speak_with_emotion(self.speech_buffer.strip(), self.current_emotion)
                self.speech_buffer = ""
                
        except Exception as e:
            st.error(f"Error in response generation: {e}")

    def check_for_interrupt(self):
        """Check for keyboard interrupt"""
        while True:
            if keyboard.is_pressed('space'):  # You can change this to any key
                self.stop_speaking = True
                self.engine.stop()
                break
            time.sleep(0.1)

    def get_audio_level(self, audio_data):
        """Calculate audio level from audio data"""
        data = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        return np.abs(data).mean()

    def process_audio(self):
        """Process audio input and generate responses"""
        with sr.Microphone() as source:
            st.info("Adjusting for ambient noise... Please wait.")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration for better calibration
            st.success("Ready! Speak your question. Press SPACE to stop the AI's response.")
            
            # Create placeholder for audio level visualization
            audio_level_placeholder = st.empty()
            listening_indicator = st.empty()
            
            while not self.stop_speaking:
                try:
                    # Show listening indicator
                    listening_indicator.markdown("*Listening...*")
                    
                    # Start listening with longer timeout and pre-buffer
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10, 
                                                 snowboy_configuration=None)  # Disable VAD for smoother capture
                    
                    # Calculate and display audio level
                    audio_level = self.get_audio_level(audio)
                    normalized_level = min(20, int(audio_level / 500))  # Adjust scaling as needed
                    audio_bars = "🟦" * normalized_level
                    audio_level_placeholder.markdown(f"Audio Level: {audio_bars}")
                    
                    # Convert speech to text
                    text = self.recognizer.recognize_google(audio)
                    
                    if text.strip():
                        # Clear listening indicator while processing
                        listening_indicator.empty()
                        
                        # Display user's speech
                        with st.chat_message("user"):
                            st.markdown(text)
                        
                        # Save user message
                        self.save_message(self.chat_id, "user", text)
                        
                        # Start interrupt checking in a separate thread
                        interrupt_thread = threading.Thread(target=self.check_for_interrupt)
                        interrupt_thread.daemon = True
                        interrupt_thread.start()
                        
                        # Generate and stream response
                        self.stream_legal_response(text)
                        
                        # Show listening indicator after response
                        listening_indicator.markdown("*Listening for your next question...*")
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    audio_level_placeholder.empty()
                    st.warning("Could not understand audio. Please try again.")
                    time.sleep(1.5)  # Give time for the warning to be read
                except sr.RequestError as e:
                    st.error(f"Could not request results: {e}")
                except Exception as e:
                    st.error(f"Error: {e}")
                    break

    def start(self):
        """Start the voice chat system"""
        try:
            self.process_audio()
        except KeyboardInterrupt:
            st.warning("Stopping voice chat...")
        finally:
            self.engine.stop()
            self.stop_speaking = True

    def get_casual_response(self, text):
        """Generate appropriate responses for casual conversation"""
        text_lower = text.lower().strip()
        
        # Greetings
        if any(greeting in text_lower for greeting in ['hi', 'hello', 'hey']):
            return "Hello! I'm your AI Legal Adviser. How can I assist you with any legal matters today? ⚖️"
        
        # Well-being questions
        elif 'how are you' in text_lower:
            return "I'm functioning well and ready to help you with any legal questions or document analysis. What can I assist you with today?"
        
        # Gratitude
        elif any(thanks in text_lower for thanks in ['thank', 'thanks']):
            return "You're welcome! I'm always here to help with your legal questions. Feel free to ask about any legal matters or documents you need assistance with."
        
        # Simple acknowledgments
        elif text_lower in ['ok', 'okay', 'yes', 'no', 'sure']:
            return "I'm here to help with any legal questions you have. Would you like me to explain something specific about law or analyze any legal documents?"
        
        # Farewells
        elif any(bye in text_lower for bye in ['bye', 'goodbye', 'see you', 'take care']):
            return "Goodbye! Remember, I'm always here when you need legal guidance or document analysis. Take care! ⚖️"
        
        # Default casual response
        else:
            return "I'm your AI Legal Adviser, specialized in legal matters and document analysis. Would you like help with understanding laws, analyzing legal documents, or answering any legal questions? ⚖️"

    def is_greeting_or_casual(self, text):
        """Check if the input is a greeting or casual conversation"""
        text_lower = text.lower().strip()
        
        # Common casual patterns
        casual_patterns = [
            r'\b(hi|hello|hey|hii|hiii|heyyy)\b',
            r'\bhow are you\b',
            r'\bhow\'s it going\b',
            r'\bgood (morning|afternoon|evening)\b',
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
            r'\bwassup\b'
        ]
        
        # Legal terms to filter out
        legal_terms = [
            'law', 'legal', 'court', 'case', 'lawyer', 'attorney',
            'contract', 'agreement', 'notice', 'sue', 'rights',
            'document', 'file', 'complaint', 'judge', 'evidence'
        ]
        
        # Check for casual patterns
        for pattern in casual_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Consider short messages without legal terms as casual
        if len(text_lower.split()) <= 3 and not any(term in text_lower for term in legal_terms):
            return True
        
        return False
