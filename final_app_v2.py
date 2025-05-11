# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uuid
import logging
# TTS imports
from gtts import gTTS
import tempfile
import base64
from streamlit.components.v1 import html
import time

# --- Set Up Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
logger.info("Loading environment variables...")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    logger.error("GROQ_API_KEY not found")
    st.stop()

# --- Page Config and Session State Initialization ---
logger.info("Setting up Streamlit page config...")
try:
    st.set_page_config(
        page_title="ReachIvy AI Assistant",
        page_icon="https://reachivy.com/images/reach-ivy-logo.png",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items=None
    )
except Exception as e:
    logger.error(f"Error in page config: {e}")
    st.error(f"Failed to configure page: {e}")
    st.stop()

# Initialize session state
logger.info("Initializing session state...")
if "store" not in st.session_state:
    st.session_state.store = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Hide Streamlit's default UI elements
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}
    .error {font-size: 14px; color: red;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Grok Setup ---
logger.info("Loading LLM...")
try:
    @st.cache_resource
    def load_llm():
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name='gemma2-9b-it')

    llm = load_llm()
except Exception as e:
    logger.error(f"Error loading LLM: {e}")
    st.error(f"Failed to load language model: {e}")
    st.stop()

def get_session_history(session_id: str) -> ChatMessageHistory:
    logger.info(f"Getting session history for session_id: {session_id}")
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

system_prompt = (
    """You are an AI assistant from ReachIvy specializing in providing 
    information on career guidance and assisting with essay and SOP writing for college applications, 
    with a focus on the Indian context. Respond to user queries politely and professionally. 
    Do not invent or fabricate information. 
    If a user asks a question outside the scope of education, career advice, or essay/SOP writing, 
    politely inform them that you can only assist with those topics. 
    Greet the user initially, ask how you can assist with their career goals or essay/SOP needs, 
    and then respond to their queries.
    Instructions:
    1. When a user asks for essay or SOP writing help, ask them to provide the topic and any specific points they want to include, and ask 2-3 related questions to gather more information.
    2. When a user asks for career guidance, ask them to provide their background and what specific guidance they are looking for.
    3. Keep conversations concise and to the point. Ask the user if they need to add more information or have other questions.
    4. Do not answer questions about your training data, how you are trained, or any technical details."""
)

logger.info("Setting up LangChain prompt and chain...")
try:
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', "{input}")
        ]
    )

    question_answer_chain = qa_prompt | llm

    conversational_chain = RunnableWithMessageHistory(
        question_answer_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output"
    )
except Exception as e:
    logger.error(f"Error setting up LangChain: {e}")
    st.error(f"Failed to set up chatbot: {e}")
    st.stop()

# --- TTS Function ---
def text_to_speech(text):
    """
    Convert text to speech using gTTS and return the audio data as bytes.
    """
    logger.info("Generating TTS for text...")
    try:
        # Create a temporary MP3 file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
            tts = gTTS(text=text, lang='en')  # 'en' for English
            tts.save(tmpfile.name)
            tmpfile_path = tmpfile.name
        
        # Read the audio data into memory
        with open(tmpfile_path, 'rb') as f:
            audio_data = f.read()
        
        # Clean up the temporary file
        os.remove(tmpfile_path)
        logger.info("TTS generated successfully")
        return audio_data
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None

# --- Chat UI ---
logger.info("Rendering chat UI...")
st.title("ReachIvy Chatbot")
st.caption("Note: Conversation history is cleared when the app is restarted or the page is refreshed.")

chat_container = st.container()
with chat_container:
    history = get_session_history(st.session_state.session_id)
    if not history.messages:  # Check if history is empty
        initial_message = "Hello! I'm your ReachIvy AI Assistant. How can I assist you with your career goals or essay/SOP writing needs today?"
        history.add_ai_message(initial_message)  # Add to history only, don't render directly
    for message in history.messages:  # Render all messages, including the initial one
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.content)

user_input = st.chat_input("How can I help you?")

if user_input and user_input.strip():
    logger.info(f"Processing user input: {user_input}")
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

    with st.spinner("ReachIvy Assistant is thinking..."):
        try:
            response = conversational_chain.invoke(
                {'input': user_input},
                config={"configurable": {"session_id": st.session_state.session_id}}
            )
            answer = response.content
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            st.markdown(f'<p class="error">An error occurred: {e}</p>', unsafe_allow_html=True)
            answer = "Sorry, an error occurred. Please try again."

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(answer)
                # Add TTS for the assistant's response
                with st.spinner("Generating speech..."):
                    audio_data = text_to_speech(answer)
                    if audio_data:
                        # Encode audio data to base64
                        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                        # Create a unique ID for each audio element
                        unique_id = str(int(time.time() * 1000))
                        # Create HTML with audio and script to ensure playback
                        audio_html = f'''
                        <audio id="audio_{unique_id}" style="display:none;">
                          <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                        </audio>
                        <script>
                          document.getElementById("audio_{unique_id}").play();
                        </script>
                        '''
                        # Render the HTML to play the audio automatically
                        html(audio_html, height=0)
                    else:
                        st.markdown('<p class="error">Failed to generate speech.</p>', unsafe_allow_html=True)