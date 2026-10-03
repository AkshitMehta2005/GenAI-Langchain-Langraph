
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

# Load environment variables
load_dotenv(override=True)

HF_TOKEN = os.getenv("HF_TOKEN")

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

# Custom styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    h1 {
        text-align: center;
        color: #60a5fa;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chatbot")
st.markdown(
    '<p class="subtitle">Powered by Qwen + Hugging Face + LangChain</p>',
    unsafe_allow_html=True,
)

if not HF_TOKEN:
    st.error(
        "HF_TOKEN is missing. Please check your .env file."
    )
    st.stop()


# Initialize the language model once per app session
@st.cache_resource
def load_model():
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.7,
    )

    return ChatHuggingFace(llm=llm)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model initialization failed: {e}")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar
with st.sidebar:
    st.header("Settings")

    st.write("Model: Qwen3-4B-Instruct-2507")
    st.write("Provider: Hugging Face")
    st.write("Output: Streaming")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Prompt input
prompt = st.chat_input("Ask me anything...")

if prompt:
    # Display user prompt
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate streaming response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Connecting to AI..."):
                response_stream = model.stream(prompt)

            response = st.write_stream(response_stream)

            # Save generated response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })

        except Exception as e:
            st.error(f"Error generating response: {e}")