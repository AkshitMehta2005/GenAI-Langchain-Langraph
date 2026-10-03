
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_core.prompts import load_prompt

# Load environment variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    st.error("HF_TOKEN is missing. Check your .env file.")
    st.stop()

# Configure Qwen model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=1024,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

# Streamlit page
st.set_page_config(
    page_title="Research Paper Tool",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Research Paper Tool")
st.write("Summarize research papers using Qwen AI.")

# User inputs
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

# Generate summary
if st.button("Summarize", use_container_width=True):
    prompt = f"""
    You are an AI research assistant.

    Explain the following research paper:
    Paper: {paper_input}

    Explanation style: {style_input}
    Explanation length: {length_input}

    Instructions:
    1. Explain the paper's main objective.
    2. Describe its key methodology.
    3. Explain the major findings and contributions.
    4. Include limitations when relevant.
    5. Follow the requested explanation style and length.
    6. Do not invent research findings or citations.

    Note: If you do not know specific details about the paper,
    clearly mention the uncertainty.
    """

    try:
        with st.spinner("Qwen is generating the summary..."):
            response = model.invoke(prompt)

        st.subheader("Research Paper Summary")
        st.markdown(response.content)

    except Exception as e:
        st.error(f"Error generating summary: {e}")