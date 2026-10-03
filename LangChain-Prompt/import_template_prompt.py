
from dotenv import load_dotenv
import os
import streamlit as st

from langchain_core.load import loads
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

# Load environment variables
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# Check Hugging Face token
if not hf_token:
    st.error("HF_TOKEN is missing. Please check your .env file.")
    st.stop()

# Configure the Hugging Face Qwen model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=1024,
    temperature=0.7,
)

# Convert endpoint to chat model
model = ChatHuggingFace(llm=llm)

# Load the saved prompt template
try:
    with open("template.json", "r", encoding="utf-8") as file:
        template = loads(file.read())
except FileNotFoundError:
    st.error("template.json was not found in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Could not load prompt template: {e}")
    st.stop()

# Streamlit page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered",
)

# Page title
st.title("📚 Research Paper Summarizer")
st.write("Summarize research papers using the Qwen AI model.")

# Select research paper
paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

# Select explanation style
style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical",
    ],
)

# Select explanation length
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
    try:
        # Combine the prompt template and Qwen model
        chain = template | model

        # Display loading message
        with st.spinner("Qwen is generating your summary..."):
            result = chain.invoke(
                {
                    "paper_input": paper_input,
                    "style_input": style_input,
                    "length_input": length_input,
                }
            )

        # Display result
        st.subheader("Research Paper Summary")
        st.markdown(result.content)

    except Exception as e:
        st.error(f"Error generating summary: {e}")