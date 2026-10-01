
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env
load_dotenv(override=True)

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN is missing. Check your .env file.")

# Connect to the Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=100,
    temperature=0.7,
)

# Convert the endpoint into a chat model
model = ChatHuggingFace(llm=llm)

# Send a prompt
response = model.invoke("What is the capital of India?")

# Display the response
print(response.content)