
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.7,
)

# Convert LLM into a chat model
model = ChatHuggingFace(llm=llm)


# chat history

chat_history = []
# Chat loop
while True:
    user_input = input("You: ")
    chat_history.append(user_input)

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI:", result.content)