
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

# Initialize chat history
chat_history = [
    SystemMessage(content="You are a helpful AI assistant.")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # Get response using complete chat history
    result = model.invoke(chat_history)

    # Add AI response to history
    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)