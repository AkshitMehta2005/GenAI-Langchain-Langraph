from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()


class Review(TypedDict):
    summary: str
    sentiment: str


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.2,
)

model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(
    Review,
    method="function_calling"
)

prompt = """
Analyze this movie review:
I really loved this movie. The acting was amazing!
"""

result = structured_model.invoke(prompt)

print(result)
