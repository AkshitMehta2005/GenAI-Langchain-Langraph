from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)


# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)


# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text:\n{text}",
    input_variables=["text"]
)


# Generate detailed report
prompt1 = template1.invoke({
    "topic": "black hole"
})

result = model.invoke(prompt1)


# Generate summary from the report
prompt2 = template2.invoke({
    "text": result.content
})

result1 = model.invoke(prompt2)


# Print summary
print(result1.content)