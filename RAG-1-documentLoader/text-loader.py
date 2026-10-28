import os

# Store downloaded Hugging Face models on the D drive
os.environ["HF_HOME"] = "D:/huggingface_cache"

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# 1. Load the local Hugging Face model
llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100,
        "do_sample": True,
    },
)

# 2. Convert it into a LangChain chat model
model = ChatHuggingFace(llm=llm)

# 3. Create the prompt template
prompt = PromptTemplate(
    template="Write a summary of the following text:\n\n{poem}",
    input_variables=["poem"],
)

# 4. Create the output parser
parser = StrOutputParser()

# 5. Load the text document
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# 6. Inspect the loaded document
print("Document type:", type(docs))
print("Number of documents:", len(docs))

print("\nDocument content:")
print(docs[0].page_content)

print("\nDocument metadata:")
print(docs[0].metadata)

# 7. Create the LangChain pipeline
chain = prompt | model | parser

# 8. Generate a summary using the local model
result = chain.invoke({
    "poem": docs[0].page_content
})

print("\nSummary:")
print(result)