from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

# Step 1: Create the first prompt
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Step 2: Configure the Hugging Face model
model = HuggingFaceEndpoint(
    repo_id="YOUR_MODEL_REPOSITORY_ID",
    temperature=1,
    max_new_tokens=200
)

# Step 3: Create the output parser
parser = StrOutputParser()

# Step 4: Create the second prompt
prompt2 = PromptTemplate(
    template="Explain the following joke: {text}",
    input_variables=["text"]
)

# Step 5: Connect everything sequentially
chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

# Step 6: Execute the chain
result = chain.invoke({"topic": "AI"})

print(result)
