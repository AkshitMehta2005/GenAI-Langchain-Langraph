from langchain_huggingFace import huggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parser import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence


load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variable = ['topic']
)


model = huggingFaceEndpoint(
    repo_id="queen-62",
    temperature = 1
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,parser)

print(chain.invoke({'topic':'AI'}))
