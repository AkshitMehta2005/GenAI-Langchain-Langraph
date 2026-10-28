from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('D:\GenAI\RAG-1-documentLoader\pdf\Building Machine Learning Systems with Python - Second Edition.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)