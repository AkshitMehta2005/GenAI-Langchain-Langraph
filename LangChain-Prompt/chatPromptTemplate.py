from langchain_core.prompts import chatPromptTemplate


chat_template = chatPromptTemplate([
    {'system',"You are the helpful {domain} expert"},
    {'human',"Explain in simple terms,what is {topic}"},
])

prompt = chat_template.invoke({'domain':"cricket","topic":"dusra"})


print(prompt) 