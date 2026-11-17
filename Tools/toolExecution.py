from langchaincore.tools import tool

# create tool

@tool 
def multiply(a:int,b:int) -> int:
    """
    given 2 numebr make them in product a*b 
    """
    return a*b


print(multiply.invoke({'a':3,'b':4}))
print(multiply.description)
print(multiply.args)


# tool binding
llm = chatOpenAI()

llm_will_tools = llm.bind_tools([multiply])



llm_will_tools.invoke("hello how are you!")
# tool calling 
llm_will_tools.invoke("3 multiply by 10 = 3*10 ").tool_calls[0]

result = llm_will_tools.invoke("3 multiply by 10 = 3*10 ")






    
