
while True:
    user_input = input("You: ")
    chat_history.append(user_input)

    if user_input.lower() == "exit":
        print("Chat ended.")
        break

    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI:", result.content)
