from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory

template = """
    Answer the question below (shortly).

    Here is the conversation history: {context}

    Question: {question}

    Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def handle_conversation():
    chat_history = InMemoryChatMessageHistory()
    print("Welcome to the Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("👻 " + "\033[46;1mYou:\033[0m ")

        if user_input.lower() == "exit":
            break

        chat_history.add_user_message(user_input)

        context = "\n".join(
            [f"{msg.type}: {msg.content}" for msg in chat_history.messages])

        response = chain.invoke({"context": context, "question": user_input})

        chat_history.add_ai_message(response)

        print("🤖 " + "\033[45;1mBot:\033[0m", response)


if __name__ == "__main__":
    handle_conversation()


# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate

# model = OllamaLLM(model="llama3")

# template = """
#     Answer the question below (shortly).

#     Here is the conversation history: {context}

#     Question: {question}

#     Answer:
# """

# prompt = ChatPromptTemplate.from_template(template)

# chain = prompt | model


# def handle_conversation():
#     context = ""
#     print("Welcome to the my Chatbot! Type 'exit' to quit.")

#     while True:
#         user_input = input("👻 " + "\033[46;1mYou:\033[0m ")

#         if user_input.lower() == "exit":
#             break

#         result = chain.invoke({"context": context, "question": user_input})
#         print("🤖 " + "\033[45;1mBot:\033[0m", result)
#         context += f"\nUser: {user_input}\nAI: {result}"


# if __name__ == "__main__":
#     handle_conversation()
