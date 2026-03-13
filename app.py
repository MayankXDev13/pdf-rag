from rag import ask

print("PDF RAG Chatbot (Gemini + Pinecone)")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    answer = ask(question)

    print("\nAI:", answer, "\n")