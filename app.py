import argparse
from rag import ask


def chat_loop(filename: str | None = None, k: int = 3):
    print("PDF RAG Chatbot (Gemini + Pinecone)")
    print("Type 'exit' to quit\n")

    if filename:
        print(f"Filtering by file: {filename}\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            break

        result = ask(question, k=k, filename=filename)

        print("\nAI:", result["answer"], "\n")

        if result["sources"]:
            print("Sources:")
            for source in result["sources"]:
                print(f"  - {source['filename']} (Page {source['page']})")
            print()


def main():
    parser = argparse.ArgumentParser(description="PDF RAG Chat")
    parser.add_argument("--filename", help="Filter by specific file")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    args = parser.parse_args()

    chat_loop(filename=args.filename, k=args.k)


if __name__ == "__main__":
    main()
