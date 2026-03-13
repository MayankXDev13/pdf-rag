import google.generativeai as genai
from pinecone import Pinecone

from config import *

genai.configure(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(INDEX_NAME)

model = genai.GenerativeModel(LLM_MODEL)


def get_embedding(text):

    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )

    return result["embedding"]


def retrieve(query, k=3):

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )

    contexts = []

    for match in results["matches"]:
        contexts.append(match["metadata"]["text"])

    return contexts


def ask(question):

    contexts = retrieve(question)

    context_text = "\n".join(contexts)

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context_text}

    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text