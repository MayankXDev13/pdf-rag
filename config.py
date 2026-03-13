import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "pdf-rag"

EMBED_MODEL = "gemini-embedding-2-preview"
LLM_MODEL = "gemini-1.5-flash"
