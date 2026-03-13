import argparse
import sys
from pathlib import Path

from config import INDEX_NAME
from pdf_utils import load_pdf, chunk_text, get_file_hash, get_filename
from vectorstore import create_index, file_exists_in_index, upsert_chunks, delete_file, list_indexed_files
from rag import ask


def ingest_command(args):
    pdf_path = args.pdf
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Loading PDF: {pdf_path}")
    pages, num_pages = load_pdf(pdf_path)
    print(f"Loaded {num_pages} pages")
    
    print("Chunking text...")
    chunks = chunk_text(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"Created {len(chunks)} chunks")
    
    filename = get_filename(pdf_path)
    file_hash = get_file_hash(pdf_path)
    
    if args.rebuild:
        print(f"Rebuilding index, deleting existing data for: {filename}")
        delete_file(filename, file_hash)
    else:
        if file_exists_in_index(filename, file_hash):
            print(f"File already indexed. Use --rebuild to re-index.")
            sys.exit(0)
    
    print("Creating embeddings and upserting to Pinecone...")
    upsert_chunks(chunks, filename, file_hash)
    
    print(f"Successfully indexed: {filename}")


def query_command(args):
    result = ask(
        question=args.question,
        k=args.k,
        filename=args.filename,
        show_sources=True
    )
    
    print("\n" + "=" * 50)
    print("ANSWER:")
    print("=" * 50)
    print(result["answer"])
    
    if args.show_sources and result["sources"]:
        print("\n" + "=" * 50)
        print("SOURCES:")
        print("=" * 50)
        for i, source in enumerate(result["sources"], 1):
            print(f"\n[{i}] {source['filename']} - Page {source['page']}")
            print(f"    Score: {source['score']:.4f}")
            print(f"    Text: {source['text']}")


def list_command(args):
    files = list_indexed_files()
    if files:
        print("Indexed files:")
        for f in files:
            print(f"  - {f}")
    else:
        print("No files indexed yet.")


def main():
    parser = argparse.ArgumentParser(description="PDF RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF file")
    ingest_parser.add_argument("--pdf", required=True, help="Path to PDF file")
    ingest_parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size")
    ingest_parser.add_argument("--overlap", type=int, default=100, help="Chunk overlap")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Rebuild index for file")
    ingest_parser.set_defaults(func=ingest_command)
    
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    query_parser.add_argument("--filename", help="Filter by specific file")
    query_parser.add_argument("--show-sources", action="store_true", default=True, help="Show sources")
    query_parser.set_defaults(func=query_command)
    
    list_parser = subparsers.add_parser("list", help="List indexed files")
    list_parser.set_defaults(func=list_command)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
