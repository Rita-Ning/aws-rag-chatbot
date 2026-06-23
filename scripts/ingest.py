import os
import time
import sys
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY not set.")
    sys.exit(1)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def load_pdfs(data_dir):
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: No PDF files found in ./{data_dir}/")
        sys.exit(1)
    all_docs = []
    print(f"Found {len(pdf_files)} PDF(s):\n")
    for filename in pdf_files:
        filepath = os.path.join(data_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Loading: {filename} ({size_mb:.1f} MB)...")
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
        print(f"    → {len(docs)} pages loaded")
        all_docs.extend(docs)
    print(f"\nTotal pages loaded: {len(all_docs)}")
    return all_docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def sanity_check(chunks, n=3):
    import random
    print(f"\n=== Sanity check: {n} random chunks ===\n")
    samples = random.sample(chunks, min(n, len(chunks)))
    for i, chunk in enumerate(samples, 1):
        source = chunk.metadata.get("source_file", "unknown")
        page = chunk.metadata.get("page", "?")
        preview = chunk.page_content[:200].replace("\n", " ")
        print(f"[{i}] {source} | Page {page}")
        print(f"    {preview}...")
        print()

def embed_and_save(chunks):
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    print("Embedding chunks... (需要幾分鐘，費用約 $0.01)\n")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"  Batch {batch_num}/{total_batches}...")
        time.sleep(2)
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\n✓ FAISS index 已儲存到 ./{VECTORSTORE_DIR}/")
    return vectorstore

def quick_retrieval_test(vectorstore):
    print("\n=== 測試 retrieval ===\n")
    test_query = "What is Amazon Bedrock and what are its main use cases?"
    print(f"Query: '{test_query}'\n")
    results = vectorstore.similarity_search(test_query, k=3)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"Result {i} | {source} p.{page}")
        print(f"  {preview}...")
        print()
    print("✓ Retrieval 正常！Day 1 完成！")

def main():
    print("=== Day 1: PDF Ingestion Pipeline ===\n")
    docs = load_pdfs(DATA_DIR)
    print()
    chunks = chunk_documents(docs)
    sanity_check(chunks)
    vectorstore = embed_and_save(chunks)
    quick_retrieval_test(vectorstore)
    print("\n=== Day 1 完成！明天繼續 Day 2 ===")

if __name__ == "__main__":
    main()