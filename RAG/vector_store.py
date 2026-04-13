import os
import json
import chromadb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_vector_store(input_path="RAG/embeddings.json", db_path="RAG/chroma_db"):
    # ── Disable telemetry ──
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    # 1. Load data
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts_all = [c["text"] for c in chunks]

    # 2. Start ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="meeting_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Add data if empty (Indexing)
    if collection.count() == 0:
        print("Adding data to ChromaDB...")
        collection.add(
            ids=[str(c["chunk_id"]) for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=texts_all,
            metadatas=[{"meeting_id": c["meeting_id"], "chunk_id": c["chunk_id"]} for c in chunks]
        )
        print(f"✅ Indexed {len(chunks)} chunks in ChromaDB.")
    else:
        print("💡 ChromaDB already contains data. Skipping indexing.")
