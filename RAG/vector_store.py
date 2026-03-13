import json
import chromadb
from sentence_transformers import SentenceTransformer

# ── Step 1: Load embeddings file ──────────────────────────────
with open("RAG/embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks with embeddings")

# ── Step 2: Start ChromaDB ────────────────────────────────────
client = chromadb.PersistentClient(path="RAG/chroma_db")

# ── Step 3: Create collection ─────────────────────────────────
collection = client.get_or_create_collection(
    name="meeting_chunks",
    metadata={"hnsw:space": "cosine"}
)

print("Collection ready")

# ── Step 4: Store all chunks in the collection ────────────────
ids        = []
embeddings = []
documents  = []
metadatas  = []

for chunk in chunks:
    ids.append(str(chunk["chunk_index"]))
    embeddings.append(chunk["embedding"])
    documents.append(chunk["text"])
    metadatas.append({
        "meeting_id": str(chunk["meeting_id"]),
        "chunk_id":   str(chunk["chunk_id"]),
        "word_count": str(chunk["word_count"])
    })

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas
)

print(f"Stored {len(ids)} chunks in ChromaDB")

# ── Step 5: Test similarity search ───────────────────────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

query = "What is the status of the backend API"
query_embedding = embed_model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\n── Search Results ──────────────────────")
print(f"Query: {query}\n")
for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}:")
    print(f"  Text: {doc}")
    print(f"  Meeting ID: {results['metadatas'][0][i]['meeting_id']}")
    print()
    
# ── Step 6: Save search results to file ──────────────────────
with open("RAG/search_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "query": query,
        "results": [
            {
                "rank": i + 1,
                "text": doc,
                "meeting_id": results["metadatas"][0][i]["meeting_id"]
            }
            for i, doc in enumerate(results["documents"][0])
        ]
    }, f, ensure_ascii=False, indent=2)

print("Results saved to RAG/search_results.json")
