import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Load chunks from chunking step
with open("RAG/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# Load embedding model
print("Loading embedding model")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded")

# Extract texts to embed
texts = [chunk["text"] for chunk in chunks]

# Generate embeddings
print("Generating embeddings")
embeddings = model.encode(texts, show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings of size {embeddings.shape[1]}")

# Attach embeddings to chunks
chunks_with_embeddings = []
for i, chunk in enumerate(chunks):
    chunks_with_embeddings.append({
        "meeting_id": chunk["meeting_id"],
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "word_count": chunk["word_count"],
        "text": chunk["text"],
        "embedding": embeddings[i].tolist()
    })

# Save to file
with open("RAG/embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)

print("Embeddings saved to RAG/embeddings.json")
print(f"Total chunks embedded: {len(chunks_with_embeddings)}")