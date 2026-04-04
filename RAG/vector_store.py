import os
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Disable telemetry ──
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# ── Load data ──
with open("RAG/embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts_all = [c["text"] for c in chunks]

# ── Start ChromaDB ──
client = chromadb.PersistentClient(path="RAG/chroma_db")
collection = client.get_or_create_collection(
    name="meeting_chunks",
    metadata={"hnsw:space": "cosine"}
)

# ── Add data if empty ──
if collection.count() == 0:
    collection.add(
        ids=[str(c["chunk_index"]) for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=texts_all,
        metadatas=[{"meeting_id": c["meeting_id"], "chunk_id": c["chunk_id"]} for c in chunks]
    )

# ── Models ──
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Query ──
query = "What is the status of the backend API"
query_embedding = embed_model.encode(query).tolist()

# ── 1. Vector Search (top_k = 10) ──
top_k = 10
vector_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k
)

vector_texts = vector_results["documents"][0]
vector_metadatas = vector_results["metadatas"][0]

# ── 2. TF-IDF + Cosine Similarity (top_k = 10) ──
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts_all)
query_vec = vectorizer.transform([query])
cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

top_k_indices = np.argpartition(-cos_scores, top_k-1)[:top_k]
top_k_indices = top_k_indices[np.argsort(-cos_scores[top_k_indices])]

tfidf_texts = [texts_all[i] for i in top_k_indices]
tfidf_metadatas = [
    {"meeting_id": chunks[i]["meeting_id"], "chunk_id": chunks[i]["chunk_id"]}
    for i in top_k_indices
]

# ── 3. Combine Results ──
all_texts = vector_texts + tfidf_texts
all_metadatas = vector_metadatas + tfidf_metadatas

# Remove duplicates
unique = {}
for text, meta in zip(all_texts, all_metadatas):
    if text not in unique:
        unique[text] = meta

final_texts = list(unique.keys())
final_metadatas = list(unique.values())

# ── 4. Rerank ──
rerank_inputs = [[query, t] for t in final_texts]
rerank_scores = reranker.predict(rerank_inputs)

# ── 5. Sort and take top 5 ──
sorted_idx = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
top_5 = sorted_idx[:5]

final_results = [
    {
        "rank": rank + 1,
        "text": final_texts[i],
        "meeting_id": final_metadatas[i]["meeting_id"]
    }
    for rank, i in enumerate(top_5)
]

# ── Save Results ──
with open("RAG/search_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "query": query,
        "results": final_results
    }, f, ensure_ascii=False, indent=2)

print("Results saved to RAG/search_results.json")