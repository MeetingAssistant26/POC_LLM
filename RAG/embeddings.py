import json
import os
from sentence_transformers import SentenceTransformer

def run_embeddings(input_path="RAG/chunks.json", output_path="RAG/embeddings.json"):
    if not os.path.exists(input_path):
        print(f"❌ Error: Chunks file {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

      
    print(f"Loaded {len(chunks)} chunks. Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded")
    
    texts = [chunk["text"] for chunk in chunks]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings of size {embeddings.shape[1]}")

    # Attach embeddings to chunks
    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        
        new_chunk = chunk.copy()
        new_chunk["embedding"] = embeddings[i].tolist()
        chunks_with_embeddings.append(new_chunk)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_with_embeddings, f, ensure_ascii=False, indent=2)

    print(f"✅ Embeddings saved to {output_path}")