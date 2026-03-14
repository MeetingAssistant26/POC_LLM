import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="RAG/chroma_db")
collection = chroma_client.get_collection("meeting_chunks")

print("AI Meeting Assistant Ready\n")

# User menu
print("Choose an option:")
print("1 - Generate Meeting Summary")
print("2 - Extract Tasks")
print("3 - Ask a Question")

choice = input("\nEnter choice (1/2/3): ")

if choice == "1":
    query = "Summarize the meeting discussion"

elif choice == "2":
    query = "Extract all tasks assigned in the meeting"

elif choice == "3":
    query = input("\nAsk a question about the meeting: ")

else:
    print("Invalid choice, defaulting to question mode.")
    query = input("Ask a question about the meeting: ")

print("\nQuery:", query)

# Convert query to embedding
query_embedding = embed_model.encode(query).tolist()

# Retrieve chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]

context = "\n".join(retrieved_chunks)

print("\nRetrieved Chunks:\n")

for i, chunk in enumerate(retrieved_chunks):
    print(f"Chunk {i+1}: {chunk}\n")

# Prompt for LLM
prompt = f"""
You are an AI meeting assistant.

Using the meeting transcript context below, complete the task.

Context:
{context}

Task:
{query}

Provide a clear answer.
"""

response = groq_client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content

print("\nFinal Answer:\n")
print(answer)

# Save results folder
results_folder = "RAG/rag_results"

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

existing_files = [f for f in os.listdir(results_folder) if f.startswith("result_")]
file_number = len(existing_files) + 1

file_path = f"{results_folder}/result_{file_number}.json"

output_data = {
    "task_type": choice,
    "query": query,
    "retrieved_chunks": retrieved_chunks,
    "answer": answer
}

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\nResult saved to {file_path}")
