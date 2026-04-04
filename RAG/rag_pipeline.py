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

# ── Conversation History ──────────────────────────────────────
conversation_history = []

# ── Main Loop ─────────────────────────────────────────────────
while True:

    print("\nChoose an option:")
    print("1 - Generate Meeting Summary")
    print("2 - Extract Tasks")
    print("3 - Ask a Question")
    print("4 - Exit")

    choice = input("\nEnter choice (1/2/3/4): ")

    if choice == "4":
        print("Goodbye!")
        break

    elif choice == "1":
        query = "Summarize the meeting discussion"

    elif choice == "2":
        query = "Extract all tasks assigned in the meeting"

    elif choice == "3":
        query = input("\nAsk a question about the meeting: ")

    else:
        print("Invalid choice.")
        continue

    print("\nQuery:", query)

    # ── Improve Retrieval using History ────────────────────────
    history_questions = " ".join(
        [turn["question"] for turn in conversation_history[-2:]]
    )

    enhanced_query = query + " " + history_questions

    query_embedding = embed_model.encode(enhanced_query).tolist()

    # Increased top_k for better retrieval
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    retrieved_chunks = results["documents"][0]
    context = "\n".join(retrieved_chunks)

    print("\nRetrieved Chunks:\n")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1}: {chunk}\n")

    # ── Build History Text ────────────────────────────────────
    history_text = ""

    for turn in conversation_history[-3:]:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    if history_text.strip() == "":
        history_text = "No previous conversation."

    # ── Improved Prompt ───────────────────────────────────────
    prompt = f"""
You are an AI meeting assistant.
Use BOTH the conversation history and the meeting context to answer.
Answer directly without mentioning the sources or context in your response.
Do not say "Based on..." or "According to..." or "From the context...".
If the answer is not available, say "I don't have enough information."

Previous conversation:
{history_text}

Meeting context:
{context}

Current question:
{query}
"""

    # ── LLM Call ──────────────────────────────────────────────
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print("\nFinal Answer:\n")
    print(answer)

    # ── Save to History ───────────────────────────────────────
    conversation_history.append({
        "question": query,
        "answer": answer
    })

    # ── Save Result ───────────────────────────────────────────
    results_folder = "RAG/rag_results"
    os.makedirs(results_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(results_folder) if f.startswith("result_")]
    file_number = len(existing_files) + 1
    file_path = f"{results_folder}/result_{file_number}.json"

    output_data = {
        "task_type": choice,
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "conversation_history": conversation_history
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved to {file_path}")