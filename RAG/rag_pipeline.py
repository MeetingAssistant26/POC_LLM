import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import os
import json
import re
from dotenv import load_dotenv


load_dotenv()

chromadb.Settings(anonymized_telemetry=False)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(
    path="RAG/chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_collection("meeting_chunks")

all_data = collection.get()
texts_all = all_data["documents"]
metadatas_all = all_data["metadatas"]

print("AI Meeting Assistant Ready\n")

conversation_history = []


# 🟢 STEP 1: Get top meetings (Hierarchical Level 1)
def get_relevant_meetings(query_embedding, top_k=3):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=30
    )

    meeting_ids = [meta["meeting_id"] for meta in results["metadatas"][0]]

    # unique + keep order
    seen = set()
    unique_meetings = []
    for m in meeting_ids:
        if m not in seen:
            seen.add(m)
            unique_meetings.append(m)

    return unique_meetings[:top_k]


# 🔵 STEP 2: Retrieve chunks inside meetings (Level 2)
def retrieve_chunks_hierarchical(query, query_embedding, meeting_ids, top_k=8):

    all_texts = []
    all_metas = []

    for meeting_id in meeting_ids:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"meeting_id": meeting_id}
        )

        all_texts.extend(vector_results["documents"][0])
        all_metas.extend(vector_results["metadatas"][0])

    if not all_texts:
        return [], []

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    query_vec = vectorizer.transform([query])
    cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    top_indices = np.argsort(-cos_scores)[:10]

    tfidf_texts = [all_texts[i] for i in top_indices]
    tfidf_metas = [all_metas[i] for i in top_indices]

    # Combine + deduplicate
    unique = {}
    for text, meta in zip(all_texts + tfidf_texts, all_metas + tfidf_metas):
        if text not in unique:
            unique[text] = meta

    final_texts = list(unique.keys())
    final_metas = list(unique.values())

    # Rerank
    scores = reranker.predict([[query, t] for t in final_texts])
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [final_texts[i] for i in top_indices], [final_metas[i] for i in top_indices]


# ── Main Loop ─────────────────────────────────────────────────
while True:

    print("\nChoose an option:")
    print("1 - Generate Meeting Summary")
    print("2 - Extract Tasks")
    print("3 - Ask a Question")
    print("4 - Exit")

    choice = input("\nEnter choice (1/2/3/4): ")

    if choice == "4":
        break

    meeting_id = None

    if choice == "1":
        meeting_num = input("Enter meeting number (1-10): ")
        meeting_id = int(meeting_num)
        query = f"Summarize meeting {meeting_num}"

    elif choice == "2":
        meeting_num = input("Enter meeting number (1-10): ")
        meeting_id = int(meeting_num)
        query = f"Extract tasks from meeting {meeting_num}"

    elif choice == "3":
        query = input("\nAsk a question: ")

        match = re.search(r'meeting\s*(\d+)', query.lower())

        follow_up_words = ['he', 'she', 'his', 'her', 'they', 'their', 'it', 'this', 'that']
        is_follow_up = any(word in query.lower().split() for word in follow_up_words)

        if match:
            meeting_id = int(match.group(1))

        elif is_follow_up:
            for turn in reversed(conversation_history):
                if turn.get("meeting_id") is not None:
                    meeting_id = turn["meeting_id"]
                    break

    print("\nQuery:", query)

    query_embedding = embed_model.encode(query).tolist()

    # 🔥 Hierarchical logic
    if meeting_id is not None:
       meeting_ids = [meeting_id]
    else:
      meeting_ids = get_relevant_meetings(query_embedding)

      if not meeting_ids:
          print("[DEBUG] No meetings found, fallback to full search")
          meeting_ids = list(set([m["meeting_id"] for m in metadatas_all]))
    print(f"[DEBUG] Retrieved meetings: {meeting_ids}")

    retrieved_chunks, _ = retrieve_chunks_hierarchical(
        query, query_embedding, meeting_ids
    )

    context = "\n".join(retrieved_chunks[:8])

    history_text = ""
    for turn in conversation_history[-3:]:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    header= ""
    
    prompt = f"""
You are an AI meeting assistant.

Instructions:

- If the question is asking for a summary:
  Provide a clear and structured summary of the meeting.

  Focus ONLY on:
  - Main goal
  - Key discussion points
  - Important decisions

  STRICT RULES:
  - Do NOT list tasks.
  - Do NOT mention any assigned work.
  - Do NOT include sentences with future actions (e.g., "will", "should", "plan to").
  - Do NOT include responsibilities of individuals.
  - Keep the summary high-level and descriptive only.

- If the question is asking for tasks:
  Extract ONLY actionable tasks explicitly assigned in the meeting.

  Rules:
  - A task must be a clear action (e.g., prepare, fix, review, test, run, document).
  - Only include tasks explicitly assigned to a specific person.
  - If the task owner is not clearly mentioned, DO NOT include the task.

  - ALWAYS merge all similar names into ONE fixed name: "Sara".
  - Each person must appear ONLY once.
  - Do NOT create multiple sections for the same person.

  - Do NOT include any task that contains "decide".
  - Do NOT include discussions, observations, or past actions.
  - Do NOT infer or generate new tasks.

  - Merge similar or repeated tasks into ONE concise task.
  - Do NOT duplicate or rephrase the same task.
  - Do NOT split one task into multiple similar tasks.

  - Output tasks as a flat list (no nested bullets).
  - Output ONLY the task list.
  - Do NOT include explanations, notes, comments, or reasoning.
  - Do NOT mention removed or excluded tasks.
  - Prefer active tasks (e.g., "review", "prepare") and avoid passive ones (e.g., "receive").
  - Output MUST follow the exact format strictly.
  - Do NOT use bold formatting (**).
  - Each person name MUST start with "- " and end with ":".
  - Example:
    - Sara:
      - Task 1
    
  - STRICT RULE: Remove any task that contains "decide".
  - Only include tasks that are clearly assigned using direct instructions (e.g., "please", "try to", "can you").
  - Do NOT include general responsibilities (e.g., monitoring, observing).
 
   Format:
  - Person Name:
    - Task 1
    - Task 2

  If no tasks are found, return: "No tasks found."

- If the question is a general question:
  Answer using ONLY information explicitly mentioned in the meeting context.

  Rules:
  - Do NOT add any information that is not clearly stated.
  - Do NOT infer, assume, or generate new tasks.
  - Do NOT expand beyond the given context.
  - Use the exact meaning from the context (no extra interpretation).
  - Keep the answer concise and directly relevant to the question.
  - If partial information is found, return only what is explicitly available.
  - If no relevant information is found, say: "I don't have enough information."
  - If the question asks about tasks for a specific person:
    apply the SAME strict task extraction rules.
  - Only return tasks that are explicitly assigned to that person.

General rules:
- Do NOT repeat information.
- Do NOT mention context or sources.
- Do NOT include explanations outside the required format.

{header}

Previous conversation:
{history_text}

Meeting context:
{context}

Question:
{query}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print("\nFinal Answer:\n", answer)

    conversation_history.append({
        "question": query,
        "answer": answer,
        "meeting_id": meeting_id
    })