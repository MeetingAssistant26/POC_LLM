import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
from TTS.api import TTS
import os
import json
import re
import time
import wave  
from dotenv import load_dotenv

# ── Load environment variables ─────────────────────────────────
load_dotenv()

# ── Disable chroma telemetry warning ──────────────────────────
chromadb.Settings(anonymized_telemetry=False)

# ── Initialize embedding, reranking, and LLM models ───────────
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Load TTS model once to avoid reloading on every query ─────
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# ── Connect to ChromaDB persistent vector store ───────────────
chroma_client = chromadb.PersistentClient(path="TTS/chroma_db")
collection = chroma_client.get_collection("meeting_chunks")

# ── Load all documents and metadata for TF-IDF search ─────────
all_data = collection.get()
texts_all = all_data["documents"]
metadatas_all = all_data["metadatas"]

print("AI Meeting Assistant Ready\n")

# ── Initialize conversation history for multi-turn context ────
conversation_history = []


# ── Clean answer text before passing to TTS ───────────────────
def clean_for_tts(text):
    text = text.replace("-", "")
    text = text.replace("\n\n", ". ")
    text = text.replace("\n", ", ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Split long text into smaller chunks to avoid XTTS token limit
def split_text(text, max_len=800):
    sentences = text.split(". ")
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "

    if current:
        chunks.append(current.strip())

    return chunks


def retrieve_chunks(query, query_embedding, meeting_id=None, top_k=8):
    """Retrieve chunks using Vector Search + TF-IDF + Reranking"""

    # ── 1. Vector search using query embedding ────────────────
    if meeting_id is not None:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where={"meeting_id": meeting_id}
        )
    else:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

    vector_texts = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]

    # ── 2. TF-IDF keyword search on filtered documents ────────
    if meeting_id is not None:
        filtered_texts = [t for t, m in zip(texts_all, metadatas_all) if m.get("meeting_id") == meeting_id]
        filtered_metas = [m for m in metadatas_all if m.get("meeting_id") == meeting_id]
    else:
        filtered_texts = texts_all
        filtered_metas = metadatas_all

    if filtered_texts:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_texts)
        query_vec = vectorizer.transform([query])
        cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_indices = np.argsort(-cos_scores)[:10]
        tfidf_texts = [filtered_texts[i] for i in top_indices]
        tfidf_metadatas = [filtered_metas[i] for i in top_indices]
    else:
        tfidf_texts = []
        tfidf_metadatas = []

    # ── 3. Merge vector and TF-IDF results, remove duplicates ─
    unique = {}
    for text, meta in zip(vector_texts + tfidf_texts, vector_metadatas + tfidf_metadatas):
        if text not in unique:
            unique[text] = meta

    final_texts = list(unique.keys())
    final_metadatas = list(unique.values())

    if not final_texts:
        return [], []

    # ── 4. Rerank merged results using cross-encoder ──────────
    scores = reranker.predict([[query, t] for t in final_texts])
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    retrieved_chunks = [final_texts[i] for i in top_indices]
    retrieved_metadatas = [final_metadatas[i] for i in top_indices]

    return retrieved_chunks, retrieved_metadatas
# ── Helper functions for text processing and audio generation ──

def clean_for_tts(text):
    text = re.sub(r'[^a-zA-Z0-9.,!?\'" ]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(text, max_len=500):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 5]


def merge_wav_files(input_files, output_file):
    import wave

    data = []
    params = None

    for file in input_files:
        with wave.open(file, 'rb') as w:
            if params is None:
                params = w.getparams()
            data.append(w.readframes(w.getnframes()))

    with wave.open(output_file, 'wb') as output:
        output.setparams(params)
        for frames in data:
            output.writeframes(frames)

# ── Main interaction loop ──────────────────────────────────────
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

    meeting_id = None

    # ── Build query based on user choice ──────────────────────
    if choice == "1":
        meeting_num = input("Enter meeting number (1-10): ")
        meeting_id = int(meeting_num)
        query = f"Summarize meeting {meeting_num} discussion and key points"

    elif choice == "2":
        meeting_num = input("Enter meeting number (1-10): ")
        meeting_id = int(meeting_num)
        query = f"List ALL tasks assigned to each person in meeting {meeting_num} with deadlines"

    elif choice == "3":
        query = input("\nAsk a question about the meetings: ")

        # ── Check if a meeting number is mentioned in the query
        match = re.search(r'meeting\s*(\d+)', query.lower())

        # ── Check if query is a follow-up using pronouns ──────
        follow_up_words = ['he', 'she', 'his', 'her', 'they', 'their', 'it', 'this', 'that']
        is_follow_up = any(word in query.lower().split() for word in follow_up_words)

        if match:
            # ── Meeting number found directly in query ────────
            meeting_id = int(match.group(1))
            print(f"\n[DEBUG] Meeting detected in question: {meeting_id}")

        elif is_follow_up:
            # ── Follow-up: reuse last meeting_id from history ─
            for turn in reversed(conversation_history):
                if turn.get("meeting_id") is not None:
                    meeting_id = turn["meeting_id"]
                    print(f"\n[DEBUG] Follow-up question, using meeting from history: {meeting_id}")
                    break

        else:
            # ── No meeting specified: use last known meeting ──
            if conversation_history:
                for turn in reversed(conversation_history):
                    if turn.get("meeting_id") is not None:
                        meeting_id = turn["meeting_id"]
                        print(f"\n[DEBUG] No meeting specified, using last meeting: {meeting_id}")
                        break

            if meeting_id is None:
                print("\n[DEBUG] No meeting context found, please specify meeting number.")

    else:
        print("Invalid choice.")
        continue

    print("\nQuery:", query)

    # ── Encode query into embedding vector ────────────────────
    query_embedding = embed_model.encode(query).tolist()

    # ── Retrieve relevant chunks from vector store ────────────
    if meeting_id is not None:
        print(f"\n[DEBUG] Filtering by meeting_id = {meeting_id}")

    retrieved_chunks, retrieved_metadatas = retrieve_chunks(
        query, query_embedding, meeting_id=meeting_id
    )

    # ── Build context string from top retrieved chunks ────────
    context = "\n".join(retrieved_chunks[:5]) if retrieved_chunks else ""

    print("\nRetrieved Chunks:\n")
    if not retrieved_chunks:
        print("No chunks retrieved!")
    else:
        for i, chunk in enumerate(retrieved_chunks):
            print(f"Chunk {i+1}: {chunk}\n")

    # ── Build conversation history string for prompt ──────────
    history_text = ""
    for turn in conversation_history[-3:]:
        history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    if not history_text.strip():
        history_text = "No previous conversation."

    # ── Set answer header based on task type ──────────────────
    if choice == "1":
        header = f"Start your answer with: 'Meeting {meeting_num} Summary:'"
    elif choice == "2":
        header = f"Start your answer with: 'Meeting {meeting_num} Tasks:'"
    else:
        header = ""

    # ── Build final prompt for LLM ────────────────────────────
    prompt = f"""
You are an AI meeting assistant.

Instructions:
- If the question is asking for a summary:
  Summarize the meeting clearly.
  Focus on:
  - The main goal
  - Key discussion points
  - Important decisions
  Do NOT list tasks or assign tasks.

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
  - Prefer active tasks (e.g., "review", "prepare") and avoid passive ones (e.g., "receive").
  
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

General rules:
- Do NOT repeat information.
- Do NOT mention context or sources.

{header}

Previous conversation:
{history_text}

Meeting context:
{context}

Question:
{query}
"""

    # ── Send prompt to LLM and get response ───────────────────
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    print("\nFinal Answer:\n")
    print(answer)

    # ── Append current turn to conversation history ───────────
    conversation_history.append({
        "question": query,
        "answer": answer,
        "meeting_id": meeting_id
    })

    # ── Save result as JSON file in results folder ────────────
    results_folder = "TTS/tts_results"
    os.makedirs(results_folder, exist_ok=True)

    existing_files = [f for f in os.listdir(results_folder) if f.startswith("result_")]
    file_number = len(existing_files) + 1
    file_path = f"{results_folder}/result_{file_number}.json"

    output_data = {
        "task_type": choice,
        "query": query,
        "meeting_id": meeting_id,
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "conversation_history": conversation_history
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved to {file_path}")

    # ── Clean answer text and convert to speech using TTS ─────
    clean_text = clean_for_tts(answer)
    output_audio = f"TTS/tts_results/audio_{int(time.time())}.wav"

    # Split long text into chunks to avoid XTTS token limit
    chunks = split_text(clean_text)

    audio_files = []

  # Generate audio for each chunk separately
    for i, chunk in enumerate(chunks):
        temp_audio = f"TTS/tts_results/temp_{i}.wav"


        tts.tts_to_file(
          text=chunk,
          speaker_wav="TTS/reference.wav",
          language="en",
          file_path=temp_audio
    )

        audio_files.append(temp_audio)

 # Merge all generated audio chunks into one final file
    existing_files = [f for f in os.listdir("TTS/tts_results") if f.startswith("audio_") and f.endswith(".wav")]
    file_number = len(existing_files) + 1

    final_audio = f"TTS/tts_results/audio_{file_number}.wav"
    merge_wav_files(audio_files, final_audio)

    print(f"Final audio saved to {final_audio}")

   # Automatically play the final audio file
    os.startfile(os.path.abspath(final_audio))

   # delete temporary chunk files after merging
    for f in audio_files:
        if os.path.exists(f):
          os.remove(f)

   