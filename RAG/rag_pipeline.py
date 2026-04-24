import os
import json
import re
import numpy as np
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import subprocess

# Import internal python files
from RAG.chunking import run_chunking
from RAG.embeddings import run_embeddings
from RAG.vector_store import run_vector_store

def generate_audio(text):
    print("🚀 Running TTS subprocess...")

    result = subprocess.run(
        ["venv\\Scripts\\python.exe", "TTS/coqui_tts.py", text],
        capture_output=False,
        text=True
    )

    print("✅ TTS finished")
    return "done"

def run_pipeline():
    print("Inside run_pipeline function")

    # ── Setup ──
    load_dotenv()

    # ── Smart Check: ──
    db_path = "RAG/chroma_db"

    if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
        print(" First time setup: Preparing data pipeline (Indexing documents)...")
        run_chunking()
        run_embeddings()
        run_vector_store()
        print("✅ Data pipeline preparation complete.\n")
    else:
        print("✅ System Ready: Loading existing database...\n")

    # Load Models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    collection = chroma_client.get_or_create_collection("meeting_chunks")
    if collection.count() == 0:
        print("⚠️ Collection empty, rebuilding...")
        run_chunking()
        run_embeddings()
        run_vector_store()
        collection = chroma_client.get_collection("meeting_chunks")

    all_data = collection.get()
    metadatas_all = all_data["metadatas"]

    conversation_history = []

    # 🟢 STEP 1: Get top meetings
    def get_relevant_meetings(query_embedding, top_k=3):
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=30
        )
        meeting_ids = [meta["meeting_id"] for meta in results["metadatas"][0]]
        seen = set()
        unique_meetings = []
        for m in meeting_ids:
            if m not in seen:
                seen.add(m)
                unique_meetings.append(m)
        return unique_meetings[:top_k]

    # 🔵 STEP 2: Retrieve chunks
    def retrieve_chunks_hierarchical(query, query_embedding, meeting_ids, top_k=3):
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

        # TF-IDF Re-ranking
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = vectorizer.transform([query])
        cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        top_indices_tfidf = np.argsort(-cos_scores)[:10]
        tfidf_texts = [all_texts[i] for i in top_indices_tfidf]
        tfidf_metas = [all_metas[i] for i in top_indices_tfidf]

        # Combine + Deduplicate
        unique = {}
        for text, meta in zip(all_texts + tfidf_texts, all_metas + tfidf_metas):
            if text not in unique:
                unique[text] = meta

        final_texts = list(unique.keys())
        final_metas = list(unique.values())

        # Final Rerank with CrossEncoder
        scores = reranker.predict([[query, t] for t in final_texts])
        top_indices_final = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [final_texts[i] for i in top_indices_final], [final_metas[i] for i in top_indices_final]

    # ── MAIN LOOP ──
    print("✅ AI Meeting Assistant is now active.")
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
        meeting_ids = None

        if choice == "1":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Summarize meeting {meeting_num}"
            mode = "summary"
        elif choice == "2":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Extract tasks from meeting {meeting_num}"
            mode = "tasks"
        elif choice == "3":
            query = input("Ask a question: ")
            query_lower = query.lower()
            mode = "qa"

            if "summary" in query_lower:
                mode = "summary"
            elif "task" in query_lower:
                mode = "tasks"
            else:
                mode = "qa"

            matches = re.findall(r'meeting\s*(\d+)|\b(\d+)\b', query.lower())

            meeting_ids = []
            for m1, m2 in matches:
                if m1:
                    meeting_ids.append(int(m1))
                elif m2:
                    meeting_ids.append(int(m2))

            if not meeting_ids:
                meeting_ids = None

            follow_up_words = ['he', 'she', 'his', 'her', 'they', 'their', 'it', 'this', 'that']
            is_follow_up = any(word in query.lower().split() for word in follow_up_words)

            if not matches and is_follow_up:
                for turn in reversed(conversation_history):
                    if turn.get("meeting_id") is not None:
                        meeting_id = turn["meeting_id"]
                        break
        else:
            continue

        print("\nQuery Processing:", query)
        query_embedding = embed_model.encode(query).tolist()

        # Selection Logic
        if meeting_ids is not None:
            pass
        elif meeting_id is not None:
            meeting_ids = [meeting_id]
        else:
            meeting_ids = get_relevant_meetings(query_embedding)
            if not meeting_ids:
                meeting_ids = list(set([m["meeting_id"] for m in metadatas_all]))

        print("[DEBUG] meeting_ids:", meeting_ids)
        retrieved_chunks, retrieved_metas = retrieve_chunks_hierarchical(query, query_embedding, meeting_ids)
        print("[DEBUG] retrieved_chunks:", len(retrieved_chunks))

        grouped_context = {}
        for text, meta in zip(retrieved_chunks, retrieved_metas):
            m_id = meta["meeting_id"]
            if m_id not in grouped_context:
                grouped_context[m_id] = []
            grouped_context[m_id].append(text)

        context = ""
        for m_id, texts in grouped_context.items():
            context += f"\n### Meeting {m_id}:\n"
            context += "\n".join(texts[:4])

        history_text = ""
        for turn in conversation_history[-3:]:
            history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        # ── Language Detection ──
        arabic_chars = sum(1 for c in context if '\u0600' <= c <= '\u06FF')
        total_chars = len(context.replace(" ", ""))
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0

        colloquial_words = ['عايز', 'مش', 'كده', 'إيه', 'عشان', 'بتاع', 'هنعمل', 'بيجي', 
                    'لقيت', 'هبدأ', 'يالا', 'تمام', 'ممتاز', 'هعمل', 'هتيست', 
                    'هراجع', 'دلوقتي', 'إحنا', 'احنا', 'بيعمل', 'هيعمل', 'مفيش',
                    'فيه', 'عليه', 'بقى', 'كمان', 'لو', 'ده', 'دي', 'هنا']
        colloquial_count = sum(1 for w in colloquial_words if w in context)

        if arabic_ratio > 0.1:
            if colloquial_count >= 1:
                lang_instruction = """The meeting is in Egyptian Arabic dialect (عامية مصرية).
You MUST respond in Egyptian spoken Arabic.

RULES:
- Never use formal words like: يجب، اتضح، ينبغي، حيث، إذ
- Use instead: لازم، اتكلموا عن، عشان، لما
- Use simple natural Egyptian Arabic like you are telling a friend
- Keep technical words in English (onboarding, push notifications, backend, mockups)
- Avoid formal Arabic words like: تم، هذه، هذا، حيث، إذ، لذلك
- Use instead: اتعمل، ده، دي، عشان، لما

For summary:
- Start with: "الميتينج كان عن..."
- Write 2 to 3 simple sentences only
- No bullet points or stars
- Only use information from the context, do not add anything

For tasks: output ONLY valid JSON

For general question:
Rules:
  - Always answer in a full sentence, never return a name or word alone.
  - If the answer is a person, say "X هو المسؤول عن..." or "X is responsible for..."
  - Do NOT add any information that is not clearly stated.
  - Do NOT infer, assume, or generate new tasks.


"""

            else:
                lang_instruction = """The meeting is in Modern Standard Arabic (فصحى).
You MUST respond in Modern Standard Arabic only.
- Keep technical words in English
- For summary: write in flowing sentences, no bullet points or stars (*)
- Do NOT invent information not explicitly in the context
- For tasks: output ONLY valid JSON"""
        else:
            lang_instruction = """The meeting is in English.
Respond in English only.
- For summary: write in flowing sentences, no bullet points or stars (*)
- Do NOT invent information not explicitly in the context
- For tasks: output ONLY valid JSON"""

        prompt = f"""
You are an AI meeting assistant.
{lang_instruction}
IMPORTANT:

- If the question asks about "goal" or "goals", return ONLY the goal directly.
- Do NOT generate a full summary unless explicitly asked.
- If mode is "qa", NEVER return a summary.

Instructions:

- If the question is asking for a summary:
  Provide a clear and structured summary of the meeting.

  Focus ONLY on:
  - Main goal
  - Key discussion points
  - Important decisions

  STRICT RULES:
  - Start with: "الميتينج كان عن..." if Arabic, or "The meeting was about..." if English
  - Write 2 to 3 simple sentences only
  - No bullet points or stars
  - Do NOT list tasks.
  - Do NOT mention any assigned work.
  - Do NOT include sentences with future actions (e.g., "will", "should", "plan to").
  - Do NOT include responsibilities of individuals.
  - Keep the summary high-level and descriptive only.
  - Convert any task-like statements into general discussion points (do NOT mention names or assignments).
  - NEVER use these Arabic words: يجب، اتضح، ينبغي، حيث، إذ، لذلك، نظراً
  - Use instead: لازم، اتكلموا، عشان، وكمان

- If the question is asking for tasks:
  Extract ONLY actionable tasks explicitly assigned in the meeting.
  Output MUST be in valid JSON format only.

  JSON Structure:
  {{
    "tasks": [
      {{
        "assignee": "Person Name",
        "task": "task description",
        "due_date": "deadline if mentioned, otherwise empty string"
      }}
    ]
  }}
  Rules:
  - A task must be a clear action (review, clean, test, prepare, fix, document).
  - Only include tasks explicitly assigned to a person.
  - If no assignee is mentioned, ignore the task.
  - Keep original names as they appear in the transcript.
  - Do NOT normalize or change any names.
  - Do NOT include tasks containing "decide".
  - Do NOT include discussions or observations.
  - Merge similar tasks into one.
  - Keep task description short and clear.
  - Write task description in third person (e.g., "يعمل full audit" not "أعمل full audit")
  - Output ONLY the JSON, no text before or after.
  - Do NOT infer tasks that are not explicitly stated in the meeting.
  - Do NOT rephrase or creatively rewrite tasks beyond the original meaning.
  - Use only information directly present in the provided context.
  - Do NOT add new tasks even if they seem logically implied.
  - Preserve exact meaning of due dates as written in the text (do not reformat or "clean" them).
  - Do NOT correct or modify names in any way except normalization to "Sara".
  If no tasks found return:
  {{
    "tasks": []
  }}

- If the question is a general question:
  Answer using ONLY information explicitly mentioned in the meeting context.

  Rules:
  - Always answer in a full sentence, never return a name or word alone.
  - If the answer is a person, say "X هو المسؤول عن..." or "X is responsible for..."
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
  - When comparing, clearly highlight differences instead of similarities unless explicitly stated.

General rules:
- Do NOT repeat information.
- Do NOT mention context or sources.
- Do NOT include explanations outside the required format.

Previous conversation:
{history_text}
Meeting context:
{context}
Mode: {mode}
Question:
{query}
"""

        # API Call
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        print("✅ Answer generated")
        print(answer)

        print("➡️ Going to TTS...")
        audio_path = generate_audio(answer)

        print("\nFinal Answer:\n", answer)

        conversation_history.append({
            "question": query,
            "answer": answer,
            "meeting_id": meeting_id
        })

        # --- Save results to the rag_results folder ---
        results_dir = "RAG/rag_results"
        os.makedirs(results_dir, exist_ok=True)

        current_files = os.listdir(results_dir)
        file_count = len([f for f in current_files if f.endswith('.json')]) + 1
        file_path = f"{results_dir}/result_{file_count}.json"

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "query_number": file_count,
                "meeting_id": meeting_id,
                "question": query,
                "answer": answer,
                "audio_path": audio_path,
                "context_used": retrieved_chunks
            }, f, ensure_ascii=False, indent=2)

        print(f"📂 Result saved to: {file_path}")