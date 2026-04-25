import os
import json
import re
import numpy as np
np.float_ = np.float64
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Import internal python files
from RAG.chunking import run_chunking
from RAG.embeddings import run_embeddings
from RAG.vector_store import run_vector_store


# ── Load prompt templates from files ──────────────────────────
def load_prompt(filename):
    prompt_path = os.path.join("prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# TTS is disabled — uncomment below and remove the stub to re-enable
# import subprocess
# def generate_audio(text):
#     print("🚀 Running TTS subprocess...")
#     subprocess.run(
#         ["venv\\Scripts\\python.exe", "TTS/coqui_tts.py", text],
#         capture_output=False,
#         text=True
#     )
#     print("✅ TTS finished")
#     return "done"

def generate_audio(text):
    print("🔇 TTS is currently disabled.")
    return "disabled"


def run_pipeline():
    print("Inside run_pipeline function")

    load_dotenv()

    # ── Load prompts once at startup ──
    summary_prompt_template = load_prompt("meeting_summary_prompt.txt")
    task_prompt_template = load_prompt("task_extraction_prompt.txt")

    # ── Smart Check ──
    db_path = "RAG/chroma_db"
    if not os.path.exists(db_path) or len(os.listdir(db_path)) == 0:
        print("First time setup: Preparing data pipeline...")
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
        print(" Collection empty, rebuilding...")
        run_chunking()
        run_embeddings()
        run_vector_store()
        collection = chroma_client.get_collection("meeting_chunks")

    all_data = collection.get()
    metadatas_all = all_data["metadatas"]
    conversation_history = []

    # ── STEP 1: Get top meetings ──
    # FIX: Increased candidate pool (n_results) and top_k so more meetings
    # are considered for open-ended questions that don't name a meeting.
    def get_relevant_meetings(query_embedding, top_k=5):
        total_chunks = collection.count()
        # Pull a large candidate pool so no relevant meeting is missed
        n_candidates = min(total_chunks, 100)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates
        )
        meeting_ids = [meta["meeting_id"] for meta in results["metadatas"][0]]
        seen = set()
        unique_meetings = []
        for m in meeting_ids:
            if m not in seen:
                seen.add(m)
                unique_meetings.append(m)
        return unique_meetings[:top_k]

    # ── STEP 2: Retrieve chunks ──
    # FIX: When no meeting is specified (search_all=True) we query ALL
    # meetings in one go instead of looping per meeting ID, then rerank.
    def retrieve_chunks_hierarchical(query, query_embedding, meeting_ids,
                                     top_k=8, search_all=False):
        all_texts = []
        all_metas = []

        if search_all:
            # Query across the entire collection without a meeting filter
            total_chunks = collection.count()
            n_candidates = min(total_chunks, 60)
            vector_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_candidates
            )
            all_texts.extend(vector_results["documents"][0])
            all_metas.extend(vector_results["metadatas"][0])
        else:
            for meeting_id in meeting_ids:
                vector_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=20,
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

        top_indices_tfidf = np.argsort(-cos_scores)[:20]
        tfidf_texts = [all_texts[i] for i in top_indices_tfidf]
        tfidf_metas = [all_metas[i] for i in top_indices_tfidf]

        # Combine + Deduplicate
        unique = {}
        for text, meta in zip(all_texts + tfidf_texts, all_metas + tfidf_metas):
            if text not in unique:
                unique[text] = meta

        final_texts = list(unique.keys())
        final_metas = list(unique.values())

        # Rerank with CrossEncoder
        scores = reranker.predict([[query, t] for t in final_texts])
        top_indices_final = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [final_texts[i] for i in top_indices_final], \
               [final_metas[i] for i in top_indices_final]

    # ── Language Detection ──
    def detect_language(context):
        arabic_chars = sum(1 for c in context if '\u0600' <= c <= '\u06FF')
        total_chars = len(context.replace(" ", ""))
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0

        colloquial_words = [
            'عايز', 'مش', 'كده', 'إيه', 'عشان', 'بتاع', 'هنعمل', 'بيجي',
            'لقيت', 'هبدأ', 'يالا', 'تمام', 'هعمل', 'هتيست', 'هراجع',
            'دلوقتي', 'إحنا', 'احنا', 'بيعمل', 'هيعمل', 'مفيش',
            'فيه', 'عليه', 'بقى', 'كمان', 'لو', 'ده', 'دي', 'هنا'
        ]
        colloquial_count = sum(1 for w in colloquial_words if w in context)

        if arabic_ratio > 0.1:
            return "ar_colloquial" if colloquial_count >= 1 else "ar_formal"
        return "en"

    # ── Build QA prompt (inline, not from file) ──
    def build_qa_prompt(query, context, history_text, lang):
        if lang == "ar_colloquial":
            lang_instruction = "اللغة: عامية مصرية. رد بالعامية المصرية الطبيعية دايماً."
        elif lang == "ar_formal":
            lang_instruction = "اللغة: عربي فصحى. رد بالعربي الفصحى."
        else:
            lang_instruction = "Language: English. Always respond in English."

        return f"""
You are an AI meeting assistant.
{lang_instruction}

Answer the question using ONLY information explicitly mentioned in the meeting context.
The context may contain chunks from multiple different meetings — use all of them.

Rules:
- Always answer in a full sentence, never return a name or word alone.
- If the answer is a person, say "X is responsible for..." or "X هو المسؤول عن..."
- If the answer spans multiple meetings, mention which meeting each fact comes from.
- Do NOT add information not clearly stated in the context.
- Do NOT infer or assume anything beyond what is written.
- Keep the answer concise and directly relevant to the question.
- If no relevant info found, say: "I don't have enough information about this in the meetings."

Previous conversation:
{history_text}

Context:
{context}

Question: {query}
"""

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
        # FIX: track whether user asked about a specific meeting or not
        search_all = False

        if choice == "1":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Summarize meeting {meeting_num}"
            mode = "summary"
            retrieval_top_k = 10

        elif choice == "2":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Extract tasks from meeting {meeting_num}"
            mode = "tasks"
            retrieval_top_k = 10

        elif choice == "3":
            query = input("Ask a question: ")
            query_lower = query.lower()

            if "summary" in query_lower or "summarize" in query_lower:
                mode = "summary"
                retrieval_top_k = 10
            elif "task" in query_lower:
                mode = "tasks"
                retrieval_top_k = 10
            else:
                mode = "qa"
                retrieval_top_k = 8  # slightly more chunks for cross-meeting search

            # Extract explicit meeting numbers from the query
            matches = re.findall(r'meeting\s*(\d+)|\b(\d+)\b', query_lower)
            meeting_ids = []
            for m1, m2 in matches:
                if m1:
                    meeting_ids.append(int(m1))
                elif m2:
                    meeting_ids.append(int(m2))

            if meeting_ids:
                # User asked about specific meeting(s)
                meeting_id = meeting_ids[0]
            else:
                # ── FIX: No meeting number mentioned ──
                # Check if it's a follow-up referencing a previous meeting
                follow_up_words = ['he', 'she', 'his', 'her', 'they', 'their',
                                   'it', 'this', 'that', 'هو', 'هي', 'ده', 'دي']
                is_follow_up = any(
                    word in query_lower.split() for word in follow_up_words
                )

                if is_follow_up and conversation_history:
                    # Re-use last known meeting for follow-ups
                    for turn in reversed(conversation_history):
                        if turn.get("meeting_id") is not None:
                            meeting_id = turn["meeting_id"]
                            meeting_ids = [meeting_id]
                            print(f"[INFO] Follow-up detected — using meeting {meeting_id}")
                            break

                if not meeting_ids:
                    # No specific meeting + not a follow-up → search everything
                    search_all = True
                    meeting_ids = None
                    print("[INFO] No meeting specified — searching across all meetings.")
        else:
            continue

        print("\nQuery Processing:", query)
        query_embedding = embed_model.encode(query).tolist()

        # ── Selection Logic ──
        if search_all:
            # Retrieve directly across all meetings without pre-filtering
            retrieved_chunks, retrieved_metas = retrieve_chunks_hierarchical(
                query, query_embedding, meeting_ids=None,
                top_k=retrieval_top_k, search_all=True
            )
        else:
            if meeting_ids:
                pass  # already set above
            elif meeting_id is not None:
                meeting_ids = [meeting_id]
            else:
                meeting_ids = get_relevant_meetings(query_embedding)
                if not meeting_ids:
                    meeting_ids = list(set([m["meeting_id"] for m in metadatas_all]))

            print("[DEBUG] meeting_ids:", meeting_ids)
            retrieved_chunks, retrieved_metas = retrieve_chunks_hierarchical(
                query, query_embedding, meeting_ids,
                top_k=retrieval_top_k, search_all=False
            )

        print("[DEBUG] retrieved_chunks:", len(retrieved_chunks))

        # Build context grouped by meeting
        grouped_context = {}
        for text, meta in zip(retrieved_chunks, retrieved_metas):
            m_id = meta["meeting_id"]
            if m_id not in grouped_context:
                grouped_context[m_id] = []
            grouped_context[m_id].append(text)

        context = ""
        for m_id, texts in grouped_context.items():
            context += f"\n### Meeting {m_id}:\n"
            context += "\n".join(texts)

        history_text = ""
        for turn in conversation_history[-3:]:
            history_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

        lang = detect_language(context)

        # ── Build prompt from file or inline ──
        if mode == "summary":
            prompt = summary_prompt_template.replace("{transcript}", context)
        elif mode == "tasks":
            prompt = task_prompt_template.replace("{transcript}", context)
        else:
            prompt = build_qa_prompt(query, context, history_text, lang)

        # ── API Call ──
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        print("✅ Answer generated")
        print(answer)

        # TTS disabled — to re-enable, uncomment generate_audio import above
        audio_path = generate_audio(answer)

        print("\nFinal Answer:\n", answer)

        conversation_history.append({
            "question": query,
            "answer": answer,
            "meeting_id": meeting_id
        })

        # Save results
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