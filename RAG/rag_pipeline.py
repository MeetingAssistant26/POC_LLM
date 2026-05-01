import os
import json
import re
import time
import numpy as np
np.float_ = np.float64
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


# ── Load prompt templates from files ──────────────────────────
def load_prompt(filename):
    prompt_path = os.path.join("prompts", filename)
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Check if question is meeting-related or general knowledge ──
def is_meeting_related(query, groq_client):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Is this question about a specific meeting, person, task, 
            or discussion that happened in a meeting?
            The question could be in Arabic or English.
            Answer ONLY with 'yes' or 'no'.
            Question: {query}"""
                    }],
        temperature=0
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer


def generate_audio(text):
    print("🔇 TTS is currently disabled.")
    return "disabled"


# ── Tasks Store Helpers ────────────────────────────────────────
TASKS_STORE_PATH = "RAG/tasks_store.json"

def load_tasks_store():
    if not os.path.exists(TASKS_STORE_PATH):
        return {}
    with open(TASKS_STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_tasks_store(store):
    os.makedirs(os.path.dirname(TASKS_STORE_PATH), exist_ok=True)
    with open(TASKS_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

def save_tasks_for_meeting(meeting_id, tasks):
    """حفظ الـ tasks في tasks_store.json مع merge للـ meetings الموجودة"""
    store = load_tasks_store()
    key = str(meeting_id)

    if key not in store:
        store[key] = []

    existing_tasks = {t["task"]: t for t in store[key]}
    for task in tasks:
        task_name = task.get("task", "")
        if task_name not in existing_tasks:
            task.setdefault("status", "pending")
            existing_tasks[task_name] = task

    store[key] = list(existing_tasks.values())
    save_tasks_store(store)
    print(f"💾 Tasks saved to tasks_store.json for meeting {meeting_id}")


def extract_tasks_from_answer(answer):
    """Parse الـ JSON من الـ LLM answer"""
    if isinstance(answer, str):
        try:
            clean = re.sub(r"```json|```", "", answer).strip()
            parsed = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return []
    elif isinstance(answer, dict):
        parsed = answer
    else:
        return []
    return parsed.get("tasks", [])


# ── Reminders Helpers ──────────────────────────────────────────
REMINDERS_PATH = "RAG/reminders.json"

def load_reminders():
    if not os.path.exists(REMINDERS_PATH):
        return []
    with open(REMINDERS_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_reminders(reminders):
    os.makedirs(os.path.dirname(REMINDERS_PATH), exist_ok=True)
    with open(REMINDERS_PATH, "w", encoding="utf-8") as f:
        json.dump(reminders, f, ensure_ascii=False, indent=2)


def run_pipeline():
    print("Inside run_pipeline function")

    load_dotenv()

    # ── Load prompts once at startup ──
    summary_prompt_template = load_prompt("meeting_summary_prompt.txt")
    task_prompt_template = load_prompt("task_extraction_prompt.txt")
    qa_prompt_template = load_prompt("QuestionAndAnswer_prompt.txt")
    personalized_summary_template = load_prompt("PersonalizedSummary_prompt.txt")

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
    def get_relevant_meetings(query_embedding, top_k=5):
        total_chunks = collection.count()
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
    def retrieve_chunks_hierarchical(query, query_embedding, meeting_ids,
                                     top_k=8, search_all=False):
        all_texts = []
        all_metas = []

        if search_all:
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

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        query_vec = vectorizer.transform([query])
        cos_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

        top_indices_tfidf = np.argsort(-cos_scores)[:20]
        tfidf_texts = [all_texts[i] for i in top_indices_tfidf]
        tfidf_metas = [all_metas[i] for i in top_indices_tfidf]

        unique = {}
        for text, meta in zip(all_texts + tfidf_texts, all_metas + tfidf_metas):
            if text not in unique:
                unique[text] = meta

        final_texts = list(unique.keys())
        final_metas = list(unique.values())

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

    # ── Build General Knowledge prompt ──
    def build_general_prompt(query, lang):
        if lang == "ar_colloquial":
            lang_instruction = "رد بالعامية المصرية الطبيعية."
        elif lang == "ar_formal":
            lang_instruction = "رد بالعربي الفصحى."
        else:
            lang_instruction = "Respond in English."

        return f"""
You are a helpful AI assistant.
{lang_instruction}

Answer the following general question using your knowledge.
Be concise and clear.

Question: {query}
"""

    # ── Shared: run task extraction for a given meeting_id ──
    def run_task_extraction(meeting_id, retrieval_top_k=10):
        query = f"Extract tasks from meeting {meeting_id}"
        query_embedding = embed_model.encode(query).tolist()

        retrieved_chunks, retrieved_metas = retrieve_chunks_hierarchical(
            query, query_embedding, [meeting_id],
            top_k=retrieval_top_k, search_all=False
        )

        grouped_context = {}
        for text, meta in zip(retrieved_chunks, retrieved_metas):
            m_id = meta["meeting_id"]
            grouped_context.setdefault(m_id, []).append(text)

        context = ""
        for m_id, texts in grouped_context.items():
            context += f"\n### Meeting {m_id}:\n"
            context += "\n".join(texts)

        prompt = task_prompt_template.replace("{transcript}", context)

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        tasks = extract_tasks_from_answer(answer)
        save_tasks_for_meeting(meeting_id, tasks)

        return answer, tasks, retrieved_chunks

    # ── MAIN LOOP ──
    print("✅ AI Meeting Assistant is now active.")

    # ── Show pending reminders at startup ──
    reminders = load_reminders()
    pending_reminders = [r for r in reminders if not r.get("done", False)]
    if pending_reminders:
        print(f"\n⏰ You have {len(pending_reminders)} pending reminder(s):")
        for i, r in enumerate(pending_reminders, 1):
            print(f"  {i}. {r['text']}  [{r['created_at']}]")

    while True:
        print("\nChoose an option:")
        print("1 - Generate Meeting Summary")
        print("2 - Extract Tasks")
        print("3 - Ask a Question")
        print("4 - View Pending Tasks")
        print("5 - Mark Task as Done")
        print("6 - View / Mark Reminders Done")
        print("7 - Exit")

        choice = input("\nEnter choice (1/2/3/4/5/6/7): ")
        if choice == "7":
            print("Goodbye!")
            break

        # ── Option 4: View Pending Tasks ──
        if choice == "4":
            meeting_num = input("Enter meeting number (1-10): ").strip()
            meeting_id = int(meeting_num)

            store = load_tasks_store()
            key = str(meeting_id)

            if key in store and store[key]:
                print(f"\n📋 Loading tasks from store for meeting {meeting_id}...")
                tasks = store[key]
            else:
                print(f"\n🔍 Extracting tasks from meeting {meeting_id}...")
                _, tasks, _ = run_task_extraction(meeting_id)

            pending = [t for t in tasks if t.get("status") == "pending"]

            if not pending:
                print(f"\n✅ No pending tasks in meeting {meeting_id}.")
            else:
                print(f"\n📋 Pending Tasks — Meeting {meeting_id} ({len(pending)} task(s)):\n")
                for i, t in enumerate(pending, 1):
                    due = f" — Due: {t['due_date']}" if t.get("due_date") else ""
                    print(f"  {i}. {t['assignee']}: {t['task']}{due}")
            continue

        # ── Option 5: Mark Task as Done ──
        if choice == "5":
            store = load_tasks_store()
            if not store:
                print("⚠️ No tasks found yet. Use Option 2 first to extract tasks from a meeting.")
                continue

            all_pending = []
            for m_id, tasks in store.items():
                for t in tasks:
                    if t.get("status") == "pending":
                        all_pending.append((m_id, t))

            if not all_pending:
                print("\n✅ No pending tasks found.")
                continue

            print(f"\n📋 All Pending Tasks ({len(all_pending)} task(s)):\n")
            for i, (m_id, t) in enumerate(all_pending, 1):
                due = f" — Due: {t['due_date']}" if t.get("due_date") else ""
                print(f"  {i}. [Meeting {m_id}] {t['assignee']}: {t['task']}{due}")

            task_name = input("\nEnter task name to mark as done: ").strip()
            if not task_name:
                print("⚠️ No task name entered.")
                continue

            matches = []
            for m_id, tasks in store.items():
                for t in tasks:
                    if task_name.lower() in t.get("task", "").lower():
                        if t.get("status") == "pending":
                            matches.append((m_id, t))

            if not matches:
                print(f"⚠️ No pending task found matching: \"{task_name}\"")
                continue

            if len(matches) > 1:
                meeting_ids_found = [m_id for m_id, _ in matches]
                print(f"\n⚠️ Found in multiple meetings: {', '.join(['Meeting ' + m for m in meeting_ids_found])}")
                chosen = input("Which meeting? Enter meeting number: ").strip()
                matches = [(m_id, t) for m_id, t in matches if m_id == chosen]
                if not matches:
                    print("⚠️ Invalid meeting number.")
                    continue

            for m_id, t in matches:
                t["status"] = "done"
                print(f"✅ Marked as done: \"{t['task']}\"")

            save_tasks_store(store)
            continue

        # ── Option 6: View / Mark Reminders Done ──
        if choice == "6":
            reminders = load_reminders()
            active = [r for r in reminders if not r.get("done", False)]

            if not active:
                print("\n✅ No pending reminders.")
            else:
                print(f"\n⏰ Pending Reminders ({len(active)}):\n")
                for i, r in enumerate(active, 1):
                    print(f"  {i}. {r['text']}  [{r['created_at']}]")

                mark = input("\nMark as done? Enter number (or 0 to skip): ").strip()
                if mark.isdigit() and 0 < int(mark) <= len(active):
                    idx = reminders.index(active[int(mark) - 1])
                    reminders[idx]["done"] = True
                    save_reminders(reminders)
                    print("✅ Reminder marked as done!")
            continue

        # ────────────────────────────────────────────────────────
        # Options 1 / 2 / 3
        # ────────────────────────────────────────────────────────
        meeting_id = None
        meeting_ids = None
        search_all = False

        if choice == "1":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Summarize meeting {meeting_num}"
            mode = "summary"
            retrieval_top_k = 10

            personalized = input("Generate personalized summaries per speaker? (y/n): ").strip().lower()
            if personalized == "y":
                mode = "personalized_summary"

        elif choice == "2":
            meeting_num = input("Enter meeting number (1-10): ")
            meeting_id = int(meeting_num)
            query = f"Extract tasks from meeting {meeting_num}"
            mode = "tasks"
            retrieval_top_k = 10

        elif choice == "3":
            query = input("Ask a question: ")
            query_lower = query.lower()

            # ── Reminder detection ──
            if "remind" in query_lower or "ذكرني" in query or "reminder" in query_lower:
                reminder = {
                    "text": query,
                    "created_at": time.strftime("%Y-%m-%d %H:%M"),
                    "done": False
                }
                reminders = load_reminders()
                reminders.append(reminder)
                save_reminders(reminders)
                print("\n✅ تم حفظ الـ reminder بتاعك!")
                continue

            if "summary" in query_lower or "summarize" in query_lower:
                mode = "summary"
                retrieval_top_k = 10
            elif "task" in query_lower:
                mode = "tasks"
                retrieval_top_k = 10
            else:
                if is_meeting_related(query, groq_client):
                    mode = "qa"
                    print("[INFO] Meeting-related question detected.")
                else:
                    mode = "general"
                    print("[INFO] General question detected — answering from knowledge.")
                retrieval_top_k = 8

            matches = re.findall(r'meeting\s*(\d+)|\b(\d+)\b', query_lower)
            meeting_ids = []
            for m1, m2 in matches:
                if m1:
                    meeting_ids.append(int(m1))
                elif m2:
                    meeting_ids.append(int(m2))

            if meeting_ids:
                meeting_id = meeting_ids[0]
            else:
                follow_up_words = ['he', 'she', 'his', 'her', 'they', 'their',
                                   'it', 'this', 'that', 'هو', 'هي', 'ده', 'دي']
                is_follow_up = any(
                    word in query_lower.split() for word in follow_up_words
                )

                if is_follow_up and conversation_history:
                    for turn in reversed(conversation_history):
                        if turn.get("meeting_id") is not None:
                            meeting_id = turn["meeting_id"]
                            meeting_ids = [meeting_id]
                            print(f"[INFO] Follow-up detected — using meeting {meeting_id}")
                            break

                if not meeting_ids and mode != "general":
                    search_all = True
                    meeting_ids = None
                    print("[INFO] No meeting specified — searching across all meetings.")
        else:
            continue

        print("\nQuery Processing:", query)

        # ── General mode ──
        if mode == "general":
            lang = detect_language(query)
            prompt = build_general_prompt(query, lang)
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content
            print("✅ Answer generated")
            print(answer)
            audio_path = generate_audio(answer)
            print("\nFinal Answer:\n", answer)

            conversation_history.append({
                "question": query,
                "answer": answer,
                "meeting_id": None
            })

            results_dir = "RAG/rag_results"
            os.makedirs(results_dir, exist_ok=True)
            file_count = len([f for f in os.listdir(results_dir) if f.endswith('.json')]) + 1
            file_path = f"{results_dir}/result_{file_count}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "query_number": file_count,
                    "meeting_id": None,
                    "question": query,
                    "answer": answer,
                    "audio_path": audio_path,
                    "context_used": []
                }, f, ensure_ascii=False, indent=2)
            print(f"📂 Result saved to: {file_path}")
            continue

        # ── RAG mode ──
        query_embedding = embed_model.encode(query).tolist()

        if search_all:
            retrieved_chunks, retrieved_metas = retrieve_chunks_hierarchical(
                query, query_embedding, meeting_ids=None,
                top_k=retrieval_top_k, search_all=True
            )
        else:
            if meeting_ids:
                pass
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

        if mode == "summary":
            prompt = summary_prompt_template.replace("{transcript}", context)

        elif mode == "personalized_summary":
            speakers = sorted(set(re.findall(r'SPEAKER_\d+', context)))
            print(f"\nFound speakers: {speakers}")

            all_personalized = {}
            for speaker in speakers:
                print(f"\n--- Generating summary for {speaker} ---")
                prompt = personalized_summary_template\
                    .replace("{speaker}", speaker)\
                    .replace("{transcript}", context)

                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                speaker_answer = response.choices[0].message.content
                print(speaker_answer)
                all_personalized[speaker] = speaker_answer
                audio_path = generate_audio(speaker_answer)

            results_dir = "RAG/rag_results"
            os.makedirs(results_dir, exist_ok=True)
            file_count = len([f for f in os.listdir(results_dir) if f.endswith('.json')]) + 1
            file_path = f"{results_dir}/result_{file_count}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "query_number": file_count,
                    "meeting_id": meeting_id,
                    "question": query,
                    "answer": all_personalized,
                    "audio_path": "done",
                    "context_used": retrieved_chunks
                }, f, ensure_ascii=False, indent=2)
            print(f"📂 Result saved to: {file_path}")
            continue

        elif mode == "tasks":
            prompt = task_prompt_template.replace("{transcript}", context)
        else:
            prompt = qa_prompt_template\
                .replace("{context}", context)\
                .replace("{history_text}", history_text)\
                .replace("{query}", query)

        # ── API Call ──
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content
        print("✅ Answer generated")
        print(answer)

        audio_path = generate_audio(answer)
        print("\nFinal Answer:\n", answer)

        # ── لو Option 2: احفظ الـ tasks في tasks_store.json ──
        if mode == "tasks":
            tasks = extract_tasks_from_answer(answer)
            if tasks:
                save_tasks_for_meeting(meeting_id, tasks)

        conversation_history.append({
            "question": query,
            "answer": answer,
            "meeting_id": meeting_id
        })

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
