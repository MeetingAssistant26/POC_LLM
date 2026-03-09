AI Meeting Assistant

A Proof of Concept (POC) for an AI system that analyzes meeting transcripts and automatically generates:

Meeting summaries

Actionable tasks

Responsible persons

Deadlines

The project evaluates different Large Language Models (LLMs) and prepares the foundation for a RAG-based meeting assistant.

Project Structure
AI_MEETING_ASSISTANT
│
├── data
│   ├── meeting_transcripts.json
│   └── meeting_transcripts_mixed.json
│
├── prompts
│   ├── meeting_summary_prompt.txt
│   ├── task_extraction_prompt.txt
│   └── mixed_language_prompt.txt
│
├── testing
│   ├── llm_test_llama.py
│   └── llm_test_mixtral.py
│
├── results
│   ├── results_groq.json
│   ├── results_mixed_groq.json
│   └── results_mixtral_groq.json
│
├── RAG
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── retrieval.py
Features

Meeting Summarization

Task Extraction
(task, responsible person, deadline)

Mixed Language Support
Handles transcripts containing Arabic + English

LLM Model Testing

Running the Project
1️⃣ Activate the virtual environment
venv\Scripts\activate
2️⃣ Run LLM testing
python testing/llm_test_llama.py

or

python testing/llm_test_mixtral.py
3️⃣ Check results

Generated outputs will appear in:

results/
Next Steps

Implement RAG for long meetings

Improve context retrieval

Integrate automatic transcription
