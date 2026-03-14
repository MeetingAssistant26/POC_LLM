# 🤖 AI Meeting Assistant

AI system that analyzes meeting transcripts and automatically generates:

- **Meeting summaries**
- **Action items**
- **Responsible persons**
- **Deadlines**
- **Answers to questions about the meeting**

The project evaluates **Large Language Models (LLMs)** and implements a **RAG-based meeting assistant** that retrieves relevant meeting context before generating responses.

---

# 📌 Project Components

### 📂 data
Contains meeting transcript datasets used for testing the models and the RAG system.

### 📂 prompts
Prompt templates used for:

- Meeting summarization  
- Task extraction  
- Mixed-language transcript understanding  

### 📂 llm_testing
Python scripts used to test different **LLM models**.

Example:

```python
llm_test_llama.py
llm_test_mixtral.py
```

### 📂 llm_results
Contains outputs generated from the tested LLM models.

### 📂 RAG
Core modules implementing the **Retrieval-Augmented Generation pipeline**.

Modules include:

- **chunking.py** → splits meeting transcripts into smaller chunks  
- **embeddings.py** → generates vector embeddings for chunks  
- **vector_store.py** → stores embeddings in ChromaDB  
- **rag_pipeline.py** → retrieves relevant context and generates answers using the LLM  

---

# ⚙️ System Workflow

The system processes meetings using a **RAG pipeline**:

```
Meeting Transcript
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database (ChromaDB)
        ↓
Similarity Search
        ↓
Retrieve Relevant Chunks
        ↓
LLM (Groq)
        ↓
Generated Answer
```

This allows the AI system to generate responses based on **actual meeting content** instead of relying only on the model’s training data.

---

# 🚀 Features

- **Meeting summarization**
- **Task extraction** *(task, responsible person, deadline)*
- **Question answering about meetings**
- Support for **mixed Arabic and English transcripts**
- **RAG-based semantic search**
- Logging answers for evaluation

---

# 📁 Project Structure

```text
AI_MEETING_ASSISTANT
│
├── data
│   ├── meeting_transcripts.json
│   ├── meeting_transcripts_mixed.json
│   └── meeting_transcripts_mixed_rag.json
│
├── llm_results
│   ├── results_groq_llama.json
│   └── results_mixed_groq_llama.json
│
├── llm_testing
│   ├── llm_test_llama.py
│   └── llm_test_mixtral.py
│
├── prompts
│   ├── meeting_summary_prompt.txt
│   ├── task_extraction_prompt.txt
│   └── mixed_language_prompt.txt
│
├── RAG
│   ├── chroma_db
│   ├── rag_results
│   ├── chunking.py
│   ├── chunks.json
│   ├── embeddings.py
│   ├── embeddings.json
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   └── search_results.json
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# ▶️ Running the Project

### 1️⃣ Activate the virtual environment

```bash
venv\Scripts\activate
```

---

### 2️⃣ Install dependencies

```bash
pip install chromadb sentence-transformers groq python-dotenv
```

---

### 3️⃣ Add your API key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 4️⃣ Run the RAG assistant

```bash
python RAG/rag_pipeline.py
```

You will be able to choose between:

```
1 - Generate Meeting Summary
2 - Extract Tasks
3 - Ask a Question
```

---

# 📊 Current Status

The project currently implements a working **MVP of an AI Meeting Assistant**, including:

- LLM testing
- Prompt engineering
- Meeting summarization
- Task extraction
- RAG pipeline
- Semantic retrieval
- Question answering over meeting transcripts

---

# 🔮 Future Improvements

Possible next steps:

- Conversation-aware RAG
- Hierarchical RAG
- Audio-to-transcript pipeline
- Web interface (Streamlit)

---
