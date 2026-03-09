AI Meeting Assistant

This project is a Proof of Concept (POC) for an AI-powered Meeting Assistant that can automatically analyze meeting transcripts and extract useful information such as summaries and action items.

The system uses Large Language Models (LLMs) to process meeting transcripts and generate structured outputs.

Project Features

The system currently supports:

1️⃣ Meeting Summarization

Generate concise summaries of meetings based on the transcript.

2️⃣ Task Extraction

Automatically extract actionable tasks including:

Task description

Responsible person

Deadline (if mentioned)

3️⃣ Mixed Language Handling

The system supports transcripts containing mixed Egyptian Arabic and English.

Example:

Sara: أنا هراجع الـETL scripts وأصلح المشاكل قبل الخميس

The model can still correctly extract tasks and deadlines.

4️⃣ LLM Model Evaluation

Different LLM models are tested and their outputs are stored for comparison.

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
│
├── requirements.txt
└── README.md
How the LLM Pipeline Works

The current pipeline works as follows:

Load meeting transcript dataset

Send transcript to the LLM

Generate:

Meeting summary

Actionable tasks

Convert results to structured JSON

Store outputs in the results folder

Example output:

{
  "summary": "The team reviewed user feedback and discussed preparing a report and presentation for management.",
  "tasks": [
    {
      "task": "Review user feedback",
      "responsible_person": "Sara",
      "deadline": "Thursday"
    },
    {
      "task": "Test notifications flow",
      "responsible_person": "Mariam",
      "deadline": "Wednesday"
    }
  ]
}
Running the LLM Tests

Activate the virtual environment:

venv\Scripts\activate

Run the LLM test script:

python testing/llm_test_llama.py

or

python testing/llm_test_mixtral.py

The outputs will be saved inside:

results/
RAG Module (In Progress)

The project includes an early implementation of a Retrieval-Augmented Generation (RAG) system.

Modules inside RAG/ include:

chunking.py
Splits long meeting transcripts into smaller chunks.

embeddings.py
Generates vector embeddings for transcript chunks.

vector_store.py
Stores embeddings inside a vector database.

retrieval.py
Retrieves relevant context for LLM queries.

This system will help the assistant handle long meetings and contextual questions.

Future Work

Next improvements planned:

Whisper integration for automatic transcription

Hierarchical RAG for long meetings

Conversation-aware retrieval

Integration with task management tools (Trello / Jira)

LLM model comparison (Groq vs Qwen vs DeepSeek)

Tech Stack

Python

Groq API

Llama / Mixtral

Vector embeddings

JSON structured outputs
