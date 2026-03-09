AI Meeting Assistant

AI system that analyzes meeting transcripts and automatically generates:

Meeting summaries

Action items

Responsible persons

Deadlines

The project evaluates Large Language Models (LLMs) and prepares the system for a future RAG-based meeting assistant.

Project Components

data
Contains meeting transcript datasets used for testing the models.

prompts
Prompt templates used for meeting summarization and task extraction.

testing
Python scripts used to test different LLM models.

results
Generated outputs from the tested models.

RAG
Initial modules for the Retrieval-Augmented Generation system.

Features

Meeting summarization

Task extraction (task, responsible person, deadline)

Support for mixed Arabic and English transcripts

Testing multiple LLM models

Running the Project

Activate the virtual environment:

venv\Scripts\activate

Run the LLM testing script:

python testing/llm_test_llama.py

or

python testing/llm_test_mixtral.py

The generated outputs will be saved inside the results folder.

Next Steps

Implement RAG for long meetings

Improve context retrieval

Add automatic transcription
