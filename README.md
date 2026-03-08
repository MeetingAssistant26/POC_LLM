# AI Meeting Assistant – LLM POC

This project is a **Proof of Concept (POC)** for an AI Meeting Assistant that extracts **action items from meeting transcripts** using Large Language Models (LLMs).

The system takes a meeting transcript and automatically identifies:

* Tasks discussed in the meeting
* Responsible person for each task
* Deadlines (if mentioned)

The output is returned in **structured JSON format**.

---

# Project Goal

The goal of this POC is to evaluate how well different **LLM models** can extract structured information from meeting transcripts.

This is part of the **AI Meeting Assistant Graduation Project**.

---

# Example Input

Meeting Transcript:

Ahmed: We need to deploy the recommendation model to production by Friday.
Sara: I will review the dataset pipeline today.
Omar: I'll set up the Docker containers by Wednesday.

---

# Example Output

```json
[
  {
    "task": "deploy the recommendation model to production",
    "responsible_person": "Ahmed",
    "deadline": "Friday"
  },
  {
    "task": "review the dataset pipeline",
    "responsible_person": "Sara",
    "deadline": "today"
  },
  {
    "task": "set up the Docker containers",
    "responsible_person": "Omar",
    "deadline": "Wednesday"
  }
]
```

---

# Project Structure

```
POC_LLM/
│
├── data/
│   └── meeting_transcripts.json
│
├── prompts/
│   └── task_extraction_prompt.txt
│
├── llm_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

* Python
* Groq API
* Llama 3.1 LLM
* Prompt Engineering
* JSON processing

Future components of the full system include:

* WhisperX for speech-to-text
* Vector database (ChromaDB)
* Retrieval Augmented Generation (RAG)

---

# How to Run the Project

### 1️⃣ Clone the repository

```
git clone https://github.com/MeetingAssistant26/POC_LLM.git
```

---

### 2️⃣ Enter the project folder

```
cd POC_LLM
```

---

### 3️⃣ Create a virtual environment

```
python -m venv venv
```

---

### 4️⃣ Activate the environment

Windows:

```
venv\Scripts\activate
```

---

### 5️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 6️⃣ Create a `.env` file

Inside the project folder create a file called:

```
.env
```

and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

---

### 7️⃣ Run the script

```
python llm_test.py
```

---

# Future Improvements

* Support multiple LLM models for evaluation
* Process multiple meeting transcripts automatically
* Add automatic evaluation metrics
* Integrate speech-to-text using WhisperX
* Build a full AI meeting assistant pipeline

---

# Authors

AI Meeting Assistant Graduation Project Team
