import os
import json
from dotenv import load_dotenv
from groq import Groq

# load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# initialize groq client
client = Groq(api_key=api_key)

# load dataset
with open("data/meeting_transcripts.json", "r") as f:
    meetings = json.load(f)

# take first meeting transcript
transcript = meetings[0]["transcript"]

# -----------------------------
# SUMMARY GENERATION
# -----------------------------

# load summary prompt
with open("prompts/meeting_summary_prompt.txt", "r") as f:
    summary_prompt = f.read()

summary_input = summary_prompt.replace("{transcript}", transcript)

summary_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": summary_input}
    ]
)

summary_text = summary_response.choices[0].message.content

print("\n=== MEETING SUMMARY ===\n")
print(summary_text)


# -----------------------------
# TASK EXTRACTION
# -----------------------------

# load task extraction prompt
with open("prompts/task_extraction_prompt.txt", "r") as f:
    task_prompt = f.read()

task_input = task_prompt.replace("{transcript}", transcript)

task_response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": task_input}
    ]
)

tasks_output = task_response.choices[0].message.content

print("\n=== ACTION ITEMS ===\n")
print(tasks_output)