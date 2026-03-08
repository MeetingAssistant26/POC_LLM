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

# load prompts
with open("prompts/meeting_summary_prompt.txt", "r") as f:
    summary_prompt_template = f.read()

with open("prompts/task_extraction_prompt.txt", "r") as f:
    task_prompt_template = f.read()

results = []

# loop over all meetings
for meeting in meetings:

    transcript = meeting["transcript"]
    meeting_id = meeting["meeting_id"]

    # ----- SUMMARY -----
    summary_prompt = summary_prompt_template.replace("{transcript}", transcript)

    summary_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": summary_prompt}]
    )

    summary_text = summary_response.choices[0].message.content

    # ----- TASK EXTRACTION -----
    task_prompt = task_prompt_template.replace("{transcript}", transcript)

    task_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": task_prompt}]
    )

    tasks_output = task_response.choices[0].message.content

    try:
        tasks_json = json.loads(tasks_output)
    except:
        tasks_json = tasks_output

    results.append({
        "meeting_id": meeting_id,
        "summary": summary_text,
        "tasks": tasks_json
    })

# save results to groq file
with open("results/results_groq.json", "w") as f:
    json.dump(results, f, indent=2)

print("Results saved to results/results_groq.json")