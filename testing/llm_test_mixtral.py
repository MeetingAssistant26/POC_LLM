import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

url = "https://openrouter.ai/api/v1/chat/completions"

# load dataset
with open("data/meeting_transcripts.json", "r", encoding="utf-8") as f:
    meetings = json.load(f)

# load prompts
with open("prompts/meeting_summary_prompt.txt", "r", encoding="utf-8") as f:
    summary_prompt_template = f.read()

with open("prompts/task_extraction_prompt.txt", "r", encoding="utf-8") as f:
    task_prompt_template = f.read()

results = []

for meeting in meetings[:5]:

    transcript = meeting["transcript"]
    meeting_id = meeting["meeting_id"]

    print(f"Processing meeting {meeting_id}...")

    summary_prompt = summary_prompt_template.replace("{transcript}", transcript)

    summary_response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {"role": "user", "content": summary_prompt}
            ]
        }
    )

    summary_text = summary_response.json()["choices"][0]["message"]["content"]

    task_prompt = task_prompt_template.replace("{transcript}", transcript)

    task_response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [
                {"role": "user", "content": task_prompt}
            ]
        }
    )

    tasks_output = task_response.json()["choices"][0]["message"]["content"]

    try:
        tasks_json = json.loads(tasks_output)
    except:
        tasks_json = tasks_output

    results.append({
        "meeting_id": meeting_id,
        "summary": summary_text,
        "tasks": tasks_json
    })

with open("results/results_mixtral_groq.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Results saved to results/results_mixtral_groq.json")