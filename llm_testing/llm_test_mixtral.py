import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"

day_map = {
    "الاثنين": "Monday", "الثلاثاء": "Tuesday",
    "الأربعاء": "Wednesday", "الخميس": "Thursday",
    "الجمعة": "Friday", "السبت": "Saturday",
    "الأحد": "Sunday", "النهاردة": None,
    "قبل الاثنين": "Monday", "قبل الثلاثاء": "Tuesday",
    "قبل الأربعاء": "Wednesday", "قبل الخميس": "Thursday",
    "قبل الجمعة": "Friday", "قبل السبت": "Saturday",
    "قبل الأحد": "Sunday"
}

def clean_tasks(tasks_output):
    # Clean markdown code fences
    tasks_output = tasks_output.strip()
    tasks_output = re.sub(r"```json|```", "", tasks_output).strip()

    # Fix escaped underscores
    tasks_output = tasks_output.replace("\\_", "_")

    try:
        return json.loads(tasks_output)
    except:
        # Try to extract JSON array from partial/malformed response
        match = re.search(r"\[.*\]", tasks_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return tasks_output
        return tasks_output

def fix_deadlines(tasks_json):
    if isinstance(tasks_json, list):
        for task in tasks_json:
            if isinstance(task, dict):
                deadline = task.get("deadline")
                if deadline in day_map:
                    task["deadline"] = day_map[deadline]
                elif isinstance(deadline, str):
                    task["deadline"] = re.sub(r"(?i)before\s+", "", deadline).strip()
    return tasks_json

def call_api(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mistralai/mixtral-8x7b-instruct",
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                return None
    return None

# Load dataset
with open("data/meeting_transcripts_mixed.json", "r", encoding="utf-8") as f:
    meetings = json.load(f)

# Load prompts
with open("prompts/meeting_summary_prompt.txt", "r", encoding="utf-8") as f:
    summary_prompt_template = f.read()

with open("prompts/task_extraction_prompt.txt", "r", encoding="utf-8") as f:
    task_prompt_template = f.read()

results = []
failed_meetings = []

for meeting in meetings:
    transcript = meeting["transcript"]
    meeting_id = meeting["meeting_id"]

    print(f"Processing meeting {meeting_id}...")

    # --- Summary ---
    summary_prompt = summary_prompt_template.replace("{transcript}", transcript)
    summary_text = call_api(summary_prompt)

    if summary_text is None:
        print(f"   Summary failed for meeting {meeting_id}")
        failed_meetings.append(meeting_id)
        results.append({
            "meeting_id": meeting_id,
            "summary": "ERROR: Failed to generate summary",
            "tasks": []
        })
        continue

    # --- Tasks ---
    task_prompt = task_prompt_template.replace("{transcript}", transcript)
    tasks_output = call_api(task_prompt)

    if tasks_output is None:
        print(f"   Tasks failed for meeting {meeting_id}")
        failed_meetings.append(meeting_id)
        results.append({
            "meeting_id": meeting_id,
            "summary": summary_text,
            "tasks": []
        })
        continue

    # --- Parse and clean tasks ---
    tasks_json = clean_tasks(tasks_output)
    tasks_json = fix_deadlines(tasks_json)

    # --- Warn if tasks is not a list ---
    if not isinstance(tasks_json, list):
        print(f"    Meeting {meeting_id} tasks could not be parsed as JSON list")
        failed_meetings.append(meeting_id)

    results.append({
        "meeting_id": meeting_id,
        "summary": summary_text,
        "tasks": tasks_json
    })

    print(f"   Meeting {meeting_id} done")

# Save results
with open("results/results_mixtral_groq.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n Results saved to results/results_mixtral_groq.json")

if failed_meetings:
    print(f"  Meetings with issues: {failed_meetings}")
else:
    print(" All meetings processed successfully!")