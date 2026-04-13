import os
import json
from dotenv import load_dotenv
from groq import Groq

# load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# initialize groq client
client = Groq(api_key=api_key)

# -----------------------------
# LOAD DATASET (MIXED LANGUAGE)
# -----------------------------
with open("data/meeting_transcripts_mixed.json", "r", encoding="utf-8") as f:
    meetings = json.load(f)

# -----------------------------
# LOAD PROMPT
# -----------------------------
with open("prompts/mixed_language_prompt.txt", "r", encoding="utf-8") as f:
    prompt_template = f.read()

results = []

# -----------------------------
# LOOP OVER MEETINGS
# -----------------------------
for meeting in meetings[:5]:  

    transcript = meeting["transcript"]
    meeting_id = meeting["meeting_id"]

    print(f"Processing meeting {meeting_id}...")

    final_prompt = prompt_template.replace("{transcript}", transcript)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": final_prompt}]
    )

    output = response.choices[0].message.content

    # convert output to JSON
    try:
        parsed_output = json.loads(output)
    except:
        parsed_output = output

    results.append({
        "meeting_id": meeting_id,
        "llm_output": parsed_output
    })

# -----------------------------
# SAVE RESULTS
# -----------------------------
with open("results/results_mixed_groq.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Results saved to results/results_mixed_groq.json")