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

# load prompt
with open("prompts/task_extraction_prompt.txt", "r") as f:
    prompt_template = f.read()

# take first meeting transcript
transcript = meetings[0]["transcript"]

# insert transcript into prompt
final_prompt = prompt_template.replace("{transcript}", transcript)

# send request to LLM
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": final_prompt}
    ]
)

# print result
print(response.choices[0].message.content)