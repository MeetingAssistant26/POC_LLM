# AI Meeting Assistant – LLM POC

This project is a Proof of Concept (POC) for an AI-powered Meeting Assistant.
The system processes meeting transcripts and uses a Large Language Model (LLM) to generate structured insights.

## Features

The system currently performs:

### 1. Meeting Summary

Generates a concise summary of the meeting discussion.

### 2. Task Extraction

Extracts actionable tasks from the meeting transcript including:

* Task description
* Responsible person
* Deadline (if mentioned)

### Example Output

Summary:
The meeting focused on deploying the recommendation model to production by Friday. Sara will review the dataset pipeline and verify the API endpoints. Omar will set up the Docker containers by Wednesday.

Action Items:

[
{
"task": "deploy the recommendation model to production",
"responsible_person": null,
"deadline": "Friday"
},
{
"task": "review the dataset pipeline",
"responsible_person": "Sara",
"deadline": "today"
},
{
"task": "check the API endpoints",
"responsible_person": "Sara",
"deadline": null
},
{
"task": "set up the Docker containers",
"responsible_person": "Omar",
"deadline": "Wednesday"
}
]

## Project Structure

ai_meeting_assistant/

prompts/

* task_extraction_prompt.txt
* meeting_summary_prompt.txt

data/

* meeting_transcripts.json

llm_test.py

requirements.txt

README.md

## Installation

Install dependencies:

pip install -r requirements.txt

## Environment Variables

Create a `.env` file and add your API key:

GROQ_API_KEY=your_api_key_here

## Run the Project

python llm_test.py

The system will:

1. Load a meeting transcript
2. Generate a meeting summary
3. Extract action items from the transcript

## Future Work

* Process audio meetings using WhisperX (Speech-to-Text)
* Analyze multiple meetings automatically
* Save results to JSON
* Integrate task creation with Trello
