import json
import os

# -------- SETTINGS --------
INPUT_PATH = "data/meeting_transcripts_audio.json"
OUTPUT_PATH = "RAG/chunks.json"

CHUNK_SIZE = 250      # number of words per chunk
OVERLAP = 30          # overlap between chunks


# -------- LOAD DATA --------
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    meetings = json.load(f)

print(f"Loaded {len(meetings)} meetings")


# -------- CHUNKING FUNCTION --------
def create_chunks(text, meeting_id, start_index):
    words = text.split()
    chunks = []
    index = start_index

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append({
            "meeting_id": meeting_id,
            "chunk_id": index,
            "chunk_index": index,
            "word_count": len(chunk_words),
            "text": chunk_text
        })

        index += 1
        start += (CHUNK_SIZE - OVERLAP)

    return chunks, index


# -------- MAIN PROCESS --------
all_chunks = []
chunk_index = 1

for meeting in meetings:
    meeting_id = meeting["meeting_id"]
    transcript = meeting["transcript"]

    chunks, chunk_index = create_chunks(transcript, meeting_id, chunk_index)
    all_chunks.extend(chunks)


# -------- SAVE OUTPUT --------
os.makedirs("RAG", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Generated {len(all_chunks)} chunks")
print(f"Saved to {OUTPUT_PATH}")