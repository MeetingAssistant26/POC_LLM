import json
import re

with open("data/meeting_transcripts_mixed_rag.json", "r", encoding="utf-8") as f:
    meetings = json.load(f)

pattern = r'(\w+):\s*(.*?)(?=\s*\w+:|$)'

chunks = []
chunk_index = 1

for meeting in meetings:
    meeting_id = meeting["meeting_id"]
    transcript = meeting["transcript"]

    matches = re.findall(pattern, transcript, re.DOTALL)

    chunk_size = 3
    current_chunk = []
    chunk_id = 1

    for speaker, speech in matches:
        line = speaker + ": " + speech.strip()
        current_chunk.append(line)

        if len(current_chunk) == chunk_size:
            chunk_text = " ".join(current_chunk)
            word_count = len(chunk_text.split())

            chunks.append({
                "meeting_id": meeting_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "word_count": word_count,
                "text": chunk_text
            })

            chunk_index += 1
            chunk_id += 1
            current_chunk = []

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        word_count = len(chunk_text.split())

        chunks.append({
            "meeting_id": meeting_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "word_count": word_count,
            "text": chunk_text
        })

        chunk_index += 1

with open("RAG/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("Chunking completed")
print("Total chunks:", len(chunks))