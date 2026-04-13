import json
import os

def run_chunking(input_path="data/meeting_transcripts_audio.json", output_path="RAG/chunks.json"):
    CHUNK_SIZE = 250
    OVERLAP = 30

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        meetings = json.load(f)
    print(f"Loaded {len(meetings)} meetings")

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
                "text": chunk_text
            })
            index += 1
            start += (CHUNK_SIZE - OVERLAP)
        return chunks, index

    all_chunks = []
    chunk_index = 1
    for meeting in meetings:
        chunks, chunk_index = create_chunks(meeting["transcript"], meeting["meeting_id"], chunk_index)
        all_chunks.extend(chunks)

    os.makedirs("RAG", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Chunking Done: {len(all_chunks)} chunks saved.")
  