import json
import os
import re

def run_chunking(input_path="data/meeting_transcripts_audio.json", output_path="RAG/chunks.json"):
    # chunk بـ speaker turns مش بـ كلمات عشوائية
    CHUNK_SIZE = 100   # كلمة للـ fallback
    OVERLAP = 20

    if not os.path.exists(input_path):
        print(f"❌ Error: Input file {input_path} not found.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        meetings = json.load(f)
    print(f"Loaded {len(meetings)} meetings")

    def split_by_speaker_turns(transcript):
        """
        بيقسم الـ transcript على speaker turns
        يتعامل مع:
          - SPEAKER_00: ...
          - [0.07-14.26] SPEAKER_00: ...
        """
        # pattern يشمل timestamps اختيارية
        pattern = r'(?:\[\d+\.\d+-\d+\.\d+\]\s*)?SPEAKER_\d+\s*:'
        parts = re.split(f'({pattern})', transcript)

        turns = []
        i = 1
        while i < len(parts):
            speaker_label = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                turns.append(f"{speaker_label} {content}")
            i += 2

        return turns

    def merge_turns_into_chunks(turns, meeting_id, start_index, max_words=120, overlap_turns=1):
        """
        بيدمج الـ speaker turns في chunks بحجم معقول
        مع overlap بين الـ chunks عشان السياق ميتقطعش
        """
        chunks = []
        index = start_index
        i = 0

        while i < len(turns):
            chunk_turns = []
            word_count = 0

            j = i
            while j < len(turns):
                turn_words = len(turns[j].split())
                if word_count + turn_words > max_words and chunk_turns:
                    break
                chunk_turns.append(turns[j])
                word_count += turn_words
                j += 1

            chunk_text = " | ".join(chunk_turns)
            chunks.append({
                "meeting_id": meeting_id,
                "chunk_id": index,
                "text": chunk_text
            })
            index += 1

            # overlap: نرجع overlap_turns عشان السياق ميتقطعش
            i = j - overlap_turns if j - overlap_turns > i else j

        return chunks, index

    all_chunks = []
    chunk_index = 1

    for meeting in meetings:
        transcript = meeting["transcript"]
        meeting_id = meeting["meeting_id"]

        # حاول تقسّم على speaker turns الأول
        turns = split_by_speaker_turns(transcript)

        if len(turns) >= 2:
            # عندنا speaker turns واضحة
            chunks, chunk_index = merge_turns_into_chunks(
                turns, meeting_id, chunk_index, max_words=120, overlap_turns=1
            )
        else:
            # fallback: قسّم على كلمات لو مفيش speaker labels
            words = transcript.split()
            i = 0
            while i < len(words):
                end = i + CHUNK_SIZE
                chunk_text = " ".join(words[i:end])
                all_chunks.append({
                    "meeting_id": meeting_id,
                    "chunk_id": chunk_index,
                    "text": chunk_text
                })
                chunk_index += 1
                i += (CHUNK_SIZE - OVERLAP)
            chunks = []

        all_chunks.extend(chunks)
        print(f"  Meeting {meeting_id}: {len(turns)} speaker turns → {len(chunks) if len(turns) >= 2 else '?'} chunks")

    os.makedirs("RAG", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunking Done: {len(all_chunks)} chunks saved.")