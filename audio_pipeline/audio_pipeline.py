import os
import re
import json
import torch
import whisperx
from pydub import AudioSegment
from pydub.silence import split_on_silence
from whisperx.diarize import DiarizationPipeline

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
AUDIO_INPUT_FOLDER  = "audio"
AUDIO_CLEAN_FOLDER  = "audio_clean"
TRANSCRIPTS_FOLDER  = "transcripts"
OUTPUT_JSON_PATH    = "data/meeting_transcripts_audio.json"
HF_TOKEN            = "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"

os.makedirs(AUDIO_CLEAN_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
os.makedirs("data", exist_ok=True)


# ══════════════════════════════════════════════════════════════
# STEP 1 — Audio Preprocessing
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — Audio Preprocessing")
print("="*60)

for file in os.listdir(AUDIO_INPUT_FOLDER):
    if file.endswith((".wav", ".mp3", ".m4a")):

        input_path  = os.path.join(AUDIO_INPUT_FOLDER, file)
        output_path = os.path.join(AUDIO_CLEAN_FOLDER, file.split(".")[0] + ".wav")

        # Skip if already preprocessed
        if os.path.exists(output_path):
            print(f"[skip] {file} — already preprocessed")
            continue

        print(f"Processing {file}")

        audio = AudioSegment.from_file(input_path)

        # Convert to mono
        audio = audio.set_channels(1)

        # Convert sample rate
        audio = audio.set_frame_rate(16000)

        # Normalize volume
        change_in_dBFS = -20.0 - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)

        # Remove long silence
        chunks = split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=audio.dBFS - 14,
            keep_silence=300
        )

        processed_audio = AudioSegment.empty()
        for chunk in chunks:
            processed_audio += chunk

        processed_audio.export(output_path, format="wav")
        print(f"✓ Saved → {output_path}")

print("Audio preprocessing completed.")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Transcription + Diarization
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — Transcription & Diarization (WhisperX)")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = whisperx.load_model("medium", device, compute_type="int8")

for filename in sorted(os.listdir(AUDIO_CLEAN_FOLDER)):
    if filename.endswith(".wav"):

        audio_path   = os.path.join(AUDIO_CLEAN_FOLDER, filename)
        meeting_name = filename.replace(".wav", "")
        txt_path     = os.path.join(TRANSCRIPTS_FOLDER, f"{meeting_name}.txt")

        # Skip if already transcribed
        if os.path.exists(txt_path):
            print(f"[skip] {filename} — transcript already exists")
            continue

        print("\nProcessing:", filename)

        # Load audio
        audio = whisperx.load_audio(audio_path)

        # Transcription
        result = model.transcribe(audio)

        # Alignment
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(result["segments"], align_model, metadata, audio, device)

        # Speaker diarization
        diarize_model    = DiarizationPipeline(token=HF_TOKEN, device=device)
        diarize_segments = diarize_model(audio_path)

        # Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Save transcript — keep generic SPEAKER_XX labels
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                speaker = seg.get("speaker", "UNKNOWN")
                start   = seg["start"]
                end     = seg["end"]
                text    = seg["text"]
                f.write(f"[{start:.2f}-{end:.2f}] {speaker}: {text}\n")

        print("Saved:", txt_path)

print("\nAll transcriptions completed!")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Build JSON dataset  (keeps SPEAKER_XX, no name mapping)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — Building JSON dataset")
print("="*60)

LINE_PATTERN = re.compile(
    r"^\[\d+\.\d+-\d+\.\d+\]\s+(SPEAKER_\d+|UNKNOWN):\s*(.+)$"
)


def build_transcript_text(lines):
    """Merge consecutive lines from the same speaker, then format as 'SPEAKER_XX: text'."""
    if not lines:
        return ""

    merged = []
    current_speaker, current_text = lines[0]

    for speaker, text in lines[1:]:
        if speaker == current_speaker:
            current_text += " " + text
        else:
            merged.append((current_speaker, current_text))
            current_speaker, current_text = speaker, text

    merged.append((current_speaker, current_text))

    return " ".join(f"{spk}: {txt}" for spk, txt in merged)


dataset = []

txt_files = sorted(
    (f for f in os.listdir(TRANSCRIPTS_FOLDER) if f.endswith(".txt")),
    key=lambda x: int(re.search(r'\d+', x).group())
)

if not txt_files:
    print(f"No .txt files found in '{TRANSCRIPTS_FOLDER}'. Did the transcription step run?")
else:
    for meeting_id, filename in enumerate(txt_files, start=1):
        filepath  = os.path.join(TRANSCRIPTS_FOLDER, filename)
        raw_lines = []
        skipped   = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = LINE_PATTERN.match(line)
                if match:
                    raw_lines.append((match.group(1), match.group(2)))
                else:
                    skipped += 1

        if skipped:
            print(f"  [{filename}] skipped {skipped} unmatched line(s)")

        if not raw_lines:
            print(f"  [{filename}] WARNING: no valid lines found — skipping")
            continue

        transcript_text = build_transcript_text(raw_lines)

        dataset.append({
            "meeting_id": meeting_id,
            "transcript": transcript_text
        })

        print(f"  [{filename}] -> meeting_id {meeting_id} | {len(raw_lines)} segments")

# Save
with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\nDataset saved to {OUTPUT_JSON_PATH}")
print(f"Total meetings: {len(dataset)}")

# Quick validation
print("\n── Validation ───────────────────────────────")
for entry in dataset:
    tid     = entry["meeting_id"]
    words   = len(entry["transcript"].split())
    preview = entry["transcript"][:80].replace("\n", " ")
    print(f"  meeting_id {tid}: {words} words | {preview}...")

print("\nDataset is ready for chunking.py")