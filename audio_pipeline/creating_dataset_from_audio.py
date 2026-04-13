import os
import re
import json

transcripts_folder = "transcripts"
output_path = "data/meeting_transcripts_audio.json"

os.makedirs("data", exist_ok=True)

# WhisperX saves each line as:
# [0.00-3.40] SPEAKER_00: some text here
LINE_PATTERN = re.compile(
    r"^\[\d+\.\d+-\d+\.\d+\]\s+(SPEAKER_\d+|UNKNOWN):\s*(.+)$"
)

# Map WhisperX generic labels to real names based on order of first appearance.
# WhisperX assigns SPEAKER_00, SPEAKER_01, SPEAKER_02 in the order they first speak.
# The meetings use Ahmed, Sara, Omar — adjust this list if your audio has a
# different speaking order.
SPEAKER_NAMES = ["Ahmed", "Sara", "Omar"]


def resolve_speakers(lines):
    """Replace SPEAKER_XX labels with real names in order of first appearance."""
    label_to_name = {}
    name_index = 0
    resolved = []

    for label, text in lines:
        if label not in label_to_name:
            if name_index < len(SPEAKER_NAMES):
                label_to_name[label] = SPEAKER_NAMES[name_index]
                name_index += 1
            else:
                label_to_name[label] = label  # keep original if more speakers found
        resolved.append((label_to_name[label], text.strip()))

    return resolved


def build_transcript_text(lines):
    """Merge consecutive lines from the same speaker, then format as 'Name: text'."""
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


# ── Main loop ────────────────────────────────────────────────
dataset = []

txt_files = sorted(
    (f for f in os.listdir(transcripts_folder) if f.endswith(".txt")),
    key=lambda x: int(re.search(r'\d+', x).group())
)

if not txt_files:
    print(f"No .txt files found in '{transcripts_folder}'. Did the transcription step run?")
else:
    for meeting_id, filename in enumerate(txt_files, start=1):
        filepath = os.path.join(transcripts_folder, filename)

        raw_lines = []
        skipped = 0

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

        named_lines = resolve_speakers(raw_lines)
        transcript_text = build_transcript_text(named_lines)

        dataset.append({
            "meeting_id": meeting_id,
            "transcript": transcript_text
        })

        print(f"  [{filename}] -> meeting_id {meeting_id} | {len(named_lines)} segments")

# ── Save ─────────────────────────────────────────────────────
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\nDataset saved to {output_path}")
print(f"Total meetings: {len(dataset)}")

# ── Quick validation ──────────────────────────────────────────
print("\n── Validation ───────────────────────────────")
for entry in dataset:
    tid = entry["meeting_id"]
    words = len(entry["transcript"].split())
    preview = entry["transcript"][:80].replace("\n", " ")
    print(f"  meeting_id {tid}: {words} words | {preview}...")

print("\nDataset is ready for chunking.py")
