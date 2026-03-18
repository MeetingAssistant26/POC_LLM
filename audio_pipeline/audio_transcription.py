import whisperx
import torch
import os

# -----------------------------
# Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

HF_TOKEN = "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"

model = whisperx.load_model(
    "medium",
    device,
    compute_type="int8"
)

audio_folder = "audio_clean"
output_folder = "transcripts"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Process each audio file
# -----------------------------
for filename in sorted(os.listdir(audio_folder)):

    if filename.endswith(".wav"):

        audio_path = os.path.join(audio_folder, filename)
        meeting_name = filename.replace(".wav", "")

        print("\nProcessing:", filename)

        # Load audio
        audio = whisperx.load_audio(audio_path)

        # -----------------------------
        # Transcription
        # -----------------------------
        result = model.transcribe(audio)

        # -----------------------------
        # Alignment
        # -----------------------------
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )

        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device
        )

        # -----------------------------
        # Speaker diarization
        # -----------------------------
        from whisperx.diarize import DiarizationPipeline

        diarize_model = DiarizationPipeline(
            token=HF_TOKEN,
            device=device
            )


        diarize_segments = diarize_model(audio_path)

        # -----------------------------
        # Assign speakers to words
        # -----------------------------
        result = whisperx.assign_word_speakers(
            diarize_segments,
            result
        )

        # -----------------------------
        # Save transcript
        # -----------------------------
        txt_path = os.path.join(output_folder, f"{meeting_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:

            for seg in result["segments"]:

                speaker = seg.get("speaker", "UNKNOWN")
                start = seg["start"]
                end = seg["end"]
                text = seg["text"]

                line = f"[{start:.2f}-{end:.2f}] {speaker}: {text}\n"
                f.write(line)

        print("Saved:", txt_path)

print("\nAll transcriptions completed!")