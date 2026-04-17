from TTS.api import TTS
import os
import re
import time
import wave
import sys

tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")


def clean_for_tts(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(text, max_len=500):
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_len:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks


def merge_wav_files(input_files, output_file):
    data = []
    params = None

    for file in input_files:
        with wave.open(file, 'rb') as w:
            if params is None:
                params = w.getparams()
            data.append(w.readframes(w.getnframes()))

    with wave.open(output_file, 'wb') as output:
        output.setparams(params)
        for frames in data:
            output.writeframes(frames)


def run_tts(text):
    print("Generating audio...")
    os.makedirs("RAG/audio_results", exist_ok=True)

    clean_text = clean_for_tts(text)
    chunks = split_text(clean_text)

    audio_files = []

    for i, chunk in enumerate(chunks):
        print(f"🎤 Processing chunk {i}")
        temp_audio = f"RAG/audio_results/temp_{i}.wav"

        tts.tts_to_file(
            text=chunk,
            file_path=temp_audio
        )

        audio_files.append(temp_audio)

    final_audio = f"RAG/audio_results/audio_{int(time.time())}.wav"
    merge_wav_files(audio_files, final_audio)

    for f in audio_files:
        if os.path.exists(f):
            os.remove(f)

    return final_audio

if __name__ == "__main__":
    text = sys.argv[1]
    output = run_tts(text)
    print(output)
