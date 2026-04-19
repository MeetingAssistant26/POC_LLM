from gtts import gTTS
import os
import time
import re
import sys

def clean_for_tts(text):
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text):
    arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
    if len(arabic_chars) > len(text) * 0.3:
        return "ar"
    return "en"

def run_tts(text):
    try:
        os.makedirs("RAG/audio_results", exist_ok=True)

        clean_text = clean_for_tts(text)
        lang = detect_language(clean_text)

        output_file = f"RAG/audio_results/audio_{int(time.time())}.mp3"

        tts = gTTS(text=clean_text, lang=lang)
        tts.save(output_file)

        print(f"✅ Audio saved: {output_file}")

        return output_file

    except Exception as e:
        print("❌ TTS Error:", e)
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide text")
    else:
        text = sys.argv[1]
        run_tts(text)
