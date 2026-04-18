import asyncio
import edge_tts
import sys
import time
import os
import re

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

async def run_tts_async(text):
    os.makedirs("RAG/audio_results", exist_ok=True)
    clean_text = clean_for_tts(text)
    
    lang = detect_language(clean_text)
    voice = "ar-EG-ShakirNeural" if lang == "ar" else "en-US-JennyNeural"
    
    output_file = f"RAG/audio_results/audio_{int(time.time())}.mp3"
    communicate = edge_tts.Communicate(clean_text, voice)
    await communicate.save(output_file)
    print(f"✅ Audio saved: {output_file}")
    return output_file

if __name__ == "__main__":
    text = sys.argv[1]
    output = asyncio.run(run_tts_async(text))
    print(output)
