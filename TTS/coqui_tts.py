import asyncio
import edge_tts
import requests
import time
import os
import re
import io
import json
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = "sk_78987933ffbe1db6723b25d7c048fc767e23de6ae4627425"
ELEVENLABS_AR_VOICE = "XSgDtfUfQcFCMgyf6Viu"


# ── تنظيف النص ───────────────────────────────────────────────
def clean_for_tts(text):
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── تحويل للهجة المصرية عن طريق Groq ────────────────────────
def call_groq(text):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a STRICT Egyptian spoken Arabic converter.

RULES:
- Keep "SPEAKER_00", "SPEAKER_01", "SPEAKER_02" etc. exactly as is, do NOT translate or change them
- Convert to NATURAL Egyptian spoken Arabic ONLY
- Do NOT use Modern Standard Arabic
- Keep technical words in English (dataset, preprocessing, pipeline, scripts)
- Keep meaning EXACT - do NOT change or invent words
- No formatting, no headers, no explanations
- Output ONLY Egyptian spoken Arabic text
- If unsure about a word, keep it as is

EXAMPLES:
"إعداد تقرير" → "تعمل ريبورت"
"تخريم تقرير" → "تعمل ريبورت"
"تطهير" → "تنضيف"
"تصحيح الإدخالات الناقصة" → "تعدل اللي ناقص"
"مراجعة كاملة" → "تراجع كل حاجة"
"إكمال" → "تخلص"
"ترقية" → "تحديث"
"يجب" → "لازم"
"ينبغي" → "لازم"
"تم" → "خلصنا"

IMPORTANT: Never invent new words. If you don't know the Egyptian equivalent, keep the original word.

Text to convert:
{text}
"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Convert Arabic to Egyptian dialect only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    return text


# ── تحويل JSON التاسكات لكلام مصري ──────────────────────────
def tasks_to_speech_text(text):
    try:
        data = json.loads(text)
        if "tasks" not in data:
            return text

        lines = []

        for task in data["tasks"]:
            assignee = task.get("assignee", "")
            task_text = task.get("task", "")
            due = task.get("due_date", "")

            line = f"{assignee} هيعمل {task_text}"
            if due:
                line += f" ولازم يخلص {due}"

            lines.append(line)

        full_text = " . ".join(lines)

        # تحويل للمصري عن طريق Groq
        egyptian_text = call_groq(full_text)
        return egyptian_text

    except:
        return text


# ── تقسيم النص ───────────────────────────────────────────────
def split_text(text, max_len=200):
    sentences = re.split(r'[\.،]', text)
    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) < max_len:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "

    if current:
        chunks.append(current.strip())

    return chunks


# ── Detect Language ───────────────────────────────────────────
def detect_language(text):
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "en"
    if arabic_chars / total_chars > 0.2:
        return "ar"
    return "en"


# ── ElevenLabs TTS ────────────────────────────────────────────
def elevenlabs_tts(text, output_file):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_AR_VOICE}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY.strip(),
        "Content-Type": "application/json"
    }

    chunks = split_text(text)
    combined = AudioSegment.empty()

    for chunk in chunks:
        if not chunk.strip():
            continue

        payload = {
            "text": chunk,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.65,
                "similarity_boost": 0.85,
                "style": 0.25,
                "use_speaker_boost": True
            }
        }

        response = requests.post(url, json=payload, headers=headers)
        print(f"ElevenLabs status: {response.status_code}")

        if response.status_code == 200:
            audio = AudioSegment.from_file(
                io.BytesIO(response.content),
                format="mp3"
            )
            combined += audio
        else:
            print(f"ElevenLabs error: {response.text}")
            return False

    combined.export(output_file, format="mp3")
    return True


# ── Edge TTS fallback ─────────────────────────────────────────
async def edge_tts_fallback(text, output_file, lang):
    voice = "ar-EG-SalmaNeural" if lang == "ar" else "en-US-JennyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)


# ── Main Pipeline ─────────────────────────────────────────────
async def run_tts_async(text):
    os.makedirs("RAG/audio_results", exist_ok=True)

    clean_text = clean_for_tts(text)

    # JSON tasks → كلام مصري
    clean_text = tasks_to_speech_text(clean_text)

    lang = detect_language(clean_text)
    output_file = f"RAG/audio_results/audio_{int(time.time())}.mp3"

    print(f"Detected language: {lang}")

    if lang == "ar":
        print("Preparing Arabic speech...")
        print("BEFORE:", clean_text)

        # pause عشان الصوت يبقى طبيعي
        clean_text = clean_text.replace(".", "... ")

        print("AFTER:", clean_text)

        success = elevenlabs_tts(clean_text, output_file)

        if not success:
            print("Falling back to edge-tts")
            await edge_tts_fallback(clean_text, output_file, lang)

    else:
        print("Using edge-tts (English)")
        await edge_tts_fallback(clean_text, output_file, lang)

    print(f"Audio saved: {output_file}")
    return output_file


if __name__ == "__main__":
    import sys
    text = sys.argv[1]
    output = asyncio.run(run_tts_async(text))
    print(output)