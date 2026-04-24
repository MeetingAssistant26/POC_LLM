import requests
import time

API_KEY = "sk_437f4e4e843a0221ee0cb79ef43bd61ecdf4149b6ebbc539"

voices_to_test = [
    ("pFZP5JQG7iQjIQuC4Bku", "voice_1"),
    ("EXAVITQu4vr4xnSDxMaL", "voice_2"),
    ("onwK4e9ZLuTAKqWW03F9", "voice_3"),
]

test_text = "هاي يا جماعة إيه الدنيا كده"

for voice_id, name in voices_to_test:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": test_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }
    r = requests.post(url, json=payload, headers=headers)
    if r.status_code == 200:
        with open(f"{name}.mp3", "wb") as f:
            f.write(r.content)
        print(f"✅ {name} saved")
    else:
        print(f"❌ {name} failed: {r.status_code}")