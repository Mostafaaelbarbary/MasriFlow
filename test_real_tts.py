"""Live smoke test for Gemini TTS. Does not print or modify the API key."""

import base64
import re
from pathlib import Path

import requests


secrets = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
match = re.search(r'^GEMINI_API_KEY\s*=\s*["\']([^"\']+)', secrets, re.MULTILINE)
assert match, "GEMINI_API_KEY is missing"
key = match.group(1)
url = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.5-flash-preview-tts:generateContent?key={key}"
)
response = requests.post(
    url,
    json={
        "contents": [{"parts": [{"text": "اقرأ بصوت مصري طبيعي: أهلاً، اختبار الصوت نجح."}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
            },
        },
    },
    timeout=90,
)
assert response.ok, f"HTTP {response.status_code}: {response.text[:300]}"
parts = response.json()["candidates"][0]["content"]["parts"]
inline = next(part.get("inlineData") for part in parts if part.get("inlineData"))
audio = base64.b64decode(inline["data"])
assert len(audio) > 1000, len(audio)
print(f"PASS: Gemini returned {len(audio)} bytes of speech audio")
