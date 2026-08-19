"""End-to-end UTF-8 smoke test for the one-shot local TTS worker."""

from __future__ import annotations

import json
import os
import subprocess
import time
import wave
from pathlib import Path

python = Path(r"D:\Mo\voicetut_env\Scripts\python.exe")
worker = Path(__file__).with_name("voicetut_worker.py")
output = Path(r"D:\Mo\whisper_temp\voicetut_utf8_test.wav")
text = (
    "\u0623\u0647\u0644\u0627 \u064a\u0627 \u0641\u0646\u062f\u0645. \u0623\u0646\u0627 \u0633\u0627\u0631\u0629 \u0645\u0646 \u062a\u0645\u0648\u064a\u0644\u064a. "
    "\u062f\u064a \u0645\u0643\u0627\u0644\u0645\u0629 \u0622\u0644\u064a\u0629 \u0628\u062e\u0635\u0648\u0635 \u0627\u0644\u0642\u0633\u0637. \u0628\u0643\u0644\u0645 \u0627\u0644\u0634\u062e\u0635 \u0627\u0644\u0635\u062d\u061f"
)
env = os.environ.copy()
env.update({
    "HF_HOME": r"D:\Mo\huggingface",
    "HF_HUB_CACHE": r"D:\Mo\huggingface\hub",
    "TRANSFORMERS_CACHE": r"D:\Mo\huggingface\transformers",
    "TORCH_HOME": r"D:\Mo\torch_cache",
    "TEMP": r"D:\Mo\whisper_temp",
    "TMP": r"D:\Mo\whisper_temp",
})
payload = json.dumps({"text": text, "speaker": "Asmaa", "output": str(output)}, ensure_ascii=False)
started = time.monotonic()
result = subprocess.run(
    [str(python), str(worker)], input=payload, text=True, encoding="utf-8",
    capture_output=True, timeout=300, env=env,
)
if result.returncode:
    raise RuntimeError(result.stderr[-1000:])
with wave.open(str(output), "rb") as wav:
    duration = wav.getnframes() / wav.getframerate()
    assert wav.getframerate() == 24000
    assert wav.getnchannels() == 1
    assert duration > 3.0, f"Unexpectedly short audio: {duration:.2f}s"
print(json.dumps({
    "ok": True, "seconds": round(duration, 2), "bytes": output.stat().st_size,
    "generation_seconds": round(time.monotonic() - started, 2),
}))
