"""Small subprocess entry point for the isolated QwenCleo environment."""

import json
import os
import sys
import time

# Keep multi-gigabyte model caches off the small Windows system drive.
os.environ.setdefault("HF_HOME", r"D:\Mo\huggingface")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Mo\huggingface\hub")
os.environ.setdefault("TORCH_HOME", r"D:\Mo\torch_cache")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from qwencleo_asr import QwenCleoASR


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")

started = time.perf_counter()
model = QwenCleoASR(device="cpu")
result = model.transcribe(sys.argv[1], language="Arabic", normalize=True)
print(json.dumps({
    "text": result.text.strip(),
    "elapsed": time.perf_counter() - started,
}, ensure_ascii=False))
