"""One-shot VoiceTut-TTS worker.

The process exits after each utterance so ASR, LLM, and TTS never compete for
the laptop's limited GPU/RAM at the same time.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    request = json.load(sys.stdin)
    text = str(request["text"]).strip()
    output = Path(request["output"])
    speaker = str(request.get("speaker", "Sarah"))
    if not text:
        raise ValueError("TTS text is empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", r"D:\Mo\huggingface")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\Mo\huggingface\hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\Mo\huggingface\transformers")
    os.environ.setdefault("TORCH_HOME", r"D:\Mo\torch_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from huggingface_hub import snapshot_download
    from voicetut_tts import VoiceTutTTS

    local_model = Path(r"D:\Mo\VoiceTut-TTS")
    model_dir = str(local_model) if (local_model / "model.safetensors").exists() else snapshot_download(
        "mohammedaly22/VoiceTut-TTS",
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "reference_speakers/*",
        ],
    )
    tts = VoiceTutTTS.from_pretrained(model_dir, language="arz")
    tts.synthesize_long(
        text,
        str(output),
        speaker=speaker,
        language="arz",
        normalize=False,
        max_chars=40,
        gap_ms=160,
    )
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError("VoiceTut did not create a valid audio file")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
