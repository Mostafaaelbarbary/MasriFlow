"""One-shot local Egyptian Arabic TTS worker using Lahgtna OmniVoice v3.

The worker exits after every reply so its GPU memory is released before the
ASR or chat model runs.  It accepts one JSON request on stdin and prints one
JSON result on stdout.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


MODEL_DIR = Path(r"D:\Mo\Lahgtna-v3")
REFERENCE_AUDIO = MODEL_DIR / "reference.wav"
REFERENCE_TEXT = "كان العمل التطوعي واللي لما تفتح الباب بس ليه الناس"


def _chunks(text: str, limit: int = 115) -> list[str]:
    """Keep prosodic phrases short without chopping individual words."""
    sentences = re.split(r"(?<=[.!?؟؛،])\s+", text.strip())
    result: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join((*current, word))
            if current and len(candidate) > limit:
                result.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            result.append(" ".join(current))
    return result or [text.strip()]


def main() -> None:
    request = json.load(sys.stdin)
    text = str(request["text"]).strip()
    output = Path(request["output"])
    steps = int(request.get("num_step", 16))
    if not text:
        raise ValueError("TTS text is empty")
    if not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError("Lahgtna v3 model weights are not installed")

    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", r"D:\Mo\huggingface")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\Mo\huggingface\hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\Mo\huggingface\transformers")
    os.environ.setdefault("TORCH_HOME", r"D:\Mo\torch_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    if not torch.cuda.is_available():
        raise RuntimeError("Lahgtna v3 requires the NVIDIA GPU for interactive use")
    model = OmniVoice.from_pretrained(
        str(MODEL_DIR), device_map="cuda", dtype=torch.float16
    )
    pieces = []
    silence = np.zeros(int(0.14 * 24000), dtype=np.float32)
    for phrase in _chunks(text):
        audio = model.generate(
            text=phrase,
            language="arz",
            ref_audio=str(REFERENCE_AUDIO),
            ref_text=REFERENCE_TEXT,
            num_step=steps,
        )[0]
        pieces.extend((np.asarray(audio, dtype=np.float32), silence))
    combined = np.concatenate(pieces[:-1])
    sf.write(str(output), combined, 24000, subtype="PCM_16")
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError("Lahgtna v3 did not create a valid audio file")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
