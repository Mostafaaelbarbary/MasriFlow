"""One-shot local Egyptian TTS worker using Chatterbox Multilingual V3."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

DEFAULT_REFERENCE = Path(r"D:\Mo\Egyptian-Chatterbox\reference_audio.wav")
EGYPTIAN_CHECKPOINT = Path(r"D:\Mo\Egyptian-Chatterbox\model.safetensors")


def _chunks(text: str, limit: int = 190) -> list[str]:
    sentences = re.split(r"(?<=[.!?؟])\s+", text.strip())
    output: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        current: list[str] = []
        for word in words:
            candidate = " ".join((*current, word))
            if current and len(candidate) > limit:
                output.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            output.append(" ".join(current))
    return [part for part in output if part.strip()] or [text.strip()]


def main() -> None:
    request = json.load(sys.stdin)
    text = str(request["text"]).strip()
    output = Path(request["output"])
    reference_audio = Path(request.get("reference_audio", str(DEFAULT_REFERENCE)))
    if not text:
        raise ValueError("TTS text is empty")
    if not reference_audio.exists():
        raise FileNotFoundError("The Egyptian reference voice is missing")
    if not EGYPTIAN_CHECKPOINT.exists():
        raise FileNotFoundError("The Egyptian Chatterbox checkpoint is missing")

    os.environ.setdefault("HF_HOME", r"D:\Mo\huggingface")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\Mo\huggingface\hub")
    os.environ.setdefault("TORCH_HOME", r"D:\Mo\torch_cache")
    os.environ.setdefault("PKUSEG_HOME", r"D:\Mo\pkuseg")
    os.environ.pop("TRANSFORMERS_CACHE", None)

    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file as load_safetensors

    requested_device = str(request.get("device", "cuda")).lower()
    if requested_device not in {"cuda", "cpu"}:
        raise ValueError("TTS device must be 'cuda' or 'cpu'")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but the NVIDIA GPU is unavailable")
    use_egyptian_checkpoint = bool(request.get("egyptian_checkpoint", False))
    model_version = "v2" if use_egyptian_checkpoint else "v3"
    model = ChatterboxMultilingualTTS.from_pretrained(
        device=requested_device, t3_model=model_version
    )
    if use_egyptian_checkpoint:
        incompatible = model.t3.load_state_dict(
            load_safetensors(str(EGYPTIAN_CHECKPOINT), device="cpu"),
            strict=False,
            assign=False,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Egyptian checkpoint is incompatible: "
                f"missing={len(incompatible.missing_keys)}, "
                f"unexpected={len(incompatible.unexpected_keys)}"
            )
        model.t3.eval()

    pieces = []
    silence = np.zeros(int(0.14 * model.sr), dtype=np.float32)
    for phrase in _chunks(text):
        waveform = model.generate(
            phrase,
            language_id="ar",
            audio_prompt_path=str(reference_audio),
            exaggeration=0.5,
            cfg_weight=0.5,
            temperature=0.8,
        )
        pieces.extend((waveform.detach().float().cpu().numpy().squeeze(), silence))
    combined = np.concatenate(pieces[:-1])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), combined, model.sr, subtype="PCM_16")
    if not output.exists() or output.stat().st_size < 1000:
        raise RuntimeError("Chatterbox did not create a valid WAV file")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
