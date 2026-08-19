"""One-shot local Egyptian voice-cloning worker using Nile-XTTS."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


MODEL_DIR = Path(r"D:\Mo\NileTTS-XTTS")
DEFAULT_REFERENCE = Path(r"D:\Mo\fatima_egyptian_reference_24k.wav")


def main() -> None:
    request = json.load(sys.stdin)
    text = str(request["text"]).strip()
    output = Path(request["output"])
    reference = Path(request.get("reference_audio", str(DEFAULT_REFERENCE)))
    device = str(request.get("device", "cuda")).lower()
    if not text:
        raise ValueError("TTS text is empty")
    if not reference.exists():
        raise FileNotFoundError(f"Reference audio is missing: {reference}")
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")

    os.environ.setdefault("HF_HOME", r"D:\Mo\huggingface")
    os.environ.setdefault("TORCH_HOME", r"D:\Mo\torch_cache")

    import torch
    import torchaudio
    import transformers.pytorch_utils as transformers_pytorch_utils

    # Coqui 0.27 still imports this harmless compatibility helper, which was
    # removed from Transformers 5. torch.isin has the same required behavior.
    if not hasattr(transformers_pytorch_utils, "isin_mps_friendly"):
        transformers_pytorch_utils.isin_mps_friendly = torch.isin

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(str(MODEL_DIR / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(MODEL_DIR / "model.pth"),
        vocab_path=str(MODEL_DIR / "vocab.json"),
        use_deepspeed=False,
    )
    model.eval()
    model.to(device)

    conditioning, speaker = model.get_conditioning_latents(
        audio_path=str(reference),
        gpt_cond_len=6,
        max_ref_length=12,
        sound_norm_refs=False,
    )
    result = model.inference(
        text=text,
        language="ar",
        gpt_cond_latent=conditioning,
        speaker_embedding=speaker,
        temperature=0.65,
        repetition_penalty=3.0,
        top_k=50,
        top_p=0.85,
        speed=1.0,
        enable_text_splitting=True,
    )
    waveform = torch.tensor(result["wav"], dtype=torch.float32).unsqueeze(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), waveform.cpu(), 24000)
    if output.stat().st_size < 1000:
        raise RuntimeError("Nile-XTTS produced an invalid WAV")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
