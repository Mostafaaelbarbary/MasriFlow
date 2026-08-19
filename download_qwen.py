import os
from pathlib import Path

os.environ["HF_HOME"] = r"D:\Mo\huggingface"
os.environ["HF_HUB_CACHE"] = r"D:\Mo\huggingface\hub"
os.environ["TEMP"] = r"D:\Mo\whisper_temp"
os.environ["TMP"] = r"D:\Mo\whisper_temp"

from huggingface_hub import hf_hub_download

target = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    local_dir=Path(r"D:\Mo\qwen25"),
    force_download=True,
)
print(target)
