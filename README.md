# Egyptian Arabic Transcriber

A Streamlit application for high-accuracy Egyptian Arabic transcription. Free local OpenAI Whisper is the default engine and requires no API key. Optional Deepgram Nova-3 (`ar-EG`) remains available as a paid cloud engine. The app supports audio/video uploads, browser recording, silence-triggered automatic transcription, and per-word exports.

## Deepgram Nova-3

Create a Deepgram API key, select **Deepgram Nova-3 — cloud**, and paste the key into the password field in the sidebar. The application sends recordings to Deepgram for transcription; the key is held only in the running Streamlit session and is not stored in the project. Deepgram usage requires internet access and may incur account charges.

The **Auto-listen after silence** mode keeps the microphone open. After speech starts and approximately 1.2 seconds of silence is detected, it automatically sends that phrase to `large-v3` and resumes listening afterward. No separate Stop or Transcribe click is required. On the current CPU-only setup, maximum accuracy means each phrase may require several minutes.

## Recommended configuration

For highest accuracy on the current machine:

- Model: **large-v3**
- Device: **CPU** with the currently installed CPU-only PyTorch
- Beam size: **5**
- Silence removal: enabled

`large-v3` is the default. It generally exceeds the practical memory available on a 4 GB RTX 3050 with standard OpenAI Whisper, so CPU is the reliable high-accuracy mode. Clear speech and a close microphone further improve results.

Word timestamps are Whisper estimates. The app can export every recognized word separately, but Whisper still recognizes speech in contextual chunks rather than emitting guaranteed live words one at a time.

## Windows installation

1. Install 64-bit Python 3.10, 3.11, or 3.12. The setup script also detects the standard per-user Python installation even if `PATH` is stale.
2. Install [FFmpeg](https://ffmpeg.org/download.html) and make sure `ffmpeg` is on `PATH`.
3. For GPU processing, install a current NVIDIA driver and a CUDA-enabled PyTorch build; if CUDA is unavailable, use CPU mode.
4. Open PowerShell in this folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_app.ps1
```

On the original development machine, `run_app.ps1` automatically detects the existing Python 3.10, FFmpeg installation, and models under `D:\Mo\whisper_models`. Run `setup_windows.ps1` only on a machine where the Python packages are not already installed.

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

If FFmpeg is installed but its directory is not active on `PATH`, point the launcher to its `bin` folder:

```powershell
$env:FFMPEG_PATH = "D:\path\to\ffmpeg\bin"
.\run_app.ps1
```

The first use downloads the selected model into the local model cache. To use a custom directory:

```powershell
$env:WHISPER_MODEL_DIR = "D:\WhisperModels"
.\run_app.ps1
```

## Output formats

- `transcript.txt`: editable plain transcript
- `transcript.srt`: readable phrase-level subtitles
- `words.srt`: one subtitle cue per recognized word
- `words.vtt`: one WebVTT cue per word
- `words.csv`: word, start/end time, and recognition probability

## Accuracy tips

- Record close to the speaker in a quiet room.
- Avoid music, echo, and multiple people speaking simultaneously.
- Put uncommon names and English technical terms in the context box.
- Review low-confidence words in the CSV against the recording.
- Use `large-v3` on CPU for the most difficult recordings if processing time is acceptable.
