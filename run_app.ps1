$ErrorActionPreference = "Stop"

$Python = $null
if (Test-Path "D:\Mo\python.exe") {
    & "D:\Mo\python.exe" -c "import streamlit, whisper" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = "D:\Mo\python.exe" }
}
if (-not $Python -and (Test-Path ".venv\Scripts\python.exe")) {
    $HasStreamlit = & .\.venv\Scripts\python.exe -c "import streamlit" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = ".\.venv\Scripts\python.exe" }
}
if (-not $Python) {
    $Candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path $Candidate) { $Python = $Candidate; break }
    }
}
if (-not $Python) {
    throw "A usable Python installation was not found. Run .\setup_windows.ps1 first."
}

# Preserve support for an FFmpeg installation whose bin directory is supplied
# through FFMPEG_PATH, even when the system PATH has not refreshed yet.
if ($env:FFMPEG_PATH -and (Test-Path $env:FFMPEG_PATH)) {
    $env:Path = "$env:FFMPEG_PATH;$env:Path"
}
elseif (Test-Path "D:\Mo\ffmpeg-2026-08-06-git-95c43d7df7-essentials_build\bin\ffmpeg.exe") {
    $env:Path = "D:\Mo\ffmpeg-2026-08-06-git-95c43d7df7-essentials_build\bin;$env:Path"
}

if (-not $env:WHISPER_MODEL_DIR -and (Test-Path "D:\Mo\whisper_models")) {
    $env:WHISPER_MODEL_DIR = "D:\Mo\whisper_models"
}

$WhisperTemp = "D:\Mo\whisper_temp"
New-Item -ItemType Directory -Force -Path $WhisperTemp | Out-Null
$env:WHISPER_TEMP_DIR = $WhisperTemp
$env:TEMP = $WhisperTemp
$env:TMP = $WhisperTemp

# Keep only one AI model in RAM at a time. The app starts Qwen after ASR exits.
Get-Process "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force

& $Python -m streamlit run transcribe_arabic.py
