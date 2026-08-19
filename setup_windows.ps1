$ErrorActionPreference = "Stop"

$Python = $null
$Candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)
foreach ($Candidate in $Candidates) {
    if (Test-Path $Candidate) { $Python = $Candidate; break }
}
if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
}
if (-not $Python) {
    throw "Python 3.10-3.12 was not found. Set it on PATH or install it from python.org."
}

if (-not (Test-Path ".venv")) {
    & $Python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete. Install FFmpeg if it is not already on PATH."
Write-Host "Start the app with: .\run_app.ps1"
