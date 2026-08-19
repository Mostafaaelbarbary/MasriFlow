param(
    [Parameter(Mandatory=$true)][string]$TextFile,
    [Parameter(Mandatory=$true)][string]$OutputFile
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]
[void][Windows.Media.SpeechSynthesis.SpeechSynthesisStream, Windows.Media.SpeechSynthesis, ContentType=WindowsRuntime]

function Await-WinRt($Operation, [Type]$ResultType) {
    $method = [System.WindowsRuntimeSystemExtensions].GetMethods() |
        Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 } |
        Select-Object -First 1
    $task = $method.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.Wait()
    return $task.Result
}

$text = Get-Content -Raw -Encoding UTF8 -LiteralPath $TextFile
$synth = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::new()
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices |
    Where-Object { $_.Language -eq 'ar-EG' } |
    Select-Object -First 1
if (-not $voice) { throw 'Microsoft Hoda Arabic (Egypt) voice is not installed.' }
$synth.Voice = $voice
$synth.Options.SpeakingRate = 1.08
$synth.Options.AudioPitch = 1.02
$stream = Await-WinRt ($synth.SynthesizeTextToStreamAsync($text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$input = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream)
$output = [System.IO.File]::Create($OutputFile)
try { $input.CopyTo($output) } finally { $output.Dispose(); $input.Dispose(); $stream.Dispose(); $synth.Dispose() }
Write-Output "Generated $OutputFile with $($voice.DisplayName)"
