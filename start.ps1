[CmdletBinding()]
param(
  [ValidateSet("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")]
  [string]$Model = "base",

  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",

  [string]$Language,

  [switch]$Vtt
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Add-Type -AssemblyName System.Windows.Forms

$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = "Select a video file to caption"
$dialog.Filter = "Video files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v)|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v|All files (*.*)|*.*"
$dialog.Multiselect = $false

if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
  return
}

$videoPath = $dialog.FileName

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Write-Error "ffmpeg not found on PATH. Install ffmpeg and restart the terminal, then try again."
}

$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

$argsList = @(
  "caption_video.py",
  $videoPath,
  "--model", $Model,
  "--device", $Device
)

if ($Language) {
  $argsList += @("--language", $Language)
}

if ($Vtt) {
  $argsList += "--vtt"
}

& $python @argsList
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
  exit $exitCode
}

Write-Host ""
Write-Host "Done. Press Enter to close."
[void][System.Console]::ReadLine()
