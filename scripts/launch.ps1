$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path $PSScriptRoot -Parent
$candidates = @((Join-Path $ragRoot '.venv\Scripts\python.exe'))
$candidates += @(Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*\python.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -ExpandProperty FullName)
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($pythonCommand) { $candidates += $pythonCommand.Source }
$ragPython = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $ragPython) { throw 'Install Python 3.11+ first, then run this launcher again.' }
if ($args -contains '-Setup') {
    if (-not (Test-Path (Join-Path $ragRoot '.venv\Scripts\python.exe'))) {
        & $ragPython -m venv (Join-Path $ragRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment setup failed.' }
    }
    & (Join-Path $ragRoot '.venv\Scripts\python.exe') -m pip install -r (Join-Path $ragRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
} else {
    $env:PYTHONUTF8 = '1'
    Start-Process -FilePath $ragPython -ArgumentList @(('"' + (Join-Path $ragRoot 'launcher.py') + '"')) -WorkingDirectory $ragRoot -WindowStyle Hidden
}
