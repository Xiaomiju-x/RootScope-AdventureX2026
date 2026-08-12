param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$Python = Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
$Server = Join-Path $PSScriptRoot 'human_review_server.py'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python runtime not found: $Python"
}
if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) {
    throw "Human review server not found: $Server"
}

Write-Host "RootScope A1 production reviewer binds only to 127.0.0.1:$Port"
Write-Host "Open http://127.0.0.1:$Port/ only after mode=PRODUCTION and DATA_LOCKED=false."
Write-Host "Stop with Ctrl+C. Never edit session.json, journal_checkpoint.json, or decision_journal.jsonl by hand."

& $Python -u $Server --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
