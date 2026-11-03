# Run from any directory. Starts training in a hidden background process.
param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runRelative = 'runs/balle2017/gray-lambda512'
$runPath = Join-Path $repoRoot $runRelative
New-Item -ItemType Directory -Force -Path $runPath | Out-Null
$processFile = Join-Path $runPath 'process.json'
if (Test-Path -LiteralPath $processFile) {
    $previous = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw "Recorded training process $($previous.pid) is still running."
    }
}
$checkpoint = Join-Path $runPath 'last.pt'
if ((Test-Path -LiteralPath $checkpoint) -and -not $Resume) {
    throw 'A checkpoint already exists. Use -Resume to continue it.'
}
if ($Resume -and -not (Test-Path -LiteralPath $checkpoint)) {
    throw 'No last.pt exists to resume.'
}
$arguments = @('-u', '-m', 'balle2017.train',
    '--train-manifest', 'data/train.txt', '--val-manifest', 'data/val.txt',
    '--output', $runRelative, '--mode', 'gray', '--lambda', '512',
    '--batch-size', '8', '--steps', '10000', '--validate-every', '250',
    '--saturation-fraction', '0.05', '--validation-crop', '256',
    '--device', 'cuda', '--dataset-description', 'imagenette-v2-full-resolution')
if ($Resume) { $arguments += @('--resume', "$runRelative/last.pt") }
$pythonPath = (Get-Command python).Source
$suffix = if ($Resume) { '-resume-' + (Get-Date -Format 'yyyyMMdd-HHmmss') } else { '' }
$outPath = Join-Path $runPath ("stdout$suffix.log")
$errPath = Join-Path $runPath ("stderr$suffix.log")
$trainingProcess = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repoRoot `
    -WindowStyle Hidden -RedirectStandardOutput $outPath -RedirectStandardError $errPath -PassThru
@{pid=$trainingProcess.Id; python=$pythonPath; arguments=$arguments; run=$runRelative;
    started=(Get-Date).ToString('o'); stdout=$outPath; stderr=$errPath} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $processFile -Encoding utf8
Get-Content -LiteralPath $processFile
