# Windows wrapper; arguments and implementation are shared with Linux.
$ErrorActionPreference = 'Stop'
& python (Join-Path $PSScriptRoot 'setup_turbo.py') @args
exit $LASTEXITCODE
