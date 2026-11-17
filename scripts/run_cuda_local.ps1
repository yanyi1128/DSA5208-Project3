param(
    [ValidateSet('pilot', 'benchmark')][string]$Phase = 'pilot',
    [string]$OutputRoot = 'D:\DSA5208-CUDA-results',
    [int]$BuildJobs = 2,
    [switch]$SkipBuild,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if ($BuildJobs -lt 1 -or $BuildJobs -gt 4) { throw 'BuildJobs must be between 1 and 4.' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$imageName = 'dsa5208-cuda-local:cuda12.1.1'
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
if ($outputPath.Contains(',')) { throw 'OutputRoot must not contain a comma (Docker bind mount syntax).' }
$runName = 'local_' + $Phase + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$runArgs = @('run', '--rm', '--gpus', 'all', '--cpus', '4', '--memory', '6g',
    '--label', 'dsa5208.environment=local',
    '--mount', "type=bind,source=$projectRoot,target=/project,readonly",
    '--mount', "type=bind,source=$outputPath,target=/local-results",
    $imageName, 'python3', 'scripts/cuda_job.py', '--local', '--phase', $Phase,
    '--build-jobs', [string]$BuildJobs, '--output', "/local-results/$runName")
Write-Output 'LOCAL RTX GPU run: Windows + Docker/WSL2. This is not an Atlas/Vanda measurement.'
Write-Output "Results will be stored under: $outputPath\$runName"
if ($DryRun) {
    Write-Output ('docker ' + ($runArgs -join ' '))
    Write-Output 'Preview only: no Docker build, process launch, or output directory creation.'
    return
}
& docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with its WSL2 Linux backend, then retry.' }
if (-not $SkipBuild) {
    & docker build --progress=plain -t $imageName (Join-Path $PSScriptRoot 'docker-local')
    if ($LASTEXITCODE -ne 0) { throw 'Local CUDA image build failed.' }
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null
$provenance = [ordered]@{
    environment = 'local_windows_docker_wsl2'
    is_atlas_or_vanda = $false
    phase = $Phase
    run_directory = $runName
    utc_start = [DateTime]::UtcNow.ToString('o')
    source_root = $projectRoot
    image = $imageName
    docker_argv = $runArgs
    image_details = (& docker image inspect $imageName | Out-String)
    gpu_info = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | Out-String)
}
$provenance | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outputPath ($runName + '_host.json')) -Encoding UTF8
& docker @runArgs
$runExitCode = $LASTEXITCODE
if ($runExitCode -ne 0) { throw "Local $Phase exited with $runExitCode. Inspect saved stage logs under $outputPath\$runName." }
Write-Output "Completed local $Phase. Verify job_status.json under $outputPath\$runName."
