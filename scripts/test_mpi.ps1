param([string]$Executable = (Join-Path $PSScriptRoot '../heat_mpi.exe'))
$ErrorActionPreference = 'Stop'
# Run after compiling the latest source. Every case must exit successfully
# and explicitly report validation=pass in the machine-readable CSV.
$cases = @(
    @(8,0,1), @(8,0,3),
    @(8,1,1), @(8,1,3),
    @(17,7,3), @(17,7,5),
    @(8,2,4), @(8,2,8),
    @(33,1,4), @(33,2,4), @(33,9,4), @(33,10,4),
    @(64,100,1), @(64,100,3), @(64,100,8)
)
foreach ($case in $cases) {
    $n, $steps, $ranks = $case
    Write-Host "Testing N=$n steps=$steps ranks=$ranks"
    $output = & mpiexec -n $ranks $Executable --n $n --steps $steps --validate
    if ($LASTEXITCODE -ne 0) { throw "MPI failed: N=$n steps=$steps ranks=$ranks" }
    $records = @($output | ConvertFrom-Csv)
    if ($records.Count -ne 1 -or $records[0].validation -ne 'pass') {
        throw "Expected one passing validation CSV record; got: $output"
    }
    $output | Write-Host
}
# These commands must reject invalid input coherently.
$invalidCases = @(
    @('--n','3','--steps','0'),
    @('--n','8','--steps','-1'),
    @('--n','abc','--steps','0'),
    @('--n','8'),
    @('--n','2147483648','--steps','0')
)
foreach ($arguments in $invalidCases) {
    & mpiexec -n 2 $Executable @arguments
    if ($LASTEXITCODE -eq 0) { throw "Invalid arguments unexpectedly succeeded: $arguments" }
}
& mpiexec -n 5 $Executable --n 4 --steps 0
if ($LASTEXITCODE -eq 0) { throw 'P > N unexpectedly succeeded' }
Write-Host 'All MPI correctness cases passed.'
