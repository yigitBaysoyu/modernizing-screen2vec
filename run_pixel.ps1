# Beispiele:
# Ab ID 90 bis ID 1000:
# .\run_pixel.ps1 -StartRunId 90
#
# Nur ID 90:
# .\run_pixel.ps1 -StartRunId 90 -EndRunId 90

param(
    [ValidateRange(1, 1000)]
    [int]$StartRunId = 1,

    [ValidateRange(1, 1000)]
    [int]$EndRunId = 1000,

    [string]$PythonExecutable = ".venv\Scripts\python.exe",

    [string]$RicoLayoutsDirectory = "C:\Users\Jonathan\Dev\rico-data\data\layouts"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

if ($StartRunId -gt $EndRunId) {
    throw "StartRunId must be less than or equal to EndRunId."
}

$scriptPath = Join-Path $PSScriptRoot "pixel.py"
$logDirectory = Join-Path $PSScriptRoot "pixellogs"

if (-not (Test-Path -LiteralPath $scriptPath -PathType Leaf)) {
    throw "pixel.py was not found at: $scriptPath"
}

if (-not (Test-Path -LiteralPath $RicoLayoutsDirectory -PathType Container)) {
    throw "RICO layouts directory was not found at: $RicoLayoutsDirectory"
}

if ([System.IO.Path]::IsPathRooted($PythonExecutable)) {
    $pythonPath = $PythonExecutable
}
elseif (
    $PythonExecutable.Contains("\") -or
    $PythonExecutable.Contains("/")
) {
    $pythonPath = Join-Path $PSScriptRoot $PythonExecutable
}
else {
    $pythonCommand = Get-Command $PythonExecutable -ErrorAction Stop
    $pythonPath = $pythonCommand.Source
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python executable was not found at: $pythonPath"
}

$scriptPath = (Resolve-Path -LiteralPath $scriptPath).Path
$pythonPath = (Resolve-Path -LiteralPath $pythonPath).Path
$RicoLayoutsDirectory = (
    Resolve-Path -LiteralPath $RicoLayoutsDirectory
).Path

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$completedCount = 0
$skippedCount = 0
$skippedRunIds = @()

foreach ($runId in $StartRunId..$EndRunId) {
    $logPath = Join-Path $logDirectory "$runId.txt"
    $imagePath = Join-Path $RicoLayoutsDirectory "$runId.jpg"
    $jsonPath = Join-Path $RicoLayoutsDirectory "$runId.json"
    $startTime = Get-Date

    @(
        "=== Wrapper ==="
        "Run ID: $runId"
        "Run range: $StartRunId through $EndRunId"
        "Image: $imagePath"
        "JSON: $jsonPath"
        "Log directory: $logDirectory"
        "Started: $($startTime.ToString('yyyy-MM-dd HH\:mm\:ss zzz'))"
        ""
    ) | Tee-Object -FilePath $logPath

    if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf)) {
        $skippedCount++
        $skippedRunIds += $runId

        @(
            "[WRAPPER] Image does not exist: $imagePath"
            "[WRAPPER] Skipping run ID $runId."
            "[WRAPPER] Python was not started."
        ) | Tee-Object -FilePath $logPath -Append

        continue
    }

    @(
        "[WRAPPER] Image exists."
        "[WRAPPER] Command: `"$pythonPath`" -u `"$scriptPath`" $runId"
        ""
    ) | Tee-Object -FilePath $logPath -Append

    $previousErrorActionPreference = $ErrorActionPreference

    try {
        # Windows PowerShell 5.1 converts native stderr output into
        # PowerShell ErrorRecords. Continue preserves the full Python
        # traceback instead of stopping after its first line.
        $ErrorActionPreference = "Continue"

        & $pythonPath -u $scriptPath $runId 2>&1 |
            ForEach-Object {
                $_.ToString()
            } |
            Tee-Object -FilePath $logPath -Append

        $exitCode = $LASTEXITCODE
    }
    catch {
        $exitCode = 1

        @(
            ""
            "[WRAPPER] PowerShell could not start or complete run $runId."
            "[WRAPPER] Exception type: $($_.Exception.GetType().FullName)"
            "[WRAPPER] Exception message: $($_.Exception.Message)"
            "[WRAPPER] Full PowerShell error:"
            "$($_ | Out-String)"
        ) | Tee-Object -FilePath $logPath -Append
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $endTime = Get-Date
    $duration = $endTime - $startTime

    @(
        ""
        "=== Wrapper result ==="
        "Finished: $($endTime.ToString('yyyy-MM-dd HH\:mm\:ss zzz'))"
        "Duration: $($duration.ToString())"
        "Exit code: $exitCode"
    ) | Tee-Object -FilePath $logPath -Append

    if ($exitCode -ne 0) {
        @(
            "[WRAPPER] Run $runId failed."
            "[WRAPPER] Stopping without starting run $($runId + 1)."
            "[WRAPPER] See the complete output in: $logPath"
        ) | Tee-Object -FilePath $logPath -Append

        exit $exitCode
    }

    $completedCount++

    "[WRAPPER] Run $runId completed successfully." |
        Tee-Object -FilePath $logPath -Append
}

$summaryLines = @(
    ""
    "=== Final wrapper summary ==="
    "Requested range: $StartRunId through $EndRunId"
    "Successfully completed: $completedCount"
    "Skipped because JPG was missing: $skippedCount"
)

if ($skippedCount -gt 0) {
    $summaryLines += "Skipped run IDs: $($skippedRunIds -join ', ')"
}

$summaryLines += "[WRAPPER] All available images in the requested range were processed."

$finalLogPath = Join-Path $logDirectory "$EndRunId.txt"

$summaryLines |
    Tee-Object -FilePath $finalLogPath -Append

exit 0