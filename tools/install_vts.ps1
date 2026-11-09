<#
Install (or restore) this OpenSeeFace build as VTube Studio's webcam tracker.

  powershell -ExecutionPolicy Bypass -File install_vts.ps1            # install
  powershell -ExecutionPolicy Bypass -File install_vts.ps1 -Restore   # put VTS's original tracker back
  powershell -ExecutionPolicy Bypass -File install_vts.ps1 -VtsDir "E:\Games\VTube Studio"   # custom location

The VTube Studio folder is found through Steam's library list; pass -VtsDir if that fails.
Works from the release zip (facetracker\ next to this script) or from a source checkout (dist\facetracker).

VTube Studio runs StreamingAssets\OpenSeeFaceTracker_v_1_20_2\binary\facetracker.exe
with --model-dir pointing at ...\models, so the same layout is kept:
  binary\  <- dist\facetracker (facetracker.exe + _internal)
  models\  <- dist\facetracker\models (original models + portable\ + MediaPipe .task)
The original folder is renamed to OpenSeeFaceTracker_v_1_20_2.vts-original and is never modified.
Steam's "Verify integrity of game files" also restores the original.
#>
param(
    [string]$VtsDir = "",
    [string]$Dist = "",
    [switch]$Restore
)
$ErrorActionPreference = "Stop"

function Find-VtsDir {
    $steam = (Get-ItemProperty "HKCU:\Software\Valve\Steam" -ErrorAction SilentlyContinue).SteamPath
    $libs = @()
    if ($steam) {
        $libs += $steam
        $vdf = Join-Path $steam "steamapps\libraryfolders.vdf"
        if (Test-Path $vdf) {
            foreach ($m in Select-String -Path $vdf -Pattern '"path"\s+"([^"]+)"') {
                $libs += $m.Matches[0].Groups[1].Value -replace '\\\\', '\'
            }
        }
    }
    foreach ($lib in $libs | Select-Object -Unique) {
        $candidate = Join-Path $lib "steamapps\common\VTube Studio"
        if (Test-Path (Join-Path $candidate "VTube Studio.exe")) { return $candidate }
    }
    return $null
}

if (-not $VtsDir) { $VtsDir = Find-VtsDir }
if (-not $VtsDir -or -not (Test-Path $VtsDir)) { throw "VTube Studio not found. Pass its folder with -VtsDir." }
if (-not $Dist) {
    $Dist = @((Join-Path $PSScriptRoot "facetracker"), (Join-Path $PSScriptRoot "..\dist\facetracker")) |
        Where-Object { Test-Path (Join-Path $_ "facetracker.exe") } | Select-Object -First 1
    if (-not $Dist) { $Dist = Join-Path $PSScriptRoot "facetracker" }
}
Write-Host "VTube Studio: $VtsDir"
$assets = Join-Path $VtsDir "VTube Studio_Data\StreamingAssets"
$target = Join-Path $assets "OpenSeeFaceTracker_v_1_20_2"
$backup = "$target.vts-original"

if (Get-Process -Name "VTube Studio", "facetracker" -ErrorAction SilentlyContinue) {
    throw "Close VTube Studio (and its facetracker.exe) first."
}

if ($Restore) {
    if (-not (Test-Path $backup)) { throw "No backup found at $backup" }
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    Rename-Item $backup (Split-Path $target -Leaf)
    Write-Host "Restored VTube Studio's original tracker."
    return
}

$Dist = (Resolve-Path $Dist).Path
if (-not (Test-Path (Join-Path $Dist "facetracker.exe"))) { throw "Build first: make_exe.bat openvino" }
foreach ($need in "models\portable\mp_hand_detector.onnx", "models\hand_landmarker.task", "models\lm_model3_opt.onnx") {
    if (-not (Test-Path (Join-Path $Dist $need))) { throw "dist is missing $need" }
}

if (-not (Test-Path $backup)) {
    Rename-Item $target (Split-Path $backup -Leaf)
    Write-Host "Backed up original tracker to $backup"
} elseif (Test-Path $target) {
    Remove-Item $target -Recurse -Force   # previous install of this build; the backup stays untouched
}

New-Item -ItemType Directory -Path (Join-Path $target "binary") | Out-Null
Copy-Item (Join-Path $Dist "facetracker.exe") (Join-Path $target "binary")
Copy-Item (Join-Path $Dist "_internal") (Join-Path $target "binary") -Recurse
Copy-Item (Join-Path $Dist "models") $target -Recurse
Write-Host "Installed into $target"
