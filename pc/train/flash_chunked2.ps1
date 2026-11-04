# flash_chunked2.ps1 -- the CH343 link drops on long writes, so flash the app in
# small chunks and keep cycling over the ones that have not landed yet.
#
# Usage: powershell -File flash_chunked2.ps1 [-ChunkKB 32] [-Baud 460800]
param(
    [int]$ChunkKB = 32,
    [int]$Baud = 460800,
    [int]$MaxPasses = 12,
    [int]$MaxTriesPerPart = 40
)
$ErrorActionPreference = 'Continue'
$root = 'D:\robot_work\rig_omni'
$py = 'G:\esp\.espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe'
$bin = Join-Path $root 'build\rig-hover.bin'
$work = 'D:\robot_work\pc\train\build\chunks'
New-Item -ItemType Directory -Force -Path $work | Out-Null

$bytes = [System.IO.File]::ReadAllBytes($bin)
$chunk = $ChunkKB * 1024
$n = [math]::Ceiling($bytes.Length / $chunk)
Write-Output "image=$($bytes.Length) bytes, chunk=$chunk, parts=$n"

$parts = @()
for ($k = 0; $k -lt $n; $k++) {
    $off = $k * $chunk
    $len = [Math]::Min($chunk, $bytes.Length - $off)
    $file = Join-Path $work ("p{0:d3}.bin" -f $k)
    $slice = New-Object byte[] $len
    [Array]::Copy($bytes, $off, $slice, 0, $len)
    [System.IO.File]::WriteAllBytes($file, $slice)
    $parts += [pscustomobject]@{ idx = $k; addr = (0x20000 + $off); len = $len; file = $file; ok = $false; tries = 0 }
}

$t0 = Get-Date
for ($pass = 1; $pass -le $MaxPasses; $pass++) {
    $todo = $parts | Where-Object { -not $_.ok }
    if (-not $todo) { break }
    Write-Output ("pass {0}: {1} parts left, elapsed {2:n0}s" -f $pass, $todo.Count, ((Get-Date) - $t0).TotalSeconds)
    foreach ($p in $todo) {
        if ($p.tries -ge $MaxTriesPerPart) { continue }
        $p.tries++
        & $py -m esptool --chip esp32s3 -p COM3 -b $Baud --before default_reset --after no_reset `
            write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m `
            ("0x{0:x}" -f $p.addr) $p.file 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $p.ok = $true }
        Start-Sleep -Milliseconds 200
    }
}
$bad = $parts | Where-Object { -not $_.ok }
if ($bad) {
    Write-Output ("FAILED parts: " + (($bad | ForEach-Object { "$($_.idx)($($_.tries))" }) -join ', '))
    exit 1
}
& $py -m esptool --chip esp32s3 -p COM3 -b $Baud --before default_reset --after hard_reset run 2>&1 | Out-Null
Write-Output ("FLASH COMPLETE in {0:n0}s" -f ((Get-Date) - $t0).TotalSeconds)
exit 0
