# flash_chunked.ps1 -- CH343 link drops on long writes; flash the app in
# 256 KB chunks, retrying each chunk until it lands.
# Usage: powershell -File flash_chunked.ps1 [-ChunkKB 256] [-Baud 460800]
param(
    [int]$ChunkKB = 256,
    [int]$Baud = 460800,
    [int]$MaxTries = 8
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

for ($k = 0; $k -lt $n; $k++) {
    $off = $k * $chunk
    $len = [Math]::Min($chunk, $bytes.Length - $off)
    $part = Join-Path $work ("p{0:d2}.bin" -f $k)
    $slice = New-Object byte[] $len
    [Array]::Copy($bytes, $off, $slice, 0, $len)
    [System.IO.File]::WriteAllBytes($part, $slice)
    $addr = 0x20000 + $off
    $done = $false
    for ($t = 1; $t -le $MaxTries; $t++) {
        & $py -m esptool --chip esp32s3 -p COM3 -b $Baud --before default_reset --after no_reset `
            write_flash --flash_mode dio --flash_size 16MB --flash_freq 80m `
            ("0x{0:x}" -f $addr) $part 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Output ("part {0}/{1} @0x{2:x} len={3} OK (try {4})" -f ($k + 1), $n, $addr, $len, $t)
            $done = $true
            break
        }
        Write-Output ("part {0}/{1} @0x{2:x} fail rc={3} (try {4})" -f ($k + 1), $n, $addr, $LASTEXITCODE, $t)
        Start-Sleep -Milliseconds 800
    }
    if (-not $done) {
        Write-Output ("PART {0} FAILED after {1} tries -- stopping" -f ($k + 1), $MaxTries)
        exit 1
    }
}
& $py -m esptool --chip esp32s3 -p COM3 -b $Baud --before default_reset --after hard_reset run 2>&1 | Out-Null
Write-Output "FLASH COMPLETE"
exit 0
