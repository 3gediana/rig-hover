# Serial sniff on COM3 without asserting either control line.
#
# The CH343 bridge on this board wires DTR/RTS into the ESP32-S3 auto-reset
# circuit: asserting them can drop the board into reset or bootloader mode, and
# a reset briefly drives the motors. Both are set false BEFORE Open() so the
# lines are never asserted, not even for a moment.
#
# Read-only: nothing is written to the port.

param(
    [string]$Port = 'COM3',
    [int]$Baud = 115200,
    [int]$Seconds = 6
)

$sp = New-Object System.IO.Ports.SerialPort $Port, $Baud, 'None', 8, 'One'
$sp.DtrEnable = $false
$sp.RtsEnable = $false
$sp.ReadTimeout = 300
$sp.WriteTimeout = 300
$sp.Encoding = [System.Text.Encoding]::UTF8

try {
    $sp.Open()
} catch {
    Write-Output "OPEN FAILED: $($_.Exception.Message)"
    exit 1
}

Write-Output "opened $Port @ $Baud  dtr=$($sp.DtrEnable) rts=$($sp.RtsEnable)  listening ${Seconds}s ..."
$deadline = (Get-Date).AddSeconds($Seconds)
$total = 0
$buf = New-Object System.Text.StringBuilder
while ((Get-Date) -lt $deadline) {
    try {
        $chunk = $sp.ReadExisting()
        if ($chunk.Length -gt 0) {
            [void]$buf.Append($chunk)
            $total += $chunk.Length
        }
    } catch {
        # ReadTimeout is expected while the line is idle
    }
    Start-Sleep -Milliseconds 50
}
$sp.Close()
$sp.Dispose()

Write-Output "bytes received: $total"
if ($total -gt 0) {
    Write-Output '--- raw output ---'
    Write-Output $buf.ToString()
    Write-Output '--- end ---'
} else {
    Write-Output 'line is silent'
}
