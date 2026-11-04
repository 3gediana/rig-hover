$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime | Out-Null
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}
[void][Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime]
[void][Windows.Networking.NetworkOperators.NetworkOperatorTetheringAccessPointConfiguration, Windows.Networking.NetworkOperators, ContentType = WindowsRuntime]
[void][Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime]

$profile = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
$mgr = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($profile)
$cfg = $mgr.GetCurrentAccessPointConfiguration()
"BEFORE SSID=$($cfg.Ssid) BAND=$($cfg.Band) PASSPHRASE_LEN=$($cfg.Passphrase.Length)"

$new = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringAccessPointConfiguration]::new()
$new.Ssid = 'qasd'
$new.Passphrase = $cfg.Passphrase
$new.Band = [Windows.Networking.NetworkOperators.TetheringWiFiBand]::TwoPointFourGigahertz
try {
    $r = Await ($mgr.ConfigureAccessPointAsync($new)) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    "CONFIGURE=$($r.Status) $($r.AdditionalErrorMessage)"
} catch { "CONFIGURE_ERROR: $($_.Exception.Message)" }

Start-Sleep -Seconds 2
try {
    $s = Await ($mgr.StartTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    "START=$($s.Status) $($s.AdditionalErrorMessage)"
} catch { "START_ERROR: $($_.Exception.Message)" }
Start-Sleep -Seconds 6
"CLIENTS_AFTER=$($mgr.ClientCount)"