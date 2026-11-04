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
[void][Windows.Networking.Connectivity.NetworkInformation, Windows.Networking.Connectivity, ContentType = WindowsRuntime]
$profile = [Windows.Networking.Connectivity.NetworkInformation]::GetInternetConnectionProfile()
if (-not $profile) { 'NO_INTERNET_PROFILE'; exit 1 }
"PROFILE=$($profile.ProfileName) CONNECTIVITY=$($profile.NetworkAdapter.NetworkAdapterId)"
$caps = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::GetTetheringCapabilityFromConnectionProfile($profile)
"TETHERING_CAPABILITY=$caps"
$mgr = [Windows.Networking.NetworkOperators.NetworkOperatorTetheringManager]::CreateFromConnectionProfile($profile)
"CLIENTS=$($mgr.ClientCount) MAX=$($mgr.MaxClientCount)"
try {
    $stop = Await ($mgr.StopTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    "STOP=$($stop.Status) $($stop.AdditionalErrorMessage)"
} catch { "STOP_ERROR: $($_.Exception.Message)" }
Start-Sleep -Seconds 2
try {
    $start = Await ($mgr.StartTetheringAsync()) ([Windows.Networking.NetworkOperators.NetworkOperatorTetheringOperationResult])
    "START=$($start.Status) $($start.AdditionalErrorMessage)"
} catch { "START_ERROR: $($_.Exception.Message)" }
Start-Sleep -Seconds 3
"CLIENTS_AFTER=$($mgr.ClientCount)"