# Read-only: find anything that uses D:. Run from an elevated PowerShell ("Run as administrator"),
# otherwise other users' processes, tasks and SMB opens are invisible. ASCII-only on purpose.
$pat = '(?i)(^|[^A-Za-z])D:[\\/]'
$me  = [Security.Principal.WindowsIdentity]::GetCurrent()
$adm = (New-Object Security.Principal.WindowsPrincipal($me)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
Write-Host ("Elevated: " + $adm + "   (False = results below are INCOMPLETE)")

Write-Host "`n==== 1. processes whose exe or command line is on D: ===="
$all = @(Get-CimInstance Win32_Process)
$all | Where-Object { ($_.ExecutablePath -match $pat) -or ($_.CommandLine -match $pat) } | ForEach-Object {
    $o = Invoke-CimMethod -InputObject $_ -MethodName GetOwner
    [pscustomobject]@{ PID = $_.ProcessId; PPID = $_.ParentProcessId; Name = $_.Name; User = $o.User
                       Started = $_.CreationDate; Exe = $_.ExecutablePath; Cmd = $_.CommandLine }
} | Format-List
$blind = @($all | Where-Object { $_.ProcessId -gt 4 -and $null -eq $_.CommandLine })
Write-Host ("  processes whose command line could not be read: " + $blind.Count)

Write-Host "`n==== 2. every python / powershell / cmd / dotnet / w3wp process (wrapper scripts on E: can still write to D:) ===="
$all | Where-Object { $_.Name -match '^(python|pythonw|py|powershell|pwsh|cmd|dotnet|w3wp)\.exe$' } |
    Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine | Format-Table -AutoSize -Wrap | Out-String -Width 300

Write-Host "`n==== 3. processes with a DLL/.pyd loaded from D: (catches venv python started via launcher) ===="
Get-Process | ForEach-Object {
    $p = $_
    $mods = @()
    try { $mods = @($p.Modules) } catch { }
    $m = @($mods | Where-Object { $_.FileName -match $pat })
    if ($m.Count -gt 0) { [pscustomobject]@{ PID = $p.Id; Name = $p.ProcessName; DModules = $m.Count; Example = $m[0].FileName } }
} | Format-Table -AutoSize | Out-String -Width 300

Write-Host "`n==== 4. services: PathName on D:, or service registry values mentioning D: (NSSM etc.) ===="
Get-CimInstance Win32_Service | Where-Object { $_.PathName -match $pat } |
    Select-Object Name, State, StartMode, StartName, PathName | Format-List
Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Services' -ErrorAction SilentlyContinue | ForEach-Object {
    $svc = $_.PSChildName
    foreach ($kp in @($_.PSPath, (Join-Path $_.PSPath 'Parameters'))) {
        if (-not (Test-Path -LiteralPath $kp)) { continue }
        $props = Get-ItemProperty -LiteralPath $kp -ErrorAction SilentlyContinue
        if (-not $props) { continue }
        foreach ($pr in $props.PSObject.Properties) {
            if ($pr.Name -like 'PS*') { continue }
            $v = (@($pr.Value) -join ' ')
            if ($v -match $pat) { [pscustomobject]@{ Service = $svc; Value = $pr.Name; Data = $v } }
        }
    }
} | Format-Table -AutoSize -Wrap | Out-String -Width 300

Write-Host "`n==== 5. config files next to non-Windows service exes that mention D: ===="
Get-CimInstance Win32_Service | Where-Object { $_.PathName -and $_.PathName -notmatch '(?i)\\Windows\\' } | ForEach-Object {
    $pn = $_.PathName
    if ($pn -match '^\s*"([^"]+)"') { $exe = $Matches[1] } else { $exe = ($pn -split '\s+')[0] }
    $dir = Split-Path -Parent $exe
    if ($dir -and (Test-Path -LiteralPath $dir)) {
        Get-ChildItem -LiteralPath $dir -File -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -match '^\.(config|json|xml|ini|yaml|yml|toml|ps1|bat|cmd)$' -and $_.Length -lt 20MB } |
            Select-String -Pattern $pat -List -ErrorAction SilentlyContinue |
            ForEach-Object { [pscustomobject]@{ Service = $pn; File = $_.Path; Line = $_.Line.Trim() } }
    }
} | Format-List

Write-Host "`n==== 6. scheduled tasks whose action mentions D:, and task scripts that mention D: ===="
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue)
foreach ($t in $tasks) {
    foreach ($a in @($t.Actions)) {
        $s = "$($a.Execute) $($a.Arguments) $($a.WorkingDirectory)"
        if ($s -match $pat) {
            [pscustomobject]@{ Task = $t.TaskPath + $t.TaskName; State = $t.State; RunAs = $t.Principal.UserId; Action = $s } | Format-List
        }
        if ($t.TaskPath -notlike '\Microsoft\*') {
            foreach ($mm in [regex]::Matches($s, '(?i)[A-Z]:\\[^"<>|\r\n]+?\.(ps1|bat|cmd|py|vbs)')) {
                $f = $mm.Value
                if (Test-Path -LiteralPath $f) {
                    Select-String -LiteralPath $f -Pattern $pat -ErrorAction SilentlyContinue |
                        ForEach-Object { "  task " + $t.TaskName + " -> script " + $f + " line " + $_.LineNumber + ": " + $_.Line.Trim() }
                }
            }
        }
    }
}

Write-Host "`n==== 7. startup commands / Run keys on D: ===="
Get-CimInstance Win32_StartupCommand | Where-Object { $_.Command -match $pat } | Select-Object Name, Command, Location, User | Format-List

Write-Host "`n==== 8. environment variables (Machine + User) mentioning D: (PATH, PYTHONPATH, HF_HOME, TORCH_HOME ...) ===="
foreach ($scope in @('Machine', 'User')) {
    $h = [Environment]::GetEnvironmentVariables($scope)
    foreach ($k in $h.Keys) { if ("$($h[$k])" -match $pat) { "  [$scope] $k = $($h[$k])" } }
}
foreach ($pipIni in @((Join-Path $env:APPDATA 'pip\pip.ini'), (Join-Path $env:ProgramData 'pip\pip.ini'))) {
    if (Test-Path -LiteralPath $pipIni) { Select-String -LiteralPath $pipIni -Pattern $pat | ForEach-Object { "  $pipIni : " + $_.Line.Trim() } }
}

Write-Host "`n==== 9. SMB shares on D: and files opened over the network on D: ===="
if (Get-Command Get-SmbShare -ErrorAction SilentlyContinue) {
    Get-SmbShare -ErrorAction SilentlyContinue | Where-Object { $_.Path -match $pat } | Select-Object Name, Path, Description | Format-Table -AutoSize | Out-String -Width 300
    Get-SmbOpenFile -ErrorAction SilentlyContinue | Where-Object { $_.Path -match $pat } | Select-Object ClientComputerName, ClientUserName, Path | Format-Table -AutoSize | Out-String -Width 300
}

Write-Host "`n==== 10. IIS: applicationHost.config lines with D:, and site web.config/appsettings mentioning D: ===="
$ahc = Join-Path $env:windir 'System32\inetsrv\config\applicationHost.config'
if (Test-Path -LiteralPath $ahc) {
    Select-String -LiteralPath $ahc -Pattern $pat | ForEach-Object { "  line " + $_.LineNumber + ": " + $_.Line.Trim() }
    $phys = Select-String -LiteralPath $ahc -Pattern 'physicalPath="([^"]+)"' -AllMatches |
        ForEach-Object { $_.Matches } | ForEach-Object { [Environment]::ExpandEnvironmentVariables($_.Groups[1].Value) } | Sort-Object -Unique
    foreach ($pp in $phys) {
        if (Test-Path -LiteralPath $pp) {
            Get-ChildItem -LiteralPath $pp -File -Force -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^(web\.config|appsettings.*\.json)$' } |
                Select-String -Pattern $pat -ErrorAction SilentlyContinue |
                ForEach-Object { "  " + $_.Path + " line " + $_.LineNumber + ": " + $_.Line.Trim() }
        }
    }
} else { "  IIS not installed (no applicationHost.config)" }

Write-Host "`n==== 11. junctions / symlinks on C: and E: top levels that point into D: ===="
foreach ($top in @('C:\', 'E:\', 'E:\SSP_Work')) {
    if (-not (Test-Path -LiteralPath $top)) { continue }
    Get-ChildItem -LiteralPath $top -Force -ErrorAction SilentlyContinue |
        Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -and ((@($_.Target) -join ';') -match $pat) } |
        Select-Object FullName, LinkType, @{ n = 'Target'; e = { @($_.Target) -join ';' } } | Format-Table -AutoSize | Out-String -Width 300
}
Write-Host "`nDone. Nothing was changed."
