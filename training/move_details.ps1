# Read-only follow-up checks before moving our work from D: to E:\SSP_Work.
# ASCII-only on purpose (PS 5.1 on a GBK console misreads BOM-less UTF-8).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File training\move_details.ps1
param([string]$D = 'D:', [string]$W = 'E:\SSP_Work')
$D = $D.TrimEnd('\'); $W = $W.TrimEnd('\')

Write-Host "==== 1. file-system drives on this machine ===="
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root,
    @{ n = 'UsedGB'; e = { [math]::Round($_.Used / 1GB, 1) } }, @{ n = 'FreeGB'; e = { [math]::Round($_.Free / 1GB, 1) } } |
    Format-Table -AutoSize | Out-String -Width 200

Write-Host "==== 2. which python the commands use (D:\Python must NOT move if it is this one) ===="
Get-Command python -All -ErrorAction SilentlyContinue | Select-Object Source | Format-Table -AutoSize | Out-String -Width 300
& python -c "import sys; print('  executable  ', sys.executable); print('  base_prefix ', sys.base_prefix); print('  version     ', sys.version.split()[0])"
'  PATH entries on ' + $D + ':'
foreach ($scope in @('Machine', 'User')) {
    $p = [Environment]::GetEnvironmentVariable('Path', $scope)
    foreach ($x in @($p -split ';')) { if ($x -and $x.StartsWith($D, [StringComparison]::OrdinalIgnoreCase)) { '    [' + $scope + '] ' + $x } }
}
'  virtual environments under ' + $D + '\alipay-ai-data and the Python each one is built on (home = ...):'
Get-ChildItem -LiteralPath ($D + '\alipay-ai-data') -Directory -Force -ErrorAction SilentlyContinue | ForEach-Object {
    foreach ($v in @((Join-Path $_.FullName '.venv'), (Join-Path $_.FullName 'venv'))) {
        $cfg = Join-Path $v 'pyvenv.cfg'
        if (Test-Path -LiteralPath $cfg) {
            $home1 = (Get-Content -LiteralPath $cfg | Where-Object { $_ -match '^\s*home\s*=' }) -join ' '
            '    ' + $v + '   ' + $home1
        }
    }
}

Write-Host "`n==== 3. loose files directly in D:\ (not in any folder) ===="
Get-ChildItem -LiteralPath ($D + '\') -File -Force -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String -Width 300

Write-Host "==== 4. loose files directly in D:\download2 (.dll/.exe/.pdb/.xml counted, everything else named) ===="
$lf = @(Get-ChildItem -LiteralPath ($D + '\download2') -File -Force -ErrorAction SilentlyContinue)
$prog = @($lf | Where-Object { $_.Extension -match '^\.(dll|exe|pdb|xml|json|config)$' })
'  program-like files (.dll .exe .pdb .xml .json .config): ' + $prog.Count + '   e.g. ' + ((@($prog | Where-Object { $_.Extension -eq '.exe' }) | Select-Object -First 5 | ForEach-Object { $_.Name }) -join ', ')
$lf | Where-Object { $_.Extension -notmatch '^\.(dll|exe|pdb|xml|json|config)$' } |
    Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize | Out-String -Width 300

Write-Host "==== 5. what is inside D:\alipay-ai-data\pytest, D:\alipay-ai-data\alipay-ai, acceptance, normal, abnormal ===="
foreach ($n in @('pytest', 'alipay-ai', 'acceptance', 'normal', 'abnormal')) {
    $p = $D + '\alipay-ai-data\' + $n
    if (-not (Test-Path -LiteralPath $p)) { '  ' + $p + ' not found'; continue }
    '  -- ' + $p
    Get-ChildItem -LiteralPath $p -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            $c = 0
            try { $c = @([IO.Directory]::EnumerateFiles($_.FullName, '*', [IO.SearchOption]::AllDirectories)).Count } catch { $c = -1 }
            '     {0,-55} {1:yyyy-MM-dd}  dir  files {2}' -f $_.Name, $_.LastWriteTime, $c
        } else {
            '     {0,-55} {1:yyyy-MM-dd}  file {2}' -f $_.Name, $_.LastWriteTime, $_.Length
        }
    }
}

Write-Host "`n==== 6. D:\probe -> E:\SSP_Work\probe: files that exist on BOTH sides but differ (a merge would have to choose) ===="
$a = $D + '\probe'; $b = $W + '\probe'
if ((Test-Path -LiteralPath $a) -and (Test-Path -LiteralPath $b)) {
    $x = @(robocopy $a $b /L /E /XJ /XX /XL /NJH /NJS /NDL /FP /R:0 /W:0 | Where-Object { $_.Trim() })
    '  conflicts: ' + $x.Count + '   (0 = every D:\probe file is new to E:, a plain merge overwrites nothing)'
    $x | Select-Object -First 10 | ForEach-Object { '    ' + $_.Trim() }
} else { '  ' + $a + ' exists ' + (Test-Path -LiteralPath $a) + '   ' + $b + ' exists ' + (Test-Path -LiteralPath $b) }

Write-Host "`n==== 7. where is the old D:\download? (look for a 'download' folder at the top of every drive) ===="
Get-PSDrive -PSProvider FileSystem | ForEach-Object {
    foreach ($n in @('download', 'download_old', 'download_bak')) {
        $p = Join-Path $_.Root $n
        if (Test-Path -LiteralPath $p) { '  found: ' + $p + '   last write ' + (Get-Item -LiteralPath $p).LastWriteTime }
    }
}
'  (nothing listed = no download folder at the top of any drive)'

Write-Host "`n==== 8. is the manager's upload still arriving? (file count and newest file time, twice, 60 s apart) ===="
foreach ($round in 1, 2) {
    foreach ($n in @('BlueImages', 'OtherImages')) {
        $p = $D + '\download2\' + $n
        if (-not (Test-Path -LiteralPath $p)) { continue }
        # DirectoryInfo gives the time straight from the directory listing (no extra disk read per file)
        $cnt = 0; $newest = [datetime]::MinValue
        foreach ($f in (New-Object IO.DirectoryInfo $p).EnumerateFiles()) { $cnt++; if ($f.LastWriteTime -gt $newest) { $newest = $f.LastWriteTime } }
        '  round {0}  {1,-12} files {2,9:N0}   newest file {3:yyyy-MM-dd HH:mm:ss}' -f $round, $n, $cnt, $newest
    }
    if ($round -eq 1) { Start-Sleep -Seconds 60 }
}
'  now: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '   (count grew between rounds, or newest file is minutes old = still uploading)'

Write-Host "`nDone. Nothing was changed."
