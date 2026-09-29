# Read-only: what the target drive looks like before moving our work from D: to E:.
# ASCII-only on purpose (PS 5.1 on a GBK console misreads BOM-less UTF-8).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File training\move_layout.ps1
param([string]$D = 'D:', [string]$E = 'E:')
$D = $D.TrimEnd('\'); $E = $E.TrimEnd('\')

Write-Host "==== 1. E: drive: file system, type, bus, free space, long paths ===="
$disk = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $E + "'") -ErrorAction SilentlyContinue
if ($disk) {
    '  {0} FS={1} DriveType={2} free={3:N1} GB size={4:N1} GB' -f $disk.DeviceID, $disk.FileSystem, $disk.DriveType, ($disk.FreeSpace / 1GB), ($disk.Size / 1GB)
    try {
        Get-Partition -DriveLetter $E.Substring(0, 1) -ErrorAction Stop | Get-Disk |
            Select-Object Number, FriendlyName, BusType, OperationalStatus | Format-Table -AutoSize | Out-String -Width 200
    } catch { '  (Get-Partition not available: ' + $_.Exception.Message + ')' }
} else { '  ' + $E + ' is not a drive letter here' }
'  want: FS=NTFS, DriveType=3 (local disk), BusType not USB'
$lp = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -ErrorAction SilentlyContinue).LongPathsEnabled
'  LongPathsEnabled = ' + $lp + '   (1 = hashing works on paths longer than 260 characters)'

Write-Host "`n==== 2. what is in D:\alipay-ai-data (SHARED root: move only OUR subfolders, never the root) ===="
$r = $D + '\alipay-ai-data'
if (Test-Path -LiteralPath $r) {
    Get-ChildItem -LiteralPath $r -Force | Select-Object Mode, LastWriteTime, Length, Name | Format-Table -AutoSize | Out-String -Width 200
} else { '  ' + $r + ' not found' }

Write-Host "`n==== 3. top level of E:\ and E:\SSP_Work ===="
foreach ($p in @(($E + '\'), ($E + '\SSP_Work'))) {
    if (Test-Path -LiteralPath $p) {
        '  -- ' + $p
        Get-ChildItem -LiteralPath $p -Force | Select-Object Mode, LastWriteTime, Name | Format-Table -AutoSize | Out-String -Width 200
    } else { '  ' + $p + ' not found' }
}

Write-Host "`n==== 4. folders that may exist in BOTH places (the August move put some under E:\SSP_Work) ===="
'  counts are files that are missing or different (size/time) on the other side; /L = list only, nothing is copied'
$names = @('probe', 'localcrops', 'SSP', 'SSP-AI-Generated-Image-Detection-main', 'onnx')
foreach ($n in $names) {
    $a = $D + '\' + $n
    $b = $E + '\SSP_Work\' + $n
    $ea = Test-Path -LiteralPath $a
    $eb = Test-Path -LiteralPath $b
    if ($ea -and $eb) {
        $x = @(robocopy $a $b /L /E /XJ /XX /NJH /NJS /NDL /FP /R:0 /W:0 | Where-Object { $_.Trim() })
        $y = @(robocopy $b $a /L /E /XJ /XX /NJH /NJS /NDL /FP /R:0 /W:0 | Where-Object { $_.Trim() })
        '  {0}  vs  {1}' -f $a, $b
        '      only or different on D: {0,8}      only or different on E: {1,8}' -f $x.Count, $y.Count
        if ($x.Count) { '      e.g. ' + $x[0].Trim() }
        if ($y.Count) { '      e.g. ' + $y[0].Trim() }
    } else {
        '  {0,-45} exists {1,-6}  {2,-50} exists {3}' -f $a, $ea, $b, $eb
    }
}
Write-Host "`nDone. Nothing was changed."
