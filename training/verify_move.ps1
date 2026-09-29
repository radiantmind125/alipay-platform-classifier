# Verify one folder: D: original vs E: copy. Read-only (writes logs only). ASCII-only on purpose (PS 5.1 + GBK console).
# 1) robocopy /L compare: exit 0 = same names/sizes/times and no extra files on E:
# 2) count both sides: files + total bytes (language-neutral, long-path safe)
# 3) SHA-256 sample: N random files + all model/list-type files (capped at M)
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File verify_move.ps1 -Src D:\probe -Dst E:\probe -LogDir E:\_move_logs
param(
    [string]$Src,
    [string]$Dst,
    [string]$LogDir,
    [int]$Sample = 200,
    [int]$CriticalCap = 500
)
if (-not $Src -or -not $Dst -or -not $LogDir) {
    Write-Host 'usage: verify_move.ps1 -Src D:\probe -Dst E:\probe -LogDir E:\_move_logs'
    exit 2
}
if (-not (Test-Path -LiteralPath $Src)) { Write-Host ('source not found: ' + $Src); exit 2 }
if (-not (Test-Path -LiteralPath $Dst)) { Write-Host ('copy not found: ' + $Dst); exit 2 }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$name   = Split-Path $Src -Leaf
$none   = Join-Path $env:TEMP ('__robocopy_list_only_' + [guid]::NewGuid().ToString('N'))
$common = @('/L', '/E', '/XJ', '/XD', '.venv', '__pycache__', '/NJH', '/NJS', '/NDL', '/BYTES', '/FP', '/R:0', '/W:0')

# 1) compare
$diffLog = Join-Path $LogDir ("verify_" + $name + "_diff.log")
robocopy $Src $Dst @common "/UNILOG:$diffLog" | Out-Null
$rcDiff = $LASTEXITCODE

# 2) listings (/NC drops the localized file-class words; each line is just size + full path)
function Get-Listing([string]$Root, [string]$Log) {
    robocopy $Root $none @common '/NC' "/UNILOG:$Log" | Out-Null
    $rows = New-Object System.Collections.Generic.List[object]
    foreach ($ln in [IO.File]::ReadLines($Log, [Text.Encoding]::Unicode)) {
        if ($ln -match '^\s*(\d+)\s+([A-Za-z]:\\.+?)\s*$') {
            $rows.Add([pscustomobject]@{ Size = [int64]$Matches[1]; Path = $Matches[2] })
        }
    }
    return ,$rows
}
$srcRows = Get-Listing $Src (Join-Path $LogDir ("verify_" + $name + "_src_list.log"))
$dstRows = Get-Listing $Dst (Join-Path $LogDir ("verify_" + $name + "_dst_list.log"))
$srcBytes = ($srcRows | Measure-Object -Property Size -Sum).Sum
$dstBytes = ($dstRows | Measure-Object -Property Size -Sum).Sum
if ($null -eq $srcBytes) { $srcBytes = 0 }
if ($null -eq $dstBytes) { $dstBytes = 0 }

# 3) sample hashes
$critExt = '\.(onnx|pth|pt|ckpt|safetensors|txt|csv|jsonl|json|yaml|yml|cfg)$'
$crit = @($srcRows | Where-Object { $_.Path -match $critExt } | Select-Object -First $CriticalCap)
$rand = @($srcRows | Where-Object { $_.Path -notmatch $critExt } | Get-Random -Count ([Math]::Min($Sample, [Math]::Max(1, $srcRows.Count))))
$pick = @($crit + $rand | Sort-Object Path -Unique)
$bad = 0; $err = 0
$hashLog = Join-Path $LogDir ("verify_" + $name + "_sha256.log")
$lines = New-Object System.Collections.Generic.List[string]
foreach ($r in $pick) {
    $d = $Dst + $r.Path.Substring($Src.Length)
    try {
        $h1 = (Get-FileHash -LiteralPath $r.Path -Algorithm SHA256 -ErrorAction Stop).Hash
        $h2 = (Get-FileHash -LiteralPath $d -Algorithm SHA256 -ErrorAction Stop).Hash
        if ($h1 -ne $h2) { $bad++; $lines.Add("MISMATCH`t$($r.Path)`t$h1`t$h2") } else { $lines.Add("OK`t$($r.Path)`t$h1") }
    } catch {
        $err++; $lines.Add("ERROR`t$($r.Path)`t$($_.Exception.Message)")
    }
}
[IO.File]::WriteAllLines($hashLog, $lines, (New-Object Text.UTF8Encoding($true)))

$ok = ($rcDiff -eq 0) -and ($srcRows.Count -eq $dstRows.Count) -and ($srcBytes -eq $dstBytes) -and ($bad -eq 0) -and ($err -eq 0)
"{0,-28} files {1,10} / {2,-10} bytes {3,16} / {4,-16} diff-rc {5}  sha {6} ok {7} bad {8} err  => {9}" -f `
    $name, $srcRows.Count, $dstRows.Count, $srcBytes, $dstBytes, $rcDiff, ($pick.Count - $bad - $err), $bad, $err, $(if ($ok) { 'PASS' } else { 'FAIL' })
