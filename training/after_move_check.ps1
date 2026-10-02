# Read-only: after our work was moved from D: to E:, find where each folder went and check it arrived whole.
# The expected file counts are from the D: inventory taken on 2026-09-29, before the move.
# ASCII-only on purpose (PS 5.1 on a GBK console misreads BOM-less UTF-8).
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File training\after_move_check.ps1
param([string]$E = 'E:\', [string]$D = 'D:\', [int]$MaxDepth = 4)

# name -> expected file count on 2026-09-29 (0 = we did not record it)
$expect = [ordered]@{
    'pinyin-pairs' = 645934; 'pinyin-pairs-blue' = 86039; 'pinyin-pairs-blue2' = 178431
    'pinyin-pairs-sep-white' = 83369; 'pinyin-pairs-sep-blue' = 26515
    'pinyin-pairs-sep-white-test' = 7751; 'pinyin-pairs-sep-blue-test' = 3085
    'pinyin-all' = 9069; 'newbatch_20260925' = 2; 'newbatch_20260928' = 3318; 'cap-test' = 6
    'blown_blue' = 2; 'blown_white' = 2; 'local_ratio_look' = 24; 'mask_look' = 4; 'trunc_look' = 6
    'pytest' = 15711; 'normal' = 12; 'acceptance' = 1317; 'alipay-ai' = 55579
    'pinyin_hits' = 7206; 'pinyin_hits_blue' = 3623; 'pinyin_sheets' = 16; 'prof_paddle' = 903; 'prof_none' = 903
    'holdout30k' = 30000; 'holdout30k_out' = 4
}
# loose files of ours that sat in D:\alipay-ai-data (name -> size in bytes on 2026-09-29)
$looseExpect = [ordered]@{
    'rec_pinyin.onnx' = 39604334; 'charset.txt' = 14014; 'pinyin-pairs.7z' = 2902551586
    'dump_white.jsonl' = 1832526; 'dump_blue.jsonl' = 1418006
}

Write-Host "==== 1. python and the repo ===="
Get-Command python -All -ErrorAction SilentlyContinue | Select-Object Source | Format-Table -AutoSize | Out-String -Width 300
& python -c "import sys; print('  python', sys.version.split()[0], sys.executable)"
$repo = Join-Path $E 'SSP_Work\alipay-platform-classifier'
if (Test-Path -LiteralPath $repo) {
    '  repo ' + $repo + '   last commit: ' + (& git -C $repo log -1 --format='%h %ad %s' --date=short 2>$null)
    $dirty = @(& git -C $repo status --short 2>$null)
    '  uncommitted changes in repo: ' + $dirty.Count
} else { '  repo NOT FOUND at ' + $repo }

Write-Host "`n==== 2. drives ===="
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Root,
    @{ n = 'UsedGB'; e = { [math]::Round($_.Used / 1GB, 1) } }, @{ n = 'FreeGB'; e = { [math]::Round($_.Free / 1GB, 1) } } |
    Format-Table -AutoSize | Out-String -Width 200

Write-Host "==== 3. where each of our folders is now on E:, and whether it arrived whole ===="
'  searching ' + $E + ' up to ' + $MaxDepth + ' levels deep (skips junctions) ...'
$found = @{}
$queue = New-Object System.Collections.Generic.Queue[object]
$queue.Enqueue(@((New-Object IO.DirectoryInfo $E), 0))
while ($queue.Count -gt 0) {
    $item = $queue.Dequeue(); $dir = $item[0]; $lvl = $item[1]
    try { $subs = @($dir.EnumerateDirectories()) } catch { continue }
    foreach ($s in $subs) {
        if ($s.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
        if ($s.Name -in @('$RECYCLE.BIN', 'System Volume Information')) { continue }
        if ($expect.Contains($s.Name) -or $s.Name -in @('probe', 'alipay-ai-data', 'download2')) {
            if (-not $found.ContainsKey($s.Name)) { $found[$s.Name] = New-Object System.Collections.Generic.List[string] }
            $found[$s.Name].Add($s.FullName)
        }
        # do not descend into our own data folders (huge, and nothing of ours is nested inside them)
        if ($lvl + 1 -lt $MaxDepth -and -not $expect.Contains($s.Name)) { $queue.Enqueue(@($s, ($lvl + 1))) }
    }
}
function Count-Tree([string]$p) {
    $n = 0; $b = [int64]0
    try { foreach ($f in (New-Object IO.DirectoryInfo $p).EnumerateFiles('*', [IO.SearchOption]::AllDirectories)) { $n++; $b += $f.Length } } catch { return @(-1, -1) }
    return @($n, $b)
}
foreach ($k in $expect.Keys) {
    if (-not $found.ContainsKey($k)) { '  {0,-30} NOT FOUND on {1}' -f $k, $E; continue }
    foreach ($p in $found[$k]) {
        $c = Count-Tree $p
        $want = $expect[$k]
        $verdict = if ($c[0] -eq $want) { 'MATCH' } elseif ($c[0] -gt $want) { 'MORE (changed after 9/29?)' } else { 'FEWER - check' }
        '  {0,-30} {1,-62} files {2,9:N0} (9/29: {3,9:N0})  {4,7:N2} GB  {5}' -f $k, $p, $c[0], $want, ($c[1] / 1GB), $verdict
    }
}
if ($found.ContainsKey('probe')) {
    foreach ($p in $found['probe']) { $c = Count-Tree $p; '  {0,-30} {1,-62} files {2,9:N0}  (D:\probe had 522 that E:\SSP_Work\probe lacked)' -f 'probe', $p, $c[0] }
}

Write-Host "`n==== 4. our loose files (looked for in every folder named alipay-ai-data on E:) ===="
$adirs = @(); if ($found.ContainsKey('alipay-ai-data')) { $adirs = @($found['alipay-ai-data']) }
if (-not $adirs.Count) { '  no folder named alipay-ai-data on ' + $E }
foreach ($a in $adirs) {
    '  -- ' + $a
    foreach ($k in $looseExpect.Keys) {
        $f = Join-Path $a $k
        if (Test-Path -LiteralPath $f) {
            $len = (Get-Item -LiteralPath $f).Length
            $v = if ($len -eq $looseExpect[$k]) { 'MATCH' } else { 'SIZE DIFFERS' }
            '     {0,-22} {1,14:N0} bytes (9/29: {2,14:N0})  {3}' -f $k, $len, $looseExpect[$k], $v
        } else { '     {0,-22} not here' -f $k }
    }
}

Write-Host "`n==== 4b. loose files in every folder named download2 on E: (pinyin_full.csv = the 9/14 white probe scan record) ===="
$ddirs = @(); if ($found.ContainsKey('download2')) { $ddirs = @($found['download2']) }
if (-not $ddirs.Count) { '  no folder named download2 on ' + $E }
foreach ($dd in $ddirs) {
    '  -- ' + $dd
    Get-ChildItem -LiteralPath $dd -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -notmatch '^\.(dll|exe|pdb|xml|config)$' } |
        ForEach-Object { '     {0,-40} {1,14:N0} bytes  {2:yyyy-MM-dd HH:mm}' -f $_.Name, $_.Length, $_.LastWriteTime }
}

Write-Host "`n==== 5. shared things that should still be on D: (not ours) ===="
foreach ($p in @('Python', 'alipay-ai-data\delivery', 'alipay-ai-data\alipay-ai-inference', 'alipay-ai-data\receipt-lite-teacher-120k-v1',
                 'f3-pp-final-v4-src-20260818-r2', 'receipt-test')) {
    $full = Join-Path $D $p
    '  {0,-55} exists {1}' -f $full, (Test-Path -LiteralPath $full)
}

Write-Host "`n==== 6. the July-August library on E: ===="
foreach ($n in @('BlueImages', 'OtherImages')) {
    $p = Join-Path $E $n
    if (Test-Path -LiteralPath $p) { $cnt = 0; foreach ($f in (New-Object IO.DirectoryInfo $p).EnumerateFiles()) { $cnt++ }; '  {0,-20} files {1,10:N0}   (9/24: 487,045 blue / 873,045 other)' -f $p, $cnt }
    else { '  ' + $p + ' NOT FOUND' }
}
Write-Host "`nDone. Nothing was changed."
