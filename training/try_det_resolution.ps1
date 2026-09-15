# 试试"整页 + 提高检测分辨率": 要切片的好处, 不要切片的坏处。
#
# 为什么值得试
# ------------
# 实测切片的结果分成两簇:
#     局部就能判的      transfer_status 1->56%  payment_method 4->50%  transfer_note 2->31%
#     要整页上下文的    voucher_number/transfer_time/amount 全被打成 0
#
# 切片的**好处来自分辨率**, **坏处来自拆页**。这两件事可以分开 ——
# 保持整页不拆, 只把检测那一步的最长边上限调高, 两簇都可能受益, 而且只跑一遍。
#
# ★★ 安全: 全程在**交付包的副本**上做, 一个字节都不动原包。
#    合同 JSON 本身没有内容哈希校验(模型和字典才有), 所以改这个数不会破坏加载。
#
#   powershell -ExecutionPolicy Bypass -File try_det_resolution.ps1 `
#       -Package 原包目录 -Images 要跑的图目录 -Work 工作目录 -NewLimit 1920
#
# ★ 路径全部必填, 脚本里不写死(公开仓库)。

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Package,    # 原交付包(只读)
    [Parameter(Mandatory)] [string] $Images,     # 要跑的图(用同一批 A 组)
    [Parameter(Mandatory)] [string] $Work,       # 工作目录
    [Parameter(Mandatory)] [int]    $NewLimit,   # 新的最长边上限
    [string] $InArg  = 'InputDirectory',
    [string] $OutArg = 'OutputDirectory'
)

$ErrorActionPreference = 'Stop'
function Say ($m) { Write-Host $m -ForegroundColor Cyan }
function Ok  ($m) { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Die ($m) { Write-Host "  [stop] $m" -ForegroundColor Red; exit 1 }

foreach ($p in @($Package, $Images)) {
    if (-not (Test-Path -LiteralPath $p)) { Die "not found: $p" }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$sandbox = Join-Path $Work "pkg-limit$NewLimit-$stamp"

Say "`n=== 1. copying the package (original is never touched) ==="
if (Test-Path -LiteralPath $sandbox) { Remove-Item -LiteralPath $sandbox -Recurse -Force }
Copy-Item -LiteralPath $Package -Destination $sandbox -Recurse
$mb = [math]::Round(((Get-ChildItem -LiteralPath $sandbox -Recurse -File |
        Measure-Object Length -Sum).Sum / 1MB), 1)
Ok "copy at $sandbox  ($mb MB)"

Say "`n=== 2. finding det_limit_side_len in the copy ==="
$cfgs = Get-ChildItem -LiteralPath $sandbox -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -eq '.json' -and $_.Length -lt 1MB }
$target = $null
foreach ($c in $cfgs) {
    $txt = Get-Content -LiteralPath $c.FullName -Raw -Encoding UTF8
    if ($txt -match '"det_limit_side_len"') { $target = $c; break }
}
if (-not $target) { Die "no json in the copy contains det_limit_side_len" }
Ok "contract: $($target.FullName)"

$txt = Get-Content -LiteralPath $target.FullName -Raw -Encoding UTF8
if ($txt -notmatch '"det_limit_side_len"\s*:\s*(\d+)') { Die "could not parse the current value" }
$oldLimit = [int]$Matches[1]
Write-Host "  current = $oldLimit   ->   new = $NewLimit"
if ($oldLimit -eq $NewLimit) { Die "same value, nothing to test" }

$new = $txt -replace '("det_limit_side_len"\s*:\s*)\d+', "`${1}$NewLimit"
# 写回时保持 UTF-8 无 BOM(JSON 一般不带 BOM)
[System.IO.File]::WriteAllText($target.FullName, $new, (New-Object System.Text.UTF8Encoding($false)))
$check = Get-Content -LiteralPath $target.FullName -Raw -Encoding UTF8
if ($check -notmatch "`"det_limit_side_len`"\s*:\s*$NewLimit") { Die "write-back failed" }
Ok "changed to $NewLimit in the COPY only"

Say "`n=== 3. confirming the original is untouched ==="
$origTxt = Get-Content -LiteralPath (Join-Path $Package $target.FullName.Substring($sandbox.Length + 1)) -Raw -Encoding UTF8
if ($origTxt -match "`"det_limit_side_len`"\s*:\s*$oldLimit") { Ok "original still says $oldLimit" }
else { Die "the original changed -- STOP and check" }

Say "`n=== 4. running the same images through the modified copy ==="
$runner = (Get-ChildItem -LiteralPath $sandbox -Filter '*batch*.ps1' -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $runner) { Die "batch runner not found in the copy" }
$outDir = Join-Path $Work "out-limit$NewLimit-$stamp"
if (Test-Path -LiteralPath $outDir) { Remove-Item -LiteralPath $outDir -Recurse -Force }
$n = (Get-ChildItem -LiteralPath $Images -File).Count
$t0 = Get-Date
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
    -File $runner "-$InArg" $Images "-$OutArg" $outDir
$rc = $LASTEXITCODE
$sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
$rate = if ($sec -gt 0) { [math]::Round($n / $sec, 2) } else { 0 }
Write-Host "  exit=$rc  wall=${sec}s  $n images  ${rate}/s"

Say "`n=== 5. FIELD FILL COUNT (limit=$NewLimit, whole page, no tiling) ==="
$files = Get-ChildItem -LiteralPath $outDir -Recurse -File -Filter *.json |
         Where-Object { $_.BaseName.Length -ge 40 }
if ($files.Count -eq 0) { Die "no per-image result JSON produced -- the copied package may not run" }
$tally = @{}
$cnt = 0
foreach ($f in $files) {
    $o = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $cnt++
    foreach ($p in $o.PSObject.Properties) {
        if (-not $tally.ContainsKey($p.Name)) { $tally[$p.Name] = 0 }
        if ($null -ne $p.Value -and "$($p.Value)".Trim() -ne '') { $tally[$p.Name]++ }
    }
}
Write-Output "over $cnt images:"
$tally.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
    $pct = if ($cnt -gt 0) { [math]::Round(100 * $_.Value / $cnt, 1) } else { 0 }
    Write-Output ("  {0,-22} {1,4} / {2}   {3}" -f $_.Key, $_.Value, $cnt, $pct)
}
Write-Output ""
Write-Output "COMPARE with what we already measured on the SAME 100 pinyin images:"
Write-Output "  field             orig  erased  tiled   ceiling(non-pinyin)"
Write-Output "  transfer_status      1      19     56     94"
Write-Output "  payment_method       4      36     50     91"
Write-Output "  transfer_note        2      10     31     44"
Write-Output "  amount               3       8      0     46"
Write-Output "  transfer_time        0       7      0     42"
Write-Output "  voucher_number       0       7      0     41"
Write-Output ""
Write-Output "WHAT TO LOOK FOR:"
Write-Output "  both clusters up      -> resolution alone was the cause; no tiling, no erasing needed"
Write-Output "  only cluster 1 up     -> the whole-page fields need something else"
Write-Output "  nothing changes       -> resolution was not the mechanism; tiling helped for another reason"
Write-Output ""
Write-Output "ALSO NOTE the throughput above -- a bigger limit costs CPU, and the"
Write-Output "acceptance bar is 1.0 images/second."
Say "`ndone. sandbox: $sandbox"
Write-Host "the original package was never modified." -ForegroundColor Green
