# 非拼音对照组: 拿**普通**回单跑一遍, 和拼音组比字段填充率。
#
# 为什么必须做这个
# ----------------
# 拼音组实测 100 张: device 100%, status_bar_time 84%,
# 但所有要读回单正文的字段都是 0-4%, 订单号 **0/100**。
#
# 光看这一组分不出两种完全不同的情况:
#     普通图也是 0-4%  ->  这个包**对谁都抽不出字段**, 跟拼音无关
#     普通图是 80%     ->  **拼音就是元凶**, 而且不是"不全"是"全没了"
#
# 两种结论对应的下一步完全相反, 所以这一组是**必须**的。
#
#   powershell -ExecutionPolicy Bypass -File control_group_test.ps1 `
#       -Package 交付包目录 -Pool 原图库目录 -Pinyin 拼音图目录 -Work 工作目录
#
# ★ 路径全部必填, 脚本里不写死(公开仓库)。

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Package,   # 交付包
    [Parameter(Mandatory)] [string] $Pool,      # 原图库(从这里抽普通图)
    [Parameter(Mandatory)] [string] $Pinyin,    # 拼音图目录(要排除掉)
    [Parameter(Mandatory)] [string] $Work,      # 工作目录
    [int] $Count = 100,
    [string] $InArg  = 'InputDirectory',
    [string] $OutArg = 'OutputDirectory'
)

$ErrorActionPreference = 'Stop'
function Say ($m) { Write-Host $m -ForegroundColor Cyan }
function Die ($m) { Write-Host "  [stop] $m" -ForegroundColor Red; exit 1 }

foreach ($p in @($Package, $Pool, $Pinyin)) {
    if (-not (Test-Path -LiteralPath $p)) { Die "not found: $p" }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$inDir = Join-Path $Work "CTRL-$stamp"
$outDir = Join-Path $Work "out-CTRL-$stamp"
New-Item -ItemType Directory -Force $inDir | Out-Null

Say "`n=== picking $Count non-pinyin images ==="
# 拼音图的文件名集合, 用来排除
$hit = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
Get-ChildItem -LiteralPath $Pinyin -File | ForEach-Object { $null = $hit.Add($_.Name) }
Write-Host "  pinyin set: $($hit.Count)"

$picked = Get-ChildItem -LiteralPath $Pool -Recurse -File -ErrorAction SilentlyContinue |
          Where-Object { $_.Extension -in '.jpg', '.jpeg', '.png' -and -not $hit.Contains($_.Name) } |
          Get-Random -Count $Count
if (-not $picked -or $picked.Count -lt 1) { Die "could not pick any non-pinyin image from $Pool" }
$picked | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $inDir $_.Name) }
Write-Host "  picked: $((Get-ChildItem -LiteralPath $inDir -File).Count)"

Say "`n=== running OCR ==="
$runner = (Get-ChildItem -LiteralPath $Package -Filter '*batch*.ps1' -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 1).FullName
if (-not $runner) { Die "batch runner not found under $Package" }
# 输出目录必须全新 —— 批跑脚本会拒绝已存在的
if (Test-Path -LiteralPath $outDir) { Remove-Item -LiteralPath $outDir -Recurse -Force }
$t0 = Get-Date
& powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
    -File $runner "-$InArg" $inDir "-$OutArg" $outDir
$sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
Write-Host "  exit=$LASTEXITCODE  wall=${sec}s"

Say "`n=== FIELD FILL COUNT (non-pinyin control) ==="
$files = Get-ChildItem -LiteralPath $outDir -Recurse -File -Filter *.json |
         Where-Object { $_.BaseName.Length -ge 40 }
if ($files.Count -eq 0) { Die "no per-image result JSON produced" }
$tally = @{}
$n = 0
foreach ($f in $files) {
    $o = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $n++
    foreach ($p in $o.PSObject.Properties) {
        if (-not $tally.ContainsKey($p.Name)) { $tally[$p.Name] = 0 }
        if ($null -ne $p.Value -and "$($p.Value)".Trim() -ne '') { $tally[$p.Name]++ }
    }
}
Write-Output "over $n images:"
$tally.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object {
    $pct = if ($n -gt 0) { [math]::Round(100 * $_.Value / $n, 1) } else { 0 }
    Write-Output ("  {0,-22} {1,4} / {2}   {3}" -f $_.Key, $_.Value, $n, $pct)
}
Write-Output ""
Write-Output "(last column is percent)"
Write-Output ""
Write-Output "COMPARE against the pinyin group:"
Write-Output "  pinyin:  device 100, status_bar_time 84, everything-else 0-4, voucher_number 0"
Write-Output "  if this control looks the SAME  -> the package extracts nothing for anyone"
Write-Output "  if this control is much HIGHER  -> pinyin really is the cause"
Say "`ndone. stamp $stamp"
