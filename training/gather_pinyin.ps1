# ★ 这个文件必须存成**带 BOM 的 UTF-8**。
# PowerShell 5.1 读 .ps1 默认按 ANSI 解, 没有 BOM 的话中文会变成乱码,
# 连带把字符串和括号解析错, 整个脚本跑不起来。改这个文件时注意别丢了 BOM。
# 在服务器上把带拼音的图归到一个目录。
#
# 用法(在仓库根目录下跑):
#     .\training\gather_pinyin.ps1              先随机抽 2 万张试跑, 几分钟
#     .\training\gather_pinyin.ps1 -Full        全量跑, 要几小时
#
# 默认**先试跑**, 免得一上来就占住机器几个小时才发现路径写错了。
#
# ★ 归集用的是**硬链接**, 不占额外磁盘, 原图一张不动(不移动不删除)。
#   硬链接不能跨盘, 所以归集目录自动放在和图库**同一个盘**上。

param(
    [string]$Data   = 'D:\download2\OtherImages',
    [string]$Repo   = '',
    [string]$Gather = '',
    [int]$Workers   = 0,          # 0 = 自动用 核数减一
    [switch]$Full                 # 不加就是试跑 2 万张
)

$ErrorActionPreference = 'Stop'

# ---- 仓库路径: 没给就按脚本自己的位置推 ----
if ($Repo -eq '') {
    $Repo = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
}
$Probe = Join-Path $Repo 'training\pinyin_probe.py'

# ---- 归集目录: 没给就放到图库**同一个盘**、图库旁边 ----
if ($Gather -eq '') {
    $Gather = Join-Path (Split-Path $Data -Parent) 'pinyin_hits'
}
$OutCsv = Join-Path (Split-Path $Data -Parent) 'pinyin_full.csv'

Write-Output '================ 先检查 ================'

if (-not (Test-Path $Probe))  { Write-Output "★ 找不到 $Probe , 先 git pull"; exit 1 }
if (-not (Test-Path $Data))   { Write-Output "★ 找不到图库 $Data , 用 -Data 指一下"; exit 1 }

$dataDrive   = Split-Path $Data   -Qualifier
$gatherDrive = Split-Path $Gather -Qualifier
Write-Output ("图库      {0}" -f $Data)
Write-Output ("归集目录  {0}" -f $Gather)
Write-Output ("脚本      {0}" -f $Probe)

if ($dataDrive -ne $gatherDrive) {
    Write-Output ''
    Write-Output ("★★ 归集目录在 {0} 盘, 图库在 {1} 盘 —— **硬链接不能跨盘**。" -f $gatherDrive, $dataDrive)
    Write-Output '   这样跑会退回成拷贝, 5000 多张要占 1.5 GB。'
    Write-Output '   想省空间就用 -Gather 指一个和图库同盘的目录。'
    Write-Output ''
}

# ---- 磁盘剩余 ----
Get-PSDrive -PSProvider FileSystem |
    Where-Object { $_.Name -eq $dataDrive.TrimEnd(':') -or $_.Name -eq $gatherDrive.TrimEnd(':') } |
    Select-Object Name,
        @{n='已用GB'; e={ [math]::Round($_.Used/1GB, 1) }},
        @{n='剩余GB'; e={ [math]::Round($_.Free/1GB, 1) }} |
    Format-Table -AutoSize

# ---- 图有多少张, 大概要跑多久 ----
Write-Output '数图片张数(大库可能要等一会)...'
$count = (Get-ChildItem $Data -Recurse -File -Include *.jpg,*.jpeg,*.png -ErrorAction SilentlyContinue).Count
$cores = [Environment]::ProcessorCount
$useWorkers = if ($Workers -gt 0) { $Workers } else { [Math]::Max(1, $cores - 1) }
$n = if ($Full) { $count } else { [Math]::Min(20000, $count) }
$hours = $n * 0.099 / $useWorkers / 3600

Write-Output ("图片 {0:N0} 张, 本机 {1} 核, 用 {2} 个进程" -f $count, $cores, $useWorkers)
Write-Output ("这次要扫 {0:N0} 张, 按单张 99 毫秒估, 大概 {1:N1} 小时" -f $n, $hours)
Write-Output ("按命中率 0.68% 估, 大概能挑出 {0:N0} 张" -f ($n * 0.0068))

if (-not $Full) {
    Write-Output ''
    Write-Output '>>> 现在是**试跑**(随机抽 2 万张)。确认没问题之后加 -Full 跑全量。'
}

Write-Output ''
Write-Output '================ 开始扫 ================'
$sw = [Diagnostics.Stopwatch]::StartNew()

$argv = @($Probe, $Data, '--workers', $useWorkers, '--out', $OutCsv, '--gather', $Gather)
if (-not $Full) { $argv += @('--sample', '20000') }

& python @argv
$code = $LASTEXITCODE
$sw.Stop()

Write-Output ''
Write-Output ("用时 {0:N1} 分钟, 退出码 {1}" -f $sw.Elapsed.TotalMinutes, $code)
if ($code -ne 0) { Write-Output '★ 没跑成, 看上面的报错'; exit $code }

if (Test-Path $Gather) {
    $g = Get-ChildItem $Gather -File | Where-Object { $_.Name -ne '_manifest.csv' }
    Write-Output ("归集目录里 {0:N0} 张图" -f $g.Count)
    Write-Output ("清单 {0}" -f (Join-Path $Gather '_manifest.csv'))
    # 抽一张验证确实是硬链接
    if ($g.Count -gt 0) {
        $names = fsutil hardlink list $g[0].FullName 2>$null
        $cnt = ($names | Measure-Object).Count
        if ($cnt -ge 2) {
            Write-Output ("★ 确认是硬链接(同一份数据有 {0} 个名字), 没占额外磁盘, 原图没动。" -f $cnt)
        } else {
            Write-Output '★ 是拷贝不是硬链接 —— 多半是归集目录和图库不在同一个盘。'
        }
    }
}
Write-Output ("CSV {0}" -f $OutCsv)
