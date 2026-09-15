# 诊断前的体检。**只读**, 不动交付包, 不建目录, 不跑模型。
#
# 为什么要这个: 那条诊断要跑半天。批跑脚本本体不在 git 里, 参数名是从交付文档
# 抄的 —— 文档和实际对不上的话, 跑四个小时才发现就太亏了。
# 这个脚本先把该在的东西挨个核一遍。
#
#   powershell -ExecutionPolicy Bypass -File preflight_diag.ps1 `
#       -Package  交付包目录 `
#       -Pinyin   拼音图目录 `
#       -Pool     原图库目录 `
#       -WorkRoot 诊断工作目录
#
# 全绿才动手。有红的先解决红的。
#
# ★ 路径一律从命令行传, 脚本里不写死 —— 这是个公开仓库, 内部路径不进代码。

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Package,
    [Parameter(Mandatory)] [string] $Pinyin,
    [Parameter(Mandatory)] [string] $Pool,
    [Parameter(Mandatory)] [string] $WorkRoot,
    [string] $RunnerPattern = '*batch*.ps1',
    [int]    $NeedGB        = 5
)

$ErrorActionPreference = 'Continue'
$script:bad = 0
$script:warn = 0

function Ok   ($m) { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Bad  ($m) { Write-Host "  [缺]   $m" -ForegroundColor Red;    $script:bad++ }
function Warn ($m) { Write-Host "  [注意] $m" -ForegroundColor Yellow; $script:warn++ }
function Head ($m) { Write-Host ""; Write-Host "=== $m ===" -ForegroundColor Cyan }

Head '一 交付包在不在'
if (Test-Path -LiteralPath $Package) {
    Ok "交付包: $Package"
    $sums = Get-ChildItem -LiteralPath $Package -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '(?i)sha.*sum|checksum|manifest' }
    if ($sums) {
        Warn "有校验清单($($sums[0].Name)) —— 交付包是校验封闭的, 千万别往里加文件或改文件"
    }
    Get-ChildItem -LiteralPath $Package -File -ErrorAction SilentlyContinue |
        Select-Object -First 15 Name,
            @{n='MB';e={[math]::Round($_.Length/1MB,2)}} |
        Format-Table -AutoSize
} else {
    Bad "交付包不在: $Package"
}

Head '二 批跑脚本, 以及它真正接受的参数名'
# ★ 不写死脚本名, 按模式找 —— 名字本来就不确定, 搜比猜稳
$found = Get-ChildItem -LiteralPath $Package -Filter $RunnerPattern -Recurse -ErrorAction SilentlyContinue |
         Select-Object -First 1
$runner = if ($found) { $found.FullName } else { $null }
if ($runner) {
    Ok "批跑脚本: $runner"
    Write-Host "  --- 实际的 param 块(不要信文档, 信这里) ---" -ForegroundColor Gray
    $txt = Get-Content -LiteralPath $runner -Raw -Encoding UTF8
    # 把第一个 param(...) 抠出来
    $m = [regex]::Match($txt, '(?s)param\s*\((.*?)\n\s*\)')
    if ($m.Success) {
        $m.Groups[1].Value -split "`n" |
            Where-Object { $_ -match '\$' } |
            ForEach-Object { Write-Host "      $($_.Trim())" }
    } else {
        Warn "没解析出 param 块, 手动看一眼前 60 行"
        Get-Content -LiteralPath $runner -TotalCount 60 | ForEach-Object { Write-Host "      $_" }
    }

    foreach ($p in @('InputDirectory','OutputDirectory','Workers','MinimumThroughput')) {
        if ($txt -match [regex]::Escape($p)) { Ok "参数 -$p 存在" }
        else { Bad "参数 -$p 在脚本里**找不到** —— 文档和实际对不上, 别照抄我给的命令" }
    }
} else {
    Bad "按模式 $RunnerPattern 没找到批跑脚本"
    Write-Host "  包里有这些 ps1:" -ForegroundColor Gray
    Get-ChildItem -LiteralPath $Package -Filter *.ps1 -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 10 FullName | Format-Table -AutoSize
}

Head '三 单张模式(先拿一张试, 别一上来跑 200 张)'
$single = Get-ChildItem -LiteralPath $Package -Filter '*single*.ps1' -Recurse -ErrorAction SilentlyContinue |
          Select-Object -First 1
if ($single) { Ok "有单张脚本: $($single.FullName)" }
else { Warn "没找到单张脚本, 只能用批跑模式拿一张图的目录试" }

Head '四 拼音图和对照图库'
foreach ($pair in @(@($Pinyin,'拼音图'), @($Pool,'原图库(对照组从这里抽)'))) {
    $d = $pair[0]; $label = $pair[1]
    if (Test-Path -LiteralPath $d) {
        $n = (Get-ChildItem -LiteralPath $d -File -ErrorAction SilentlyContinue |
              Where-Object { $_.Extension -match '^\.(jpg|jpeg|png)$' }).Count
        if ($n -gt 0) { Ok "$label : $d  ($n 张)" }
        else { Bad "$label 目录在但**一张图都没有**: $d" }
    } else {
        Bad "$label 不在: $d"
    }
}

Head '五 蓝白过滤脚本(必须先过滤, 蓝图没有订单号字段)'
$split = Join-Path $PSScriptRoot 'split_blue_white.py'
if (Test-Path -LiteralPath $split) {
    Ok "分类脚本: $split"
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        Ok "python: $($py.Source)"
        $probe = & python -c "import cv2,numpy;print('deps-ok')" 2>&1
        if ($probe -match 'deps-ok') { Ok "cv2 / numpy 都在" }
        else { Bad "cv2 或 numpy 缺: $probe" }
    } else { Bad "找不到 python" }
} else {
    Bad "分类脚本不在: $split  (先把仓库 git pull 下来)"
}

Head '六 磁盘空间'
$drive = (Split-Path -Qualifier $WorkRoot).TrimEnd(':')
$disk = Get-PSDrive -Name $drive -ErrorAction SilentlyContinue
if ($disk) {
    $freeGB = [math]::Round($disk.Free/1GB, 1)
    if ($freeGB -ge $NeedGB) { Ok "$drive 盘剩 $freeGB GB (需要约 $NeedGB GB)" }
    else { Bad "$drive 盘只剩 $freeGB GB, 不够 $NeedGB GB" }
} else { Warn "读不到 $drive 盘的剩余空间" }

Head '七 工作目录(应当还不存在, 免得和上一轮混在一起)'
if (Test-Path -LiteralPath $WorkRoot) {
    Warn "$WorkRoot 已存在。里面已有的:"
    Get-ChildItem -LiteralPath $WorkRoot -Directory -ErrorAction SilentlyContinue |
        Select-Object Name, LastWriteTime | Format-Table -AutoSize
    Warn "跑的时候用带时间戳的新子目录, 别覆盖旧的"
} else {
    Ok "$WorkRoot 还不存在, 跑的时候会新建"
}

Head '结论'
if ($script:bad -eq 0) {
    Write-Host "  没有缺的项。注意项 $($script:warn) 条。" -ForegroundColor Green
    Write-Host "  下一步: 先用**一张图**跑通整条链, 确认输出 JSON 里确实有订单号那个字段," -ForegroundColor Green
    Write-Host "         并记下它实际叫什么(后面分析器要用 --field 传), 再放到 200 张。" -ForegroundColor Green
} else {
    Write-Host "  有 $($script:bad) 项缺失, $($script:warn) 项注意。" -ForegroundColor Red
    Write-Host "  **先别跑那半天的诊断**, 把缺的补齐再说。" -ForegroundColor Red
}
Write-Host ""
