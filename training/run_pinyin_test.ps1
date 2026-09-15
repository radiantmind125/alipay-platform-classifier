# 拼音对比实验: 一条命令跑完, 每一阶段都卡点
#
# 同一批图、同一个二进制、同一个密封包, 三种预处理, 比结果:
#     A 原图     (基准)
#     B 擦完拼音
#     C 切片 -> 合并
#
# ★ 不训练任何模型, 不动交付包, 不改合同。对交付包**只读**。
#
# 用法(路径全部必填, 按你服务器上的实际情况填):
#
#   powershell -ExecutionPolicy Bypass -File run_pinyin_test.ps1 `
#       -Package  '交付包目录' `
#       -Pinyin   'D:\...\pinyin_hits' `
#       -Work     'D:\...\pytest'
#
# 中途失败了修好再跑, 用 -StartAt 跳过已经做完的阶段:
#   -StartAt 3     从第 3 阶段开始
#
# ★ 路径一律从命令行传, 脚本里不写死 —— 这是个公开仓库。

[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string] $Package,     # 交付包目录
    [Parameter(Mandatory)] [string] $Pinyin,      # 拼音图目录
    [Parameter(Mandatory)] [string] $Work,        # 实验工作目录(会新建)
    [int]    $Count         = 100,                # 每组多少张
    [Parameter(Mandatory)] [int] $Limit,          # 检测最长边上限, 按合同里的值填
    [string] $RunnerPattern = '*batch*.ps1',
    [string] $InArg         = 'InputDirectory',   # 批跑脚本的入参名
    [string] $OutArg        = 'OutputDirectory',
    [int]    $StartAt       = 0
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$script:stamp = $null

function Say  ($m) { Write-Host $m -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "  [OK]  $m" -ForegroundColor Green }
function Die  ($m) { Write-Host "  [停]  $m" -ForegroundColor Red; exit 1 }
function Note ($m) { Write-Host "  [注意] $m" -ForegroundColor Yellow }

function Find-Runner {
    $r = Get-ChildItem -LiteralPath $Package -Filter $RunnerPattern -Recurse -ErrorAction SilentlyContinue |
         Select-Object -First 1
    if (-not $r) {
        Write-Host "  包里的 ps1:" -ForegroundColor Gray
        Get-ChildItem -LiteralPath $Package -Filter *.ps1 -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 10 FullName | Format-Table -AutoSize
        Die "按模式 $RunnerPattern 没找到批跑脚本"
    }
    return $r.FullName
}

function Invoke-Ocr ($runner, $inDir, $outDir, $tag) {
    New-Item -ItemType Directory -Force $outDir | Out-Null
    $n = (Get-ChildItem -LiteralPath $inDir -File).Count
    Say "  跑 $tag  ($n 张)"
    $t0 = Get-Date
    & powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass `
        -File $runner "-$InArg" $inDir "-$OutArg" $outDir
    $rc = $LASTEXITCODE
    $sec = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $rate = if ($sec -gt 0) { [math]::Round($n / $sec, 2) } else { 0 }
    $jsons = (Get-ChildItem -LiteralPath $outDir -Recurse -File -ErrorAction SilentlyContinue |
              Where-Object { $_.Extension -eq '.json' }).Count
    Write-Host ("    退出码 {0}   用时 {1} 秒   {2} 张/秒   生成 JSON {3} 份" -f $rc, $sec, $rate, $jsons)
    if ($jsons -eq 0) { Die "$tag 一份 JSON 都没生成。把上面的输出发我。" }
    if ($rc -ne 0) { Note "退出码非零 —— 多半只是吞吐没达标, JSON 已经生成了, 继续" }
    return [pscustomobject]@{ tag=$tag; sec=$sec; rate=$rate; n=$n; json=$jsons }
}

# ---------------------------------------------------------------- 阶段 0
if ($StartAt -le 0) {
    Say "`n=== 阶段 0  体检 ==="
    foreach ($p in @($Package, $Pinyin)) {
        if (-not (Test-Path -LiteralPath $p)) { Die "不在: $p" }
    }
    Ok "交付包: $Package"
    Ok "拼音图: $Pinyin  ($((Get-ChildItem -LiteralPath $Pinyin -File).Count) 个文件)"

    $runner = Find-Runner
    Ok "批跑脚本: $runner"
    Write-Host "  --- 它真正的 param 块(参数名以这里为准) ---" -ForegroundColor Gray
    $txt = Get-Content -LiteralPath $runner -Raw -Encoding UTF8
    $m = [regex]::Match($txt, '(?s)param\s*\((.*?)\n\s*\)')
    if ($m.Success) {
        $m.Groups[1].Value -split "`n" | Where-Object { $_ -match '\$' } |
            ForEach-Object { Write-Host "      $($_.Trim())" }
    } else { Note "没解析出 param 块" }
    foreach ($nm in @($InArg, $OutArg)) {
        if ($txt -match [regex]::Escape($nm)) { Ok "参数 -$nm 存在" }
        else { Die "参数 -$nm 在脚本里找不到。看上面的 param 块, 用 -InArg / -OutArg 传对的名字。" }
    }

    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Die "找不到 python" }
    $probe = & python -c "import cv2,numpy;print('ok')" 2>&1
    if ("$probe" -notmatch 'ok') { Die "python 缺 cv2 或 numpy: $probe" }
    Ok "python + cv2 + numpy"
    foreach ($s in @('split_blue_white.py','erase_pinyin.py','tile_for_ocr.py',
                     'merge_tiles.py','compare_runs.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repo "training\$s"))) { Die "缺脚本: $s (先 git pull)" }
    }
    Ok "五个脚本都在"
}

# ---------------------------------------------------------------- 阶段 1
if ($StartAt -le 1) {
    Say "`n=== 阶段 1  先跑一张, 看输出长什么样 ==="
    $runner = Find-Runner
    $one    = Join-Path $Work 'phase1_in'
    $oneOut = Join-Path $Work 'phase1_out'
    New-Item -ItemType Directory -Force $one | Out-Null
    Get-ChildItem -LiteralPath $one -File | Remove-Item -Force -ErrorAction SilentlyContinue
    $pick = Get-ChildItem -LiteralPath $Pinyin -File |
            Where-Object { $_.Extension -in '.jpg','.jpeg','.png' } |
            Get-Random -Count 1
    if (-not $pick) { Die "$Pinyin 里没有图片" }
    Copy-Item -LiteralPath $pick.FullName -Destination (Join-Path $one $pick.Name)
    Ok "挑中 $($pick.Name)"

    $null = Invoke-Ocr $runner $one $oneOut '一张'

    $j = Get-ChildItem -LiteralPath $oneOut -Recurse -File |
         Where-Object { $_.Extension -eq '.json' -and $_.Name -notlike 'batch*' -and $_.Name -notlike 'worker*' } |
         Select-Object -First 1
    if (-not $j) { Die "没有单张结果 JSON, 只有批处理产物。把 $oneOut 里的文件名发我。" }
    Write-Host "`n  ===== 这份 JSON 请完整发我 =====" -ForegroundColor Magenta
    Get-Content -LiteralPath $j.FullName -Raw -Encoding UTF8
    Write-Host "  ===== 以上 =====`n" -ForegroundColor Magenta
    Note "确认里面有没有订单号那个字段、它叫什么、值是完整还是空的"
}

# ---------------------------------------------------------------- 阶段 2
if ($StartAt -le 2) {
    Say "`n=== 阶段 2  准备三组输入 ==="
    $script:stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Set-Content -LiteralPath (Join-Path $Work 'stamp.txt') -Value $script:stamp -Encoding ascii
    Ok "时间戳 $($script:stamp)"

    $white = Join-Path $Work "white-$($script:stamp)"
    & python (Join-Path $repo 'training\split_blue_white.py') `
        --src $Pinyin --out (Join-Path $Work "split-$($script:stamp).csv") --link-white $white
    $nw = (Get-ChildItem -LiteralPath $white -File -ErrorAction SilentlyContinue).Count
    if ($nw -lt $Count) { Die "白图只有 $nw 张, 不够 $Count 张。把上面的蓝白比例发我。" }
    Ok "白图 $nw 张"

    $inA = Join-Path $Work "A-$($script:stamp)"
    New-Item -ItemType Directory -Force $inA | Out-Null
    Get-ChildItem -LiteralPath $white -File | Get-Random -Count $Count |
        ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $inA $_.Name) }
    Ok "A 组(原图) $((Get-ChildItem -LiteralPath $inA -File).Count) 张"

    & python (Join-Path $repo 'training\erase_pinyin.py') --src $inA --out (Join-Path $Work "B-$($script:stamp)")
    Ok "B 组(擦完) $((Get-ChildItem -LiteralPath (Join-Path $Work "B-$($script:stamp)") -File).Count) 个文件"

    & python (Join-Path $repo 'training\tile_for_ocr.py') --src $inA --out (Join-Path $Work "C-$($script:stamp)") --limit $Limit
    Ok "C 组(切片) 完成"
    Note "打开 B、C 各看一两张, 确认没把汉字啃坏、没把某一行整个切没"
}

# ---------------------------------------------------------------- 阶段 3
if ($StartAt -le 3) {
    Say "`n=== 阶段 3  三组各跑一遍 OCR ==="
    if (-not $script:stamp) { $script:stamp = (Get-Content -LiteralPath (Join-Path $Work 'stamp.txt') -Raw).Trim() }
    $runner = Find-Runner
    $timing = @()
    foreach ($tag in @('A','B','C')) {
        $inDir = Join-Path $Work "$tag-$($script:stamp)"
        if (-not (Test-Path -LiteralPath $inDir)) { Die "输入不在: $inDir (先跑阶段 2)" }
        $timing += Invoke-Ocr $runner $inDir (Join-Path $Work "out-$tag-$($script:stamp)") $tag
    }
    Write-Host "`n  ===== 用时请发我 =====" -ForegroundColor Magenta
    $timing | Format-Table tag, n, sec, rate, json -AutoSize
    Write-Host "  ===== 以上 =====" -ForegroundColor Magenta
    Note "C 组明显慢是预期内的, 那正是要量的代价"
}

# ---------------------------------------------------------------- 阶段 4
if ($StartAt -le 4) {
    Say "`n=== 阶段 4  把切片的结果合并 ==="
    if (-not $script:stamp) { $script:stamp = (Get-Content -LiteralPath (Join-Path $Work 'stamp.txt') -Raw).Trim() }
    & python (Join-Path $repo 'training\merge_tiles.py') `
        --tiles   (Join-Path $Work "C-$($script:stamp)") `
        --results (Join-Path $Work "out-C-$($script:stamp)") `
        --out     (Join-Path $Work "merged-C-$($script:stamp)")
    Note "冲突张数如果很多, 说明块边界切坏了行, 要调大 tile_for_ocr 的 --overlap 重来"
}

# ---------------------------------------------------------------- 阶段 5
if ($StartAt -le 5) {
    Say "`n=== 阶段 5  出结论 ==="
    if (-not $script:stamp) { $script:stamp = (Get-Content -LiteralPath (Join-Path $Work 'stamp.txt') -Raw).Trim() }
    Write-Host "`n  ===== 下面这一整段请发我 =====" -ForegroundColor Magenta
    & python (Join-Path $repo 'training\compare_runs.py') `
        --run "原图=$(Join-Path $Work "out-A-$($script:stamp)")" `
        --run "擦完=$(Join-Path $Work "out-B-$($script:stamp)")" `
        --run "切片=$(Join-Path $Work "merged-C-$($script:stamp)")"
    Write-Host "  ===== 以上 =====" -ForegroundColor Magenta
}

Say "`n跑完了。时间戳 $($script:stamp)"
Write-Host "要发我三样: 阶段1的那份 JSON / 阶段3的用时表 / 阶段5的对比表" -ForegroundColor Green
